"""Run the real ISA v1 prefill program on Qwen3.5-0.8B in the functional simulator and check it against the golden
model computing with the same quantized weights (note 31, part 2).

  python3 isa/run_fsim_prefill.py [--chunks 2] [--layers 24] [--decode 6] [--floor]

- The prompt (chunks x 128 tokens) goes through build_prefill(dn="lite", fused=True, cfg=QWEN_08B).
- Per layer and prompt chunk: the chip's residual after the mixer and after the FFN (SRAM, read when the
  out-/o-projection and the FFN-down commands land) against the golden model run chunk by chunk.
- Each sublayer in isolation: the golden sublayer fed the chip's own input and the chip's own state (DeltaNet state,
  conv history, KV cache, read from the chip's DRAM when the chunk starts).
- --floor: the golden-vs-golden yardstick: the same golden model with two harmless fp32 reorderings (attention
  tiles of 1,024 positions instead of 256; the DeltaNet's cumulative decay by a running sum instead of the scan).
- --decode N: hand the chip's KV cache, DeltaNet state and conv history to the decode program (qwen_v1) at
  position chunks x 128, feed the rest of the prompt one token per step, then generate; N decode steps in all.
"""
import argparse
import copy
import dataclasses
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "golden"))
import dn_chunked  # noqa: E402
import formats  # noqa: E402
import pack_v1 as P  # noqa: E402
import qwen35_golden as G  # noqa: E402
import qwen_prefill_v1 as QP  # noqa: E402
import qwen_v1 as Q  # noqa: E402
from fsim_v1 import FSim  # noqa: E402
from sim_v1 import SimChip  # noqa: E402

MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
CACHE = os.path.join(HERE, "..", "golden", "results", "q08b_chip_weights.pt")
RES = os.path.join(HERE, "results")

TEXT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined "
        "movable metal type, oil-based ink and a wooden screw press into a system that could produce books far "
        "faster than scribes. Within fifty years, presses were running in more than two hundred European cities, "
        "and millions of volumes had been printed. The effects were enormous. Ideas travelled further and faster; "
        "scholars could compare identical copies of a text; vernacular languages gained standard spellings; and "
        "the Reformation spread through pamphlets that were cheap enough for ordinary people to buy. Later "
        "improvements followed one another: the cylinder press in the nineteenth century, steam power, the "
        "rotary press fed from continuous rolls of paper, and finally the Linotype machine, which cast whole "
        "lines of type at once. Each step lowered the cost of a printed page and widened the audience for news, "
        "science and literature. Today the same story is often told about the internet, which lowered the cost "
        "of publishing to almost nothing. Historians caution, however, that the comparison hides important "
        "differences. A printed book was fixed once it left the press, while a web page can change at any "
        "moment. A book had a publisher who could be held responsible, while online authorship is often "
        "anonymous. And the printing press needed centuries to reach most of the world's population, while the "
        "internet reached billions of people within a few decades. Still, both technologies show the same "
        "pattern: when copying becomes cheap, the amount of text explodes, old gatekeepers lose power, and "
        "societies have to invent new ways of deciding what to trust. The capital of France is Paris, and the "
        "capital of Japan is")


def golden_model(raw, qw, cfg, n_layers):
    g = object.__new__(G.Golden)
    g.cfg = G.Config(MODEL)
    g.cfg.layers = n_layers
    g.cfg.layer_types = g.cfg.layer_types[:n_layers]
    g.dev = "cpu"
    g.num = G.Numerics(act=formats.CHIP_ACT, kv="int8_head_chip", state="chunked_lite", attn_q=formats.CHIP_ACT,
                       attn_p=formats.CHIP_ACT)
    g.W = P.golden_weights(raw, qw, cfg)
    G.CHIP.update(sfu=True, rope=True, softmax=True, dnq=True)
    return g


def set_order(g, tile, scan):
    """The two fp32 orders the yardstick varies: attention tile (positions) and how G is accumulated."""
    orig = G.Golden._tiled_attention
    g._tiled_attention = lambda sc, Vh, tile_=tile: orig(g, sc, Vh, tile=tile_)
    dn_chunked.scan_hs = SCAN if scan else (lambda x: torch.cumsum(x, -1))


SCAN = dn_chunked.scan_hs


def rel(a, b):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def chip_cache(g, cfg, info, dram, L):
    """A golden Cache holding the chip's state at the start of a prompt chunk (position L)."""
    cache = g.new_cache()
    cache.pos = L
    db = info["dbase"]
    QKV_CH, KV_LAYER, lmax = info["QKV_CH"], info["KV_LAYER"], info["lmax"]
    SH = cfg.ld * cfg.ld * 4
    KH = lmax * cfg.hd
    dn = at = 0
    for i, t in enumerate(g.cfg.layer_types):
        if t == "full_attention":
            base = db["KVB"] + at * KV_LAYER
            ks = np.zeros((L, cfg.kvh, cfg.hd), np.float32)
            vs = np.zeros((L, cfg.kvh, cfg.hd), np.float32)
            for h in range(cfg.kvh):
                k8 = dram[base + h * KH:base + h * KH + L * cfg.hd].view(np.int8).reshape(L, cfg.hd)
                sc = dram[base + 2 * cfg.kvh * KH + h * 4 * lmax:][:4 * lmax].view(np.float16).reshape(2, lmax)
                vt = dram[base + cfg.kvh * KH + h * KH:][:KH].view(np.int8).reshape(lmax // 256, cfg.hd, 256)
                v8 = vt.transpose(0, 2, 1).reshape(lmax, cfg.hd)[:L]
                ks[:, h] = k8 * sc[0, :L, None].astype(np.float32)
                vs[:, h] = v8 * sc[1, :L, None].astype(np.float32)
            cache.k[i], cache.v[i] = torch.from_numpy(ks), torch.from_numpy(vs)
            at += 1
        else:
            st = dram[db["STB"] + dn * cfg.lv * SH:][:cfg.lv * SH].view(np.float32).reshape(cfg.lv, cfg.ld, cfg.ld)
            cache.S[i] = torch.from_numpy(st.copy())
            cv = dram[db["CLOG"] + dn * 3 * QKV_CH * 4:][:3 * QKV_CH * 4].view(np.float32).reshape(3, QKV_CH)
            cache.conv[i] = torch.from_numpy(cv.copy())
            dn += 1
    return cache


def golden_sublayers(g, i, h_in, cache, positions):
    """One layer of the golden model from a given input: (mixer output, FFN input residual, FFN output)."""
    t = g.cfg.layer_types[i]
    x = g.rms(h_in, f"layers.{i}.input_layernorm.weight")
    mx = g.full_attention(i, x, cache, positions) if t == "full_attention" else g.deltanet(i, x, cache)
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(os.cpu_count())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    ids = tok.encode(TEXT).ids
    NP_ = args.chunks * 128
    assert len(ids) >= NP_, len(ids)
    prompt = ids[:NP_]
    tail = ids[NP_:]
    raw = G.load_tensors(MODEL, "cpu")
    cfg = dataclasses.replace(Q.QWEN_08B, layers=args.layers)
    qw = P.quantize_all(raw, Q.QWEN_08B, CACHE)
    lmax = 1024
    prog, info = QP.build_prefill(chunks=args.chunks, lmax=lmax, fused=True, dn="lite", cfg=cfg)
    bundles = prog.assemble()
    dram, sram = P.build_images_prefill(raw, qw, info, prompt)
    H = info["bufs"]["H"]
    HID, NL = cfg.hid, cfg.layers
    print(f"prompt {len(ids)} tokens ({NP_} prefill, {len(tail)} after); program {len(bundles)} bundles; images "
          f"{dram.nbytes / 2**20:.0f} MB DRAM ({time.time() - t0:.0f} s)", flush=True)

    snaps = {}                 # (chunk, layer, 'mix' | 'ffn' | 'in') -> residual [128, HID]
    states = {}                # chunk -> DRAM copy at the start of that chunk (state regions only matter)
    count = {"ffn": 0, "mix": 0}

    def hcopy(sim):
        return sim.sram[H:H + 128 * HID * 4].view(np.float32).reshape(128, HID).copy()

    first_in = {}

    def on_land(sim, c):
        if c.unit == "V" and c.op == "DEQ":                  # a prompt chunk starts: its embedding, the chip's state
            ch = len(first_in)
            first_in[ch] = hcopy(sim)
            states[ch] = sim.dram.copy() if ch > 0 else None
        if c.unit == "M" and c.note in ("out-projection", "o-projection"):
            n = count["mix"]
            snaps[(n // NL, n % NL, "mix")] = hcopy(sim)
            count["mix"] += 1
        if c.unit == "M" and c.note == "FFN down":
            n = count["ffn"]
            snaps[(n // NL, n % NL, "ffn")] = hcopy(sim)
            count["ffn"] += 1
            print(f"  chip chunk {n // NL} layer {n % NL} done  t={time.time() - t0:.0f}s", flush=True)

    sim = FSim(prog, bundles, sram, dram, chip=SimChip(vec_lanes=512), regs={Q.X_L: 0})
    sim.on_land = on_land
    st = sim.run()
    print(f"chip prefill done: {st['time_ns'] / 1e3:.0f} us simulated, {time.time() - t0:.0f} s wall", flush=True)
    chip_dram_end = sim.dram

    # ---------------------------------------------------------------- golden, chunk by chunk
    g = golden_model(raw, qw, Q.QWEN_08B, NL)
    set_order(g, 256, True)

    def golden_run(gm):
        cache = gm.new_cache()
        out = []
        for ch in range(args.chunks):
            trace = []
            gm.forward(torch.tensor(prompt[ch * 128:(ch + 1) * 128]), cache, trace=trace, return_hidden=True)
            out.append(trace)
        return out, cache
    gold, gcache = golden_run(g)
    rows = []
    for ch in range(args.chunks):
        for li in range(NL):
            rows.append(dict(chunk=ch, layer=li, kind=g.cfg.layer_types[li],
                             residual_rel=rel(snaps[(ch, li, "ffn")], gold[ch][li + 1])))
    emb_rel = [rel(first_in[ch], gold[ch][0]) for ch in range(args.chunks)]
    print("embedding rel. diff per chunk:", [f"{x:.1e}" for x in emb_rel])
    for r in rows:
        if r["layer"] in (0, 1, 2, 3, NL // 2, NL - 1):
            print(f"  chunk {r['chunk']} layer {r['layer']:2d} ({r['kind'][:4]}): residual rel. diff {r['residual_rel']:.2e}")

    # ---------------------------------------------------------------- each sublayer alone (chip input, chip state)
    iso = []
    for ch in range(args.chunks):
        L = ch * 128
        cache = chip_cache(g, cfg, info, states[ch], L) if ch > 0 else g.new_cache()
        positions = torch.arange(L, L + 128)
        for li in range(NL):
            h_in = torch.from_numpy(first_in[ch] if li == 0 else snaps[(ch, li - 1, "ffn")])
            h_mix = torch.from_numpy(snaps[(ch, li, "mix")])
            h_out = torch.from_numpy(snaps[(ch, li, "ffn")])
            mx = golden_sublayers(g, li, h_in, cache, positions)
            f = g.ffn(li, g.rms(h_mix, f"layers.{li}.post_attention_layernorm.weight"))
            iso.append(dict(chunk=ch, layer=li, kind=g.cfg.layer_types[li],
                            mixer_out_rel=rel(h_mix - h_in, mx), ffn_out_rel=rel(h_out - h_mix, f),
                            mixer_resid_rel=rel(h_mix, h_in + mx), ffn_resid_rel=rel(h_out, h_mix + f)))
            r = iso[-1]
            print(f"  alone: chunk {ch} layer {li:2d} ({r['kind'][:4]}): mixer out {r['mixer_out_rel']:.1e}, "
                  f"FFN out {r['ffn_out_rel']:.1e}  (residual {r['mixer_resid_rel']:.1e} / {r['ffn_resid_rel']:.1e})",
                  flush=True)

    # ---------------------------------------------------------------- next-token argmax at the end of the prompt
    def logits_of(hrow):
        return g.mm(g.rms(hrow[None], "norm.weight"), "lm_head.weight")[0]
    lc = logits_of(torch.from_numpy(snaps[(args.chunks - 1, NL - 1, "ffn")][-1]))
    lg = logits_of(gold[-1][-1][-1])
    top2 = torch.topk(lg, 2).values
    am = dict(chip=int(lc.argmax()), golden=int(lg.argmax()), golden_margin=float(top2[0] - top2[1]),
              chip_tok=tok.decode([int(lc.argmax())]), golden_tok=tok.decode([int(lg.argmax())]),
              logits_rel=rel(lc, lg))
    print(f"next token after the prompt: chip {am['chip']} ({am['chip_tok']!r}), golden {am['golden']} "
          f"({am['golden_tok']!r}), margin {am['golden_margin']:.3f}; logits rel. diff {am['logits_rel']:.2e}")

    out = dict(model="Qwen3.5-0.8B", layers=NL, chunks=args.chunks, prompt=prompt, bundles=len(bundles),
               sim_us=st["time_ns"] / 1e3, embedding_rel=emb_rel, rows=rows, alone=iso, argmax=am,
               commands=sorted(((f"{u} {o}", n) for (u, o), n in sim.counts.items()), key=lambda x: -x[1]))

    # ---------------------------------------------------------------- golden-vs-golden yardstick
    if args.floor:
        g2 = golden_model(raw, qw, Q.QWEN_08B, NL)
        set_order(g2, 1024, False)
        gold2, _ = golden_run(g2)
        set_order(g, 256, True)
        out["floor"] = [dict(chunk=ch, layer=li, residual_rel=rel(gold2[ch][li + 1], gold[ch][li + 1]))
                        for ch in range(args.chunks) for li in range(NL)]
        l2 = logits_of(gold2[-1][-1][-1])
        out["floor_argmax"] = dict(golden2=int(l2.argmax()), logits_rel=rel(l2, lg))
        for r in out["floor"]:
            if r["layer"] in (0, NL // 2, NL - 1):
                print(f"  yardstick: chunk {r['chunk']} layer {r['layer']:2d}: {r['residual_rel']:.2e}")
        set_order(g, 256, True)

    # ---------------------------------------------------------------- hand the state to the decode program
    if args.decode:
        n = args.decode
        L0 = NP_
        dprog, dinfo = Q.build("decode", steps=n, lmax=lmax, cfg=cfg)
        dbund = dprog.assemble()
        ddram, dsram = P.build_images(raw, qw, dinfo)
        db, pb = dinfo["dbase"], info["dbase"]
        assert dinfo["KV_LAYER"] == info["KV_LAYER"]
        ddram[db["KVB"]:db["KVB"] + info["NAT"] * info["KV_LAYER"]] = chip_dram_end[pb["KVB"]:pb["KVB"] + info["NAT"] * info["KV_LAYER"]]
        nst = info["NDN"] * cfg.lv * cfg.ld * cfg.ld * 4
        ddram[db["STB"]:db["STB"] + nst] = chip_dram_end[pb["STB"]:pb["STB"] + nst]
        QC4 = info["QKV_CH"] * 4
        for d_ in range(info["NDN"]):                    # decode reads rows nc .. nc+2 = 1..3 of its 4-row window
            src = chip_dram_end[pb["CLOG"] + d_ * 3 * QC4:][:3 * QC4]
            ddram[db["CLOG"] + d_ * 4 * QC4 + QC4:][:3 * QC4] = src
        YB = dinfo["bufs"]["YB"]
        feed = tail[:]
        chip_tokens = []

        def on_argmax(sim_, idx, y):
            s_ = len(chip_tokens)
            chip_tokens.append(int(idx[0]))
            nxt = feed[s_ + 1] if s_ + 1 < len(feed) else int(idx[0])
            sim_.sram[YB:YB + 4].view(np.uint32)[0] = nxt
            print(f"  decode step {s_}: argmax {int(idx[0])} ({tok.decode([int(idx[0])])!r})  t={time.time() - t0:.0f}s",
                  flush=True)
        first = feed[0] if feed else am["chip"]
        if not feed:
            feed = [first]
        dsram[YB:YB + 4].view(np.uint32)[0] = first
        dsim = FSim(dprog, dbund, dsram, ddram, chip=SimChip(vec_lanes=512), regs={Q.X_L: L0}, on_argmax=on_argmax)
        dst = dsim.run()
        # golden continues from its own prefill cache, decode numerics (one-pass DeltaNet order)
        g.num.state = "fp32_onepass"
        gfeed = [feed[0]] + [feed[s + 1] if s + 1 < len(feed) else chip_tokens[s] for s in range(n - 1)]
        gold_tokens, margins = [], []
        for s in range(n):
            lg_ = g.forward(torch.tensor([gfeed[s]]), gcache)[-1]
            gold_tokens.append(int(lg_.argmax()))
            t2 = torch.topk(lg_, 2).values
            margins.append(float(t2[0] - t2[1]))
        out["decode"] = dict(start=L0, steps=n, feed=gfeed, chip=chip_tokens, golden=gold_tokens, golden_margin=margins,
                             agree=sum(a == b for a, b in zip(chip_tokens, gold_tokens)), sim_us=dst["time_ns"] / 1e3,
                             text=tok.decode(prompt + feed + chip_tokens[len(feed) - 1:]))
        print(f"decode after prefill: {out['decode']['agree']}/{n} tokens agree; chip {chip_tokens} golden {gold_tokens}")
        print("text tail:", repr(tok.decode(prompt[-12:] + feed + chip_tokens[len(feed) - 1:])))
    out["wall_s"] = time.time() - t0
    path = args.out or os.path.join(RES, f"fsim_prefill_08b{'' if NL == 24 else f'_L{NL}'}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {path} ({out['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
