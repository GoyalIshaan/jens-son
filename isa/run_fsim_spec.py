"""Functional run of the speculative step (MTP drafts, verify, accept, replay) on Qwen3.5-0.8B (note 31, part 3).

  python3 isa/run_fsim_spec.py [--steps 10]

1. The chip's prefill program runs a 128-token prompt (as in run_fsim_prefill.py).
2. From that state, two runs continue at position 128 with x0 = prompt token 128:
   - plain greedy decode (qwen_v1.build("decode")), the reference
   - speculative steps (qwen_v1.build("spec", k=3)): verify x0 + 3 drafts, accept the leading matches, replay them
     into the DeltaNet state at the next step, draft 3 more with the MTP head
3. Checks: the tokens the spec program emits equal plain decode's, and the verify rows' logits are bit-identical to
   decode's (the replayed DeltaNet updates and the conv window are the committed tokens' exactly); the golden model's
   greedy tokens; the chip's drafts against golden/mtp.py's drafts over the same text.
The MTP's KV ring for the prompt positions is filled on the host (golden MTP layer, chip-format weights, from the
chip's own prefill hidden states): the prefill program has no MTP pass yet (an open item, note 31 part 3).
The draft vocabulary: the 32,768 most frequent tokens over this repository's design notes and Python's standard
library (no held-out corpus on this laptop), topped up with the lowest unseen ids.
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "golden"))
import mtp as GM  # noqa: E402
import pack_v1 as P  # noqa: E402
import qwen35_golden as G  # noqa: E402
import qwen_prefill_v1 as QP  # noqa: E402
import qwen_v1 as Q  # noqa: E402
import run_fsim_prefill as RP  # noqa: E402
from fsim_v1 import FSim  # noqa: E402
from sim_v1 import SimChip  # noqa: E402

MTP_CACHE = os.path.join(HERE, "..", "golden", "results", "q08b_chip_mtp_weights.pt")
LMAX = 1024


def corpus():
    import sysconfig
    texts = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "design", "*.md"))):
        texts.append(open(f).read())
    lib = sysconfig.get_paths()["stdlib"]
    for f in sorted(glob.glob(os.path.join(lib, "*.py")))[:400]:
        try:
            texts.append(open(f, encoding="utf-8").read())
        except (UnicodeDecodeError, OSError):
            pass
    return texts


def state_into(ddram, dinfo, pdram, pinfo, window_rows, nc):
    """Copy the prefill's KV cache, DeltaNet state and conv history into a decode/spec image. The conv history goes
    to rows nc .. nc + 2 of the program's (3 + T)-row window, where its first step reads it."""
    db, pb = dinfo["dbase"], pinfo["dbase"]
    n = pinfo["NAT"] * pinfo["KV_LAYER"]
    ddram[db["KVB"]:db["KVB"] + n] = pdram[pb["KVB"]:pb["KVB"] + n]
    cfg = pinfo["cfg"]
    nst = pinfo["NDN"] * cfg.lv * cfg.ld * cfg.ld * 4
    ddram[db["STB"]:db["STB"] + nst] = pdram[pb["STB"]:pb["STB"] + nst]
    QC4 = pinfo["QKV_CH"] * 4
    for d_ in range(pinfo["NDN"]):
        ddram[db["CLOG"] + d_ * window_rows * QC4 + nc * QC4:][:3 * QC4] = pdram[pb["CLOG"] + d_ * 3 * QC4:][:3 * QC4]


def put_kv(dram, info, layer, K, V):
    """Write dequantized K, V [L, kvh, 256] (int8 x fp16 scale values) into one layer of the chip's KV cache."""
    cfg, lmax = info["cfg"], info["lmax"]
    base = info["dbase"]["KVB"] + layer * info["KV_LAYER"]
    KH = lmax * cfg.hd
    L = K.shape[0]
    for x, which in ((K, 0), (V, 1)):
        s16 = (x.abs().amax(-1) / 127).half()
        q = torch.where(s16.float()[..., None] > 0, torch.round(x / s16.float()[..., None].clamp_min(1e-30)),
                        torch.zeros_like(x)).clamp(-127, 127).to(torch.int8).numpy()
        for h in range(cfg.kvh):
            sc = dram[base + 2 * cfg.kvh * KH + h * 4 * lmax + which * 2 * lmax:][:2 * L]
            sc[...] = np.ascontiguousarray(s16[:, h].numpy()).view(np.uint8)
            if which == 0:
                dram[base + h * KH:][:L * cfg.hd] = np.ascontiguousarray(q[:, h]).reshape(-1).view(np.uint8)
            else:
                vt = dram[base + cfg.kvh * KH + h * KH:][:KH].view(np.int8).reshape(lmax // 256, cfg.hd, 256)
                for p_ in range(L):
                    vt[p_ // 256, :, p_ % 256] = q[p_, h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "fsim_spec_08b.json"))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(args.threads)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(RP.MODEL, "tokenizer.json"))
    ids = tok.encode(RP.TEXT).ids
    prompt, x0 = ids[:128], ids[128]
    cfg = Q.QWEN_08B
    raw = G.load_tensors(RP.MODEL, "cpu")
    raw_mtp = GM.load_mtp(RP.MODEL)
    qw = P.quantize_all(raw, cfg, RP.CACHE)
    qmtp = P.quantize_mtp(raw_mtp, MTP_CACHE)
    remap = P.draft_vocab(tok, corpus(), Q.DRAFT_VOCAB)
    print(f"weights ready; draft vocabulary {len(remap)} ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 1. prefill on the chip
    pprog, pinfo = QP.build_prefill(chunks=1, lmax=LMAX, fused=True, dn="lite", cfg=cfg)
    pdram, psram = P.build_images_prefill(raw, qw, pinfo, prompt)
    psim = FSim(pprog, pprog.assemble(), psram, pdram, chip=SimChip(vec_lanes=512), regs={Q.X_L: 0})
    psim.run()
    H = pinfo["bufs"]["H"]
    h_chip = torch.from_numpy(psim.sram[H:H + 128 * cfg.hid * 4].view(np.float32).reshape(128, cfg.hid).copy())
    print(f"prefill done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 2a. speculative steps
    g = RP.golden_model(raw, qw, cfg, cfg.layers)
    RP.set_order(g, 256, True)
    sprog, sinfo = Q.build("spec", k=3, steps=args.steps, lmax=LMAX, cfg=cfg)
    sdram, ssram = P.spec_images(raw, raw_mtp, qw, qmtp, sinfo, remap)
    state_into(sdram, sinfo, psim.dram, pinfo, 3 + 4, 0)
    # the MTP's KV ring for positions 0..127 (host-side stand-in for a prefill MTP pass)
    mt = GM.MTP(g, RP.MODEL)
    for n, qt in qmtp.items():
        g.W[n] = qt
    post = g.rms(h_chip, "norm.weight")
    xin = mt.inputs(post, torch.tensor(prompt[1:] + [x0]))
    _, _, Km, Vm = mt.qkv(xin, torch.arange(128))
    put_kv(sdram, sinfo, sinfo["NAT"], Km, Vm)
    TK, YB = sinfo["bufs"]["TK"], sinfo["bufs"]["YB"]
    ssram[TK:TK + 16].view(np.uint32)[:] = [x0, 0, 0, 0]
    steps = []

    def on_argmax_s(sim, idx, y):
        if y.shape[1] == cfg.vocab:
            steps.append(dict(main=[int(v) for v in idx], logits=y.copy(), drafts_in=[int(v) for v in
                                                                               sim.sram[TK:TK + 16].view(np.uint32)]))
        else:
            steps[-1].setdefault("draft_idx", []).append(int(idx[0]))
    ssim = FSim(sprog, sprog.assemble(), ssram, sdram, chip=SimChip(vec_lanes=512),
                regs={Q.X_L: 128, Q.X_NC: 0, Q.X_A: 0}, on_argmax=on_argmax_s)
    sst = ssim.run()
    out_tok = [int(v) for b in ssim.host_out for v in np.frombuffer(b, np.uint32)]
    per_step = [len(np.frombuffer(b, np.uint32)) for b in ssim.host_out]
    print(f"spec: {args.steps} steps -> {len(out_tok)} tokens, accepted drafts per step {[n - 1 for n in per_step]} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- 2b. plain greedy decode, same start
    N = len(out_tok)
    dprog, dinfo = Q.build("decode", steps=N, lmax=LMAX, cfg=cfg)
    ddram, dsram = P.build_images(raw, qw, dinfo)
    state_into(ddram, dinfo, psim.dram, pinfo, 3 + 1, 1)
    dsram[dinfo["bufs"]["YB"]:][:4].view(np.uint32)[0] = x0
    dec, dlog = [], []

    def on_argmax_d(sim, idx, y):
        dec.append(int(idx[0]))
        dlog.append(y[0].copy())
        sim.sram[dinfo["bufs"]["YB"]:][:4].view(np.uint32)[0] = int(idx[0])
    dsim = FSim(dprog, dprog.assemble(), dsram, ddram, chip=SimChip(vec_lanes=512), regs={Q.X_L: 128},
                on_argmax=on_argmax_d)
    dst = dsim.run()
    print(f"decode: {N} tokens ({time.time() - t0:.0f} s)", flush=True)

    # ---------------------------------------------------------------- checks
    same_tokens = out_tok == dec
    # verify rows 0..a of each step predict emitted tokens: their logits must equal decode's bit for bit
    pos, bit_rows, max_diff = 0, 0, 0.0
    for s, n in zip(steps, per_step):
        for i in range(n):
            if pos < len(dlog):
                d = np.abs(s["logits"][i] - dlog[pos]).max()
                max_diff = max(max_diff, float(d))
                bit_rows += int(np.array_equal(s["logits"][i].view(np.uint32), dlog[pos].view(np.uint32)))
            pos += 1
    accepted = [n - 1 for n in per_step]
    # drafts: chip draft tokens per step (remapped) against golden/mtp.py over the emitted text
    seq = prompt + [x0] + out_tok
    g.num.state = "chunked_lite"
    cache = g.new_cache()
    with torch.no_grad():
        pre, postg = g.forward(torch.tensor(seq[:-1]), cache, return_prenorm=True)
    g.num.state = "fp32_onepass"
    vocab_t = torch.tensor(sorted(remap))
    gd = mt.drafts(postg, torch.tensor(seq[1:]), 3, vocab_t)          # [3, T]: draft j at i predicts seq[i + j + 2]
    agree_d, n_d = [0, 0, 0], 0
    L = 128
    for si, s in enumerate(steps[:-1]):
        a = per_step[si] - 1
        L_new = L + 1 + a
        chip_d = [remap[d] for d in s.get("draft_idx", [])]
        i = L_new - 1
        if i < gd.shape[1] and len(chip_d) == 3:
            n_d += 1
            for j in range(3):
                agree_d[j] += int(chip_d[j] == int(gd[j, i]))
        L = L_new
    # golden greedy from the golden model's own prefill
    gc = g.new_cache()
    g.num.state = "chunked_lite"
    g.forward(torch.tensor(prompt), gc)
    g.num.state = "fp32_onepass"
    gt, nxt = [], x0
    for _ in range(N):
        lg = g.forward(torch.tensor([nxt]), gc)[-1]
        nxt = int(lg.argmax())
        gt.append(nxt)
    out = dict(model="Qwen3.5-0.8B", prompt_len=128, x0=x0, steps=args.steps, k=3, spec_tokens=out_tok,
               decode_tokens=dec, golden_tokens=gt, spec_equals_decode=same_tokens,
               verify_rows=pos, verify_rows_bit_identical=bit_rows, verify_logits_max_abs_diff=max_diff,
               accepted_per_step=accepted, tokens_per_step=sum(per_step) / len(per_step),
               golden_agree=sum(a_ == b_ for a_, b_ in zip(dec, gt)),
               first_golden_diff=next((i for i, (a_, b_) in enumerate(zip(dec, gt)) if a_ != b_), None),
               drafts_vs_golden_mtp=dict(steps=n_d, agree_by_depth=agree_d),
               draft_vocab_coverage=sum(t in set(remap) for t in out_tok) / max(1, len(out_tok)),
               sim_us_spec=sst["time_ns"] / 1e3, sim_us_decode=dst["time_ns"] / 1e3,
               text=tok.decode(out_tok), wall_s=time.time() - t0,
               commands=sorted(((f"{u} {o}", n) for (u, o), n in ssim.counts.items()), key=lambda x: -x[1]))
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("commands",)}, indent=None)[:3000])

    if not (same_tokens and bit_rows == pos == len(out_tok) and len(steps) == args.steps):
        raise SystemExit("speculative host output or verified logits differ from ordinary decode")


if __name__ == "__main__":
    main()
