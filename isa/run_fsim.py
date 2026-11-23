"""Run the real ISA v1 decode program on Qwen3.5-0.8B in the functional simulator, and check it against the
golden model computing with the same quantized weights (progression step 6a).

  python3 isa/run_fsim.py [n_tokens]

The prompt is fed one token per decode step (teacher forcing through the argmax buffer), then the chip
generates greedily. Per step it compares the residual stream after every layer (the chip's SRAM copy, read
when each layer's FFN-down command lands) with the golden model's, and the argmax token.
"""
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
import formats  # noqa: E402
import isa_v1 as I  # noqa: E402
import pack_v1 as P  # noqa: E402
import qwen35_golden as G  # noqa: E402
import qwen_v1 as Q  # noqa: E402
from fsim_v1 import FSim  # noqa: E402
from sim_v1 import SimChip  # noqa: E402

MODEL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"))[0]
CACHE = os.path.join(HERE, "..", "golden", "results", "q08b_chip_weights.pt")
OUT = os.path.join(HERE, "results", "fsim_08b.json")


def golden_model(raw, qw):
    g = object.__new__(G.Golden)
    g.cfg = G.Config(MODEL)
    g.dev = "cpu"
    g.num = G.Numerics(act=formats.CHIP_ACT, kv="int8_head_chip", state="fp32_onepass", attn_q=formats.CHIP_ACT,
                       attn_p=formats.CHIP_ACT)
    g.W = P.golden_weights(raw, qw, Q.QWEN_08B)
    G.CHIP.update(sfu=True, rope=True, softmax=True, dnq=True)
    return g


def main(n_tokens=12):
    t0 = time.time()
    torch.set_num_threads(os.cpu_count())
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
    prompt = tok.encode("The capital of France is Paris. The capital of Japan is").ids
    raw = G.load_tensors(MODEL, "cpu")
    qw = P.quantize_all(raw, Q.QWEN_08B, CACHE)
    print(f"weights ready ({time.time() - t0:.0f} s); prompt {len(prompt)} tokens", flush=True)

    cfg = Q.QWEN_08B
    prog, info = Q.build("decode", steps=n_tokens, lmax=256, cfg=cfg)
    bundles = prog.assemble()
    dram, sram = P.build_images(raw, qw, info)
    YB, H = info["bufs"]["YB"], info["bufs"]["H"]
    sram[YB:YB + 4].view(np.uint32)[0] = prompt[0]
    print(f"program {len(bundles)} bundles; images built ({time.time() - t0:.0f} s)", flush=True)

    chip_tokens, chip_layers = [], []
    layer_h = []

    def on_land(sim, c):
        if c.unit == "M" and c.note.startswith("FFN down"):
            layer_h.append(sim.sram[H:H + cfg.hid * 4].view(np.float32).copy())

    def on_argmax(sim, idx, y):
        step = len(chip_tokens)
        chip_tokens.append(int(idx[0]))
        chip_layers.append(list(layer_h))
        layer_h.clear()
        nxt = prompt[step + 1] if step + 1 < len(prompt) else int(idx[0])
        sim.sram[YB:YB + 4].view(np.uint32)[0] = nxt
        print(f"  chip step {step}: argmax {int(idx[0])} ({tok.decode([int(idx[0])])!r})  t={time.time() - t0:.0f}s",
              flush=True)

    sim = FSim(prog, bundles, sram, dram, chip=SimChip(vec_lanes=512), regs={Q.X_L: 0}, on_argmax=on_argmax)
    sim.on_land = on_land
    st = sim.run()
    print(f"chip done: {st['time_ns'] / 1e3:.0f} us simulated, {time.time() - t0:.0f} s wall", flush=True)

    # golden, same inputs
    g = golden_model(raw, qw)
    cache = g.new_cache()
    feed = [prompt[0]]
    for s in range(n_tokens):
        feed.append(prompt[s + 1] if s + 1 < len(prompt) else chip_tokens[s])
    gold_tokens, rows = [], []
    for s in range(n_tokens):
        trace = []
        logits = g.forward(torch.tensor([feed[s]]), cache, trace=trace)
        gt = int(logits[-1].argmax())
        gold_tokens.append(gt)
        top2 = torch.topk(logits[-1], 2).values
        rel = []
        for li in range(cfg.layers):
            a = torch.from_numpy(chip_layers[s][li]).double()
            b = trace[li + 1][0].double()
            rel.append(((a - b).norm() / b.norm()).item())
        rows.append(dict(step=s, input=feed[s], chip=chip_tokens[s], golden=gt, golden_margin=float(top2[0] - top2[1]),
                         layer_rel=rel))
        print(f"  step {s}: chip {chip_tokens[s]} golden {gt} {'ok' if chip_tokens[s] == gt else 'DIFF'}; "
              f"residual rel. diff: layer 0 {rel[0]:.1e}, mid {rel[cfg.layers // 2]:.1e}, last {rel[-1]:.1e}",
              flush=True)
    out = dict(model="Qwen3.5-0.8B", prompt=prompt, n_tokens=n_tokens, rows=rows,
               agree=sum(r["chip"] == r["golden"] for r in rows), commands=sorted(
                   ((f"{u} {o}", n) for (u, o), n in sim.counts.items()), key=lambda x: -x[1]),
               sim_us=st["time_ns"] / 1e3, wall_s=time.time() - t0,
               text=tok.decode(prompt + chip_tokens[len(prompt) - 1:]))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"tokens agree {out['agree']}/{n_tokens}; text: {out['text']!r}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
