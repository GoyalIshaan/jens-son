"""Compare architecturally live prefill bytes across repeatable memory and engine timing variations."""
import argparse
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer

import run_fsim_prefill as pre
from fsim_v1 import FSim
from sim_v1 import SimChip


def variant(seed):
    rng = random.Random(seed)
    return SimChip(vec_lanes=rng.choice([128, 256, 512, 1024]),
                   latency_ns=rng.uniform(40, 500), bw=rng.uniform(200e9, 700e9),
                   mxu_drain=rng.randint(8, 80), l_overhead=rng.randint(4, 40),
                   vec_overhead=rng.randint(2, 30))


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--layers", type=int, default=4)
    arg.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 51])
    arg.add_argument("--out", type=Path, required=True)
    args = arg.parse_args()
    torch.set_num_threads(4)
    raw = pre.G.load_tensors(pre.MODEL, "cpu")
    qw = pre.P.quantize_all(raw, pre.Q.QWEN_08B, pre.CACHE)
    ids = Tokenizer.from_file(str(Path(pre.MODEL) / "tokenizer.json")).encode(pre.TEXT).ids[:256]
    cfg = dataclasses.replace(pre.Q.QWEN_08B, layers=args.layers)
    program, info = pre.QP.build_prefill(chunks=2, lmax=1024, fused=True, dn="lite", cfg=cfg)
    bundles = program.assemble()
    runs = []
    for seed in [None, *args.seeds]:
        dram, sram = pre.P.build_images_prefill(raw, qw, info, ids)
        chip = variant(seed) if seed is not None else SimChip(vec_lanes=512)
        sim = FSim(program, bundles, sram, dram, chip=chip, regs={pre.Q.X_L: 0})
        stats = sim.run()
        base = info["dbase"]
        def digest(mem, start, length):
            return hashlib.sha256(memoryview(mem)[start:start + length]).hexdigest()
        hashes = {
            "hidden": digest(sram, info["bufs"]["H"], 128 * cfg.hid * 4),
            "KV": digest(dram, base["KVB"], info["NAT"] * info["KV_LAYER"]),
            "state": digest(dram, base["STB"], info["NDN"] * cfg.lv * cfg.ld ** 2 * 4),
            "conv": digest(dram, base["CLOG"], info["NDN"] * 3 * info["QKV_CH"] * 4),
        }
        runs.append(dict(seed=seed, chip=dataclasses.asdict(chip), time_ns=stats["time_ns"], hashes=hashes))
        print(seed, hashes, flush=True)
        del sim, dram, sram
    passed = all(r["hashes"] == runs[0]["hashes"] for r in runs)
    args.out.write_text(json.dumps(dict(layers=args.layers, tokens=256, passed=passed, runs=runs), indent=2))
    if not passed:
        raise SystemExit("prefill output changed across schedules")

if __name__ == "__main__":
    main()
