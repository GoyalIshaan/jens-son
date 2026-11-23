"""Experiment set 6: how accurate must the special-function unit be? (progression step 2, open item 9)

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp6.py smoke      # 512 tokens per text
    $BEAM beam_exp6.py run        # 8,192 tokens of Python code and of English prose

The chip computes exp (attention softmax, the DeltaNet decay), sigmoid and SiLU (gates, FFN, conv),
softplus (DeltaNet decay), 1/sqrt (every norm) and 1/x (softmax) in a special-function unit. A unit with
relative error about 2^-b is emulated by rounding each result to b mantissa bits (qwen35_golden.SF).
Swept: all functions together, then one at a time, with bf16 weights (the sensitive test), then the
chip's full numerics (GPTQ weights, power-of-two activation scales, int8 KV). Paired KL against exact.
"""
import json
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, eval_texts, image, save, strict_fp32, weights
from beam_exp2 import calib_sets
from beam_exp5 import kl_nll, paired, per_position, summary

MODEL_DIR = "/weights/Qwen3.5-9B"
FUNCS = ("exp", "sig", "softplus", "rsqrt", "recip")
ALL_BITS = (16, 12, 10, 8, 7, 6, 5, 4)
ONE_BITS = (6, 4)
CHIP_BITS = (None, 10, 8, 6)


def _run(n_tokens, smoke):
    import torch
    from transformers import AutoTokenizer
    import qwen35_golden as G
    from formats import CHIP_ACT
    from gptq import quantize_model_gptq
    strict_fp32()
    ensure_model()
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, n_tokens).items()}
    gold = G.Golden(MODEL_DIR)
    ref = {k: per_position(gold, v) for k, v in texts.items()}
    nll_ref = {k: kl_nll(ref[k], ref[k], v)[1] for k, v in texts.items()}
    out = dict(kind="exp6_sfu", env=environment(), smoke=smoke, n_tokens={k: len(v) for k, v in texts.items()},
               emulation="each special-function result rounded to b explicit mantissa bits (RNE)", rows=[])

    def evaluate(cond, label, bits, base=None):
        for f in FUNCS:
            G.SF[f] = bits.get(f)
        t0 = time.time()
        row = dict(cond=cond, label=label, bits=dict(bits), sets={})
        for k, v in texts.items():
            kl, nll, top1 = kl_nll(ref[k], per_position(gold, v), v)
            s = summary(kl, nll, top1, nll_ref[k])
            if base is not None:
                s["kl_vs_base"] = paired(kl, base[k][0])
                s["nll_vs_base"] = paired(nll, base[k][1])
            row["sets"][k] = s
            row.setdefault("_raw", {})[k] = (kl, nll)
        row["secs"] = time.time() - t0
        print(cond, label, {k: round(x["kl_mean"], 6) for k, x in row["sets"].items()}, f"{row['secs']:.0f}s", flush=True)
        out["rows"].append(row)
        return row

    # 1. bf16 weights, fp32 activations: only the special functions change
    evaluate("bf16", "exact (sanity: must be 0)", {})
    for b in (ALL_BITS[:3] if smoke else ALL_BITS):
        evaluate("bf16", f"all functions at {b} bits", {f: b for f in FUNCS})
    for b in (ONE_BITS[:1] if smoke else ONE_BITS):
        for f in FUNCS:
            evaluate("bf16", f"only {f} at {b} bits", {f: b})
    # 2. the chip's numerics
    wkw = dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})
    calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=8 if smoke else 64,
                                                                 seq_len=128 if smoke else 512)]
    t0 = time.time()
    for f in FUNCS:
        G.SF[f] = None
    quantize_model_gptq(gold, G.Numerics(**wkw), calib, log=lambda m: print(m, flush=True))
    out["gptq_secs"] = time.time() - t0
    gold.num = G.Numerics(**wkw, act=CHIP_ACT, kv="int8_head")
    base = None
    for b in CHIP_BITS:
        r = evaluate("chip", "exact functions" if b is None else f"all functions at {b} bits",
                     {} if b is None else {f: b for f in FUNCS}, base)
        if b is None:
            base = r["_raw"]
    for r in out["rows"]:
        r.pop("_raw", None)
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke6_sfu" if smoke else "exp6_sfu", out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def sfu(smoke: bool = False):
    return _run(512 if smoke else 8192, smoke)


def _store(name, res):
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "smoke":
        res = json.loads(sfu.remote(True))
        _store("smoke6_sfu", res)
        for r in res["rows"]:
            print(r["cond"], r["label"], {k: round(s["kl_mean"], 6) for k, s in r["sets"].items()})
    elif which == "run":
        _store("exp6_sfu", json.loads(sfu.remote(False)))
