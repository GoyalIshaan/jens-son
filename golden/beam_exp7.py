"""Experiment set 7: can attention run on the int8 matrix lanes? (found in progression step 2, note 27 §4)

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp7.py smoke
    $BEAM beam_exp7.py run
    $BEAM beam_exp7.py rest       # skip the rows the first, cancelled run finished

The lanes multiply int8 x int8. For attention on them, q . K needs q as int8 and P . V needs the softmax
probabilities P as int8 (K and V are already int8, D-005). Tested: q as int8 with power-of-two scales per 32
(the chip's activation format, D-019); P as int8 with power-of-two scales per 32 positions, or on a fixed
1/127 grid. Paired KL against exact attention, at 8K (code and prose) and at 32K (prose), bf16 weights;
then with the chip's full numerics.
"""
import json
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, eval_texts, image, save, strict_fp32, weights
from beam_exp2 import calib_sets, compare_positions, run_positions
from beam_exp5 import kl_nll, paired, per_position, summary

MODEL_DIR = "/weights/Qwen3.5-9B"


def _run(smoke, rest=False):
    import torch
    from transformers import AutoTokenizer
    import beam_exp2
    from qwen35_golden import Golden, Numerics
    from formats import CHIP_ACT
    from gptq import quantize_model_gptq
    strict_fp32()
    ensure_model()
    t_all = time.time()
    n = 512 if smoke else 8192
    n_long, every = (2048, 16) if smoke else (32768, 8)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, n).items()}
    long_ids = torch.tensor(eval_texts(tok, n_long)["wikitext"], device="cuda")
    pos = torch.arange(0, len(long_ids) - 1, every, device="cuda")
    beam_exp2.BUCKETS = ((0, 4096), (4096, 8192), (8192, 16384), (16384, 32768))
    gold = Golden(MODEL_DIR)
    ref = {k: per_position(gold, v) for k, v in texts.items()}
    nll_ref = {k: kl_nll(ref[k], ref[k], v)[1] for k, v in texts.items()}
    t0 = time.time()
    ref_long = run_positions(gold, long_ids, pos)
    out = dict(kind="exp7_attn_int8", env=environment(), smoke=smoke, rest=rest, n_tokens={k: len(v) for k, v in texts.items()},
               long=dict(n_tokens=len(long_ids), every=every, ref_secs=time.time() - t0), rows=[])
    configs = [("q int8", dict(attn_q=CHIP_ACT)),
               ("P int8, power-of-two scale per 32 positions", dict(attn_p=CHIP_ACT)),
               ("P int8, fixed 1/127 grid", dict(attn_p="int8_fixed127")),
               ("q int8 + P int8 power-of-two", dict(attn_q=CHIP_ACT, attn_p=CHIP_ACT)),
               ("q int8 + P int8 fixed", dict(attn_q=CHIP_ACT, attn_p="int8_fixed127"))]

    def evaluate(cond, label, num, base=None, do_long=False):
        gold.num = num
        t0 = time.time()
        row = dict(cond=cond, label=label, numerics=num.describe(), sets={})
        raw = {}
        for k, v in texts.items():
            kl, nll, top1 = kl_nll(ref[k], per_position(gold, v), v)
            s = summary(kl, nll, top1, nll_ref[k])
            if base is not None:
                s["kl_vs_base"] = paired(kl, base[k][0])
            row["sets"][k] = s
            raw[k] = (kl, nll)
        if do_long:
            row["long"] = compare_positions(ref_long, run_positions(gold, long_ids, pos), long_ids.cpu(), pos.cpu())
        row["secs"] = time.time() - t0
        print(cond, label, {k: round(x["kl_mean"], 6) for k, x in row["sets"].items()},
              "long", round(row["long"]["kl_mean"], 6) if do_long else "-", f"{row['secs']:.0f}s", flush=True)
        out["rows"].append(row)
        return raw

    if not rest:   # rest=True: the sanity row and the 8K-only rows finished in the first (cancelled) run
        evaluate("bf16", "exact attention (sanity: 0)", Numerics())
    for label, kw in (configs[3:] if rest else configs):
        evaluate("bf16", label, Numerics(**kw), do_long=label.startswith("q int8 +"))
    # the chip's full numerics
    wkw = dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})
    calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=8 if smoke else 64,
                                                                 seq_len=128 if smoke else 512)]
    gold.num = Numerics()
    t0 = time.time()
    quantize_model_gptq(gold, Numerics(**wkw), calib, log=lambda m: print(m, flush=True))
    out["gptq_secs"] = time.time() - t0
    chip = dict(**wkw, act=CHIP_ACT, kv="int8_head")
    base = evaluate("chip", "chip numerics, exact attention", Numerics(**chip))
    for label, kw in configs[3:]:
        evaluate("chip", "chip numerics, " + label, Numerics(**chip, **kw), base)
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke7_attn_int8" if smoke else "exp7_attn_int8", out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def attn_int8(smoke: bool = False, rest: bool = False):
    return _run(smoke, rest)


def _store(name, res):
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    res = json.loads(attn_int8.remote(which == "smoke", which == "rest"))
    _store("smoke7_attn_int8" if which == "smoke" else "exp7_attn_int8", res)
    for r in res["rows"]:
        print(r["cond"], r["label"], {k: round(s["kl_mean"], 6) for k, s in r["sets"].items()},
              "long", round(r["long"]["kl_mean"], 6) if "long" in r else "-")
