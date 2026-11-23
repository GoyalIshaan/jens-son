"""Experiment set 2: find the best setting for every tunable number format.

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp2.py smoke     # tiny GPTQ run to check the code before spending GPU hours
    $BEAM beam_exp2.py sweep     # sensitivity by layer kind and depth, activation formats, 3-bit RTN
    $BEAM beam_exp2.py gptq      # GPTQ for every candidate weight format (parallel, one GPU each)
    $BEAM beam_exp2.py long32k   # KV-cache and state formats at 32K context

Results land in golden/results/exp2_<name>.json. design/15_measurements.md explains them.
"""
import json
import os
import random
import sys
import time

from beam import TaskPolicy, function

from beam_app import (BITS, GPU, KV_BYTES, compare_sets, ensure_model, environment, eval_texts, image,
                      predicted_speed, run_set, save, strict_fp32, weights)

BITS.update({"lut3_g32": 3.5, "int3_g16_fp8": 3.5, "lut4_g16_fp8": 4.5, "int8_g32": 8.5,
             "lut4_g16_fp8_t8": 4.5, "lut4_g16_fp8_te4m3": 4.5, "lut4_g16_fp8_t6": 4.5, "lut4_g16_fp8_t5": 4.5})
KV_BYTES.update({"int8_head": 1 + 2 / 256, "fp8_head": 1 + 2 / 256, "int4_g64": 0.5 + 2 / 64})
MODEL_DIR = "/weights/Qwen3.5-9B"


# ---------------------------------------------------------------- calibration data
CALIB_CODE_FILES = ["collections/__init__.py", "argparse.py", "typing.py", "functools.py", "enum.py",
                    "pathlib.py", "shutil.py", "subprocess.py", "tokenize.py", "ast.py"]


def calib_sets(tok, n_seq=64, seq_len=512, seed=0):
    """Half English (WikiText-2 *train*), half Python (stdlib files NOT used for evaluation)."""
    import json as _json
    from datasets import load_dataset
    rng = random.Random(seed)
    wiki = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    root = os.path.dirname(os.path.dirname(_json.__file__))
    code = "\n\n".join(f"# file: {f}\n" + open(os.path.join(root, f)).read() for f in CALIB_CODE_FILES)
    out = []
    for text in (wiki, code):
        ids = tok(text)["input_ids"]
        starts = sorted(rng.sample(range(0, len(ids) - seq_len), n_seq // 2))
        out += [ids[s:s + seq_len] for s in starts]
    return out


def weight_rmse(gold):
    return {k: dict(format=v[3], rel_rmse=(v[0] / v[1]) ** 0.5, weights=v[2]) for k, v in gold.weight_error.items()}


# ---------------------------------------------------------------- sweep (plain rounding)
SWEEP = [
    # where does 4-bit error come from? one layer kind at a time, everything else bf16
    ("only ffn nvfp4", dict(weights={"ffn": "nvfp4"})),
    ("only deltanet nvfp4", dict(weights={"deltanet": "nvfp4"})),
    ("only attn nvfp4", dict(weights={"attn": "nvfp4"})),
    ("only lm_head nvfp4", dict(weights={"lm_head": "nvfp4"})),
    ("only embed nvfp4", dict(weights={"embed": "nvfp4"})),
    # same at 3 bits, which magnifies the differences
    ("only ffn lut3", dict(weights={"ffn": "lut3_g32"})),
    ("only deltanet lut3", dict(weights={"deltanet": "lut3_g32"})),
    ("only attn lut3", dict(weights={"attn": "lut3_g32"})),
    ("only lm_head lut3", dict(weights={"lm_head": "lut3_g32"})),
    ("only embed lut3", dict(weights={"embed": "lut3_g32"})),
    # which depth tolerates 3-bit FFN?
    ("only ffn layers 0-7 lut3", dict(weights={"ffn@0-7": "lut3_g32"})),
    ("only ffn layers 8-15 lut3", dict(weights={"ffn@8-15": "lut3_g32"})),
    ("only ffn layers 16-23 lut3", dict(weights={"ffn@16-23": "lut3_g32"})),
    ("only ffn layers 24-31 lut3", dict(weights={"ffn@24-31": "lut3_g32"})),
    # activation formats alone (bf16 weights): can the scale unit get simpler?
    ("only act int8_g32", dict(act="int8_g32")),
    ("only act int8_g128", dict(act="int8_g128")),
    ("only act int8 per-vector", dict(act="int8_tok")),
    ("only act fp8_g32", dict(act="fp8_g32")),
    # 3-bit everywhere, plain rounding (to compare with GPTQ)
    ("rtn lut3_g32 everywhere", dict(weights={"default": "lut3_g32"})),
    ("rtn int3_g16_fp8 everywhere", dict(weights={"default": "int3_g16_fp8"})),
]


@function(gpu=GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=10800,
          task_policy=TaskPolicy(ttl=14400, timeout=10800))   # may queue for hours waiting for a GPU
def sweep():
    import torch
    from transformers import AutoTokenizer
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, 2048).items()}
    gold = Golden(MODEL_DIR)
    ref = {k: run_set(gold, v) for k, v in ids.items()}
    out = dict(kind="exp2_sweep", env=environment(), n_tokens={k: len(v) for k, v in ids.items()},
               rounding="round-to-nearest", configs=[])
    for name, kw in SWEEP:
        num = Numerics(**kw)
        t0 = time.time()
        gold.set_numerics(num)
        row = dict(name=name, numerics=num.describe(), weight_rel_rmse=weight_rmse(gold),
                   predicted=predicted_speed(gold, num), sets={})
        for k, v in ids.items():
            row["sets"][k] = compare_sets(ref[k], run_set(gold, v), v.cpu())
        row["secs"] = time.time() - t0
        out["configs"].append(row)
        print(name, {k: round(s["kl_mean"], 4) for k, s in row["sets"].items()}, flush=True)
        torch.cuda.empty_cache()
    save("exp2_sweep", out)
    return json.dumps(out)


# ---------------------------------------------------------------- GPTQ
GPTQ = {
    "gptq int4_g32": dict(weights={"default": "int4_g32"}),
    "gptq nvfp4": dict(weights={"default": "nvfp4"}),
    "gptq lut4_g32": dict(weights={"default": "lut4_g32"}),
    "gptq ffn lut3 + rest nvfp4": dict(weights={"default": "nvfp4", "ffn": "lut3_g32"}),
    "gptq ffn int3_g16_fp8 + rest nvfp4": dict(weights={"default": "nvfp4", "ffn": "int3_g16_fp8"}),
    "gptq lut3_g32 everywhere": dict(weights={"default": "lut3_g32"}),
    # round 3: refinements of the winner (GPTQ + learned 4-bit table)
    "gptq lut4_g32, embed int8": dict(weights={"default": "lut4_g32", "embed": "int8_g32"}),
    "gptq lut4_g32, embed int8, lm_head lut3": dict(weights={"default": "lut4_g32", "embed": "int8_g32",
                                                            "lm_head": "lut3_g32"}),
    "gptq lut4_g16_fp8, embed int8": dict(weights={"default": "lut4_g16_fp8", "embed": "int8_g32"}),
    "gptq lut4_g32, embed int8, 128 calib seqs": dict(weights={"default": "lut4_g32", "embed": "int8_g32"}),
    "gptq lut4_g32, embed int8, table resampled": dict(weights={"default": "lut4_g32", "embed": "int8_g32"}),
}
GPTQ.update({
    # round 4 (experiment set 3): the winner's table entries as the chip would store them
    "gptq lut4_g16_fp8, embed int8, control rerun": dict(weights={"default": "lut4_g16_fp8", "embed": "int8_g32"}),
    "gptq lut4_g16_fp8, embed int8, table int8": dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"}),
    "gptq lut4_g16_fp8, embed int8, table e4m3": dict(weights={"default": "lut4_g16_fp8_te4m3", "embed": "int8_g32"}),
    "gptq lut4_g16_fp8, embed int8, table int6": dict(weights={"default": "lut4_g16_fp8_t6", "embed": "int8_g32"}),
    "gptq lut4_g16_fp8, embed int8, table int5": dict(weights={"default": "lut4_g16_fp8_t5", "embed": "int8_g32"}),
})
CALIB_SEQS = {"gptq lut4_g32, embed int8, 128 calib seqs": 128}
EXTRA_EVAL = dict(act="int8_g32", kv="int8_g64")      # the full chip pipeline on top of the weights


@function(gpu=GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=10800,
          task_policy=TaskPolicy(ttl=14400, timeout=10800))   # may queue for hours waiting for a GPU
def gptq_job(name: str, n_calib: int = 64, smoke: bool = False):
    import torch
    from transformers import AutoTokenizer
    from gptq import quantize_model_gptq
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    n_eval = 256 if smoke else 2048
    ids = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, n_eval).items()}
    calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=n_calib, seq_len=128 if smoke else 512)]

    gold = Golden(MODEL_DIR)
    t0 = time.time()
    ref = {k: run_set(gold, v) for k, v in ids.items()}
    kw = GPTQ[name]
    num = Numerics(**kw)
    log = []
    t1 = time.time()
    quantize_model_gptq(gold, num, calib, log=lambda m: (print(m, flush=True), log.append(m)))
    gptq_secs = time.time() - t1
    row = dict(name=name, numerics=num.describe(), n_calib_seq=len(calib), calib_seq_len=len(calib[0]),
               gptq_secs=gptq_secs, weight_rel_rmse=weight_rmse(gold), predicted=predicted_speed(gold, num),
               sets={k: compare_sets(ref[k], run_set(gold, v), v.cpu()) for k, v in ids.items()})
    # same GPTQ weights, plus int8 activations and int8 KV (the full chip pipeline)
    gold.num = Numerics(**kw, **EXTRA_EVAL)
    row["with_act_int8_kv_int8"] = dict(numerics=gold.num.describe(),
                                        sets={k: compare_sets(ref[k], run_set(gold, v), v.cpu())
                                              for k, v in ids.items()})
    out = dict(kind="exp2_gptq", env=environment(), smoke=smoke, n_tokens={k: len(v) for k, v in ids.items()},
               total_secs=time.time() - t0, gpu_peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
               config=row, log=log)
    save(("smoke_" if smoke else "exp2_") + name.replace(" ", "_").replace("+", "plus"), out)
    return json.dumps(out)


# Batches: several GPTQ configs back to back in one job, so the job queues for a GPU only once.
# Fits a 24 GB GPU (only the current layer's bf16 weights are on the GPU during GPTQ).
ANY_GPU = ["RTX5090", "RTX4090", "A10G"]
BATCH_A = ["gptq nvfp4", "gptq lut4_g32", "gptq lut3_g32 everywhere"]
BATCH_B = ["gptq int4_g32", "gptq ffn int3_g16_fp8 + rest nvfp4"]
BATCH_C = ["gptq lut4_g32, embed int8", "gptq lut4_g32, embed int8, lm_head lut3"]
BATCH_D = ["gptq lut4_g16_fp8, embed int8", "gptq lut4_g32, embed int8, 128 calib seqs"]
BATCH_E = ["gptq lut4_g32, embed int8, table resampled"]   # same as C's winner, new table sampling: run-to-run check


def table_stats(gold):
    """How fine the learned tables need to be: gaps between neighbouring levels, over all tensors."""
    import torch
    from formats import QTensor
    gaps, mins, n_distinct = [], [], []
    for q in gold.W.values():
        if isinstance(q, QTensor) and len(q.table) == 16 and not float(q.table[0]).is_integer():
            t = q.table.float().cpu()
            d = t[1:] - t[:-1]
            gaps.append(d)
            mins.append(d.min().item())
            n_distinct.append(len(torch.unique(t)))
    if not gaps:
        return None
    g = torch.cat(gaps)
    return dict(n_tables=len(mins), gap_min=min(mins), gap_median=g.median().item(), gap_mean=g.mean().item(),
                smallest_gap_p01=torch.tensor(mins).quantile(0.01).item(),
                min_distinct_levels=min(n_distinct), tables_with_merged_levels=sum(n < 16 for n in n_distinct))


def _gptq_batch(names, prefix="exp2_"):
    import torch
    from transformers import AutoTokenizer
    from gptq import quantize_model_gptq
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, 2048).items()}
    calib_all = {n: [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=n, seq_len=512)]
                 for n in sorted({CALIB_SEQS.get(x, 64) for x in names})}
    gold = Golden(MODEL_DIR)
    ref = {k: run_set(gold, v) for k, v in ids.items()}
    done = []
    for j, name in enumerate(names):
        t0 = time.time()
        if j:
            gold.W = {}
            torch.cuda.empty_cache()
            gold.set_numerics(Numerics())
        kw = GPTQ[name]
        num = Numerics(**kw)
        log = []
        calib = calib_all[CALIB_SEQS.get(name, 64)]
        quantize_model_gptq(gold, num, calib, log=lambda m: (print(m, flush=True), log.append(m)))
        row = dict(name=name, numerics=num.describe(), n_calib_seq=len(calib), calib_seq_len=len(calib[0]),
                   gptq_secs=time.time() - t0, weight_rel_rmse=weight_rmse(gold), predicted=predicted_speed(gold, num),
                   table_stats=table_stats(gold),
                   sets={k: compare_sets(ref[k], run_set(gold, v), v.cpu()) for k, v in ids.items()})
        gold.num = Numerics(**kw, **EXTRA_EVAL)
        row["with_act_int8_kv_int8"] = dict(numerics=gold.num.describe(),
                                            sets={k: compare_sets(ref[k], run_set(gold, v), v.cpu())
                                                  for k, v in ids.items()})
        out = dict(kind="exp2_gptq", env=environment(), smoke=False, n_tokens={k: len(v) for k, v in ids.items()},
                   total_secs=time.time() - t0, gpu_peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9,
                   config=row, log=log)
        save(prefix + name.replace(" ", "_").replace("+", "plus").replace(",", ""), out)
        print("SAVED", name, {k: round(s["kl_mean"], 4) for k, s in row["sets"].items()}, flush=True)
        done.append(out)
    return json.dumps(done)


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def gptq_a():
    return _gptq_batch(BATCH_A)


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def gptq_b():
    return _gptq_batch(BATCH_B)


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def gptq_c():
    return _gptq_batch(BATCH_C)


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def gptq_d():
    return _gptq_batch(BATCH_D)


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def gptq_e():
    return _gptq_batch(BATCH_E)


# ---------------------------------------------------------------- 32K context
LONG32 = [
    ("kv int8_g64", dict(kv="int8_g64")),
    ("kv int8 per-head", dict(kv="int8_head")),
    ("kv fp8 per-head", dict(kv="fp8_head")),
    ("kv int4_g32", dict(kv="int4_g32")),
    ("kv int4_g64", dict(kv="int4_g64")),
    ("state bf16", dict(state="bf16")),
]
BUCKETS = ((0, 2048), (2048, 8192), (8192, 16384), (16384, 32768))


def run_positions(gold, ids, pos, chunk=512, piece=4096):
    """Prefill in 4K-token pieces, carrying the KV cache and DeltaNet state between pieces
    (identical maths to one big pass; checked in validate), then score the sampled positions."""
    import torch
    cache = gold.new_cache()
    h = torch.cat([gold.forward(ids[p0:p0 + piece], cache, return_hidden=True) for p0 in range(0, len(ids), piece)])
    h = h[pos]
    return [torch.log_softmax(gold.logits(h[c:c + chunk]), -1).cpu() for c in range(0, h.shape[0], chunk)]


def compare_positions(ref, cand, ids, pos):
    import torch
    r, q = torch.cat(ref), torch.cat(cand)
    tgt = ids[pos + 1]
    kl = (r.exp() * (r - q)).sum(-1)
    nll_r = -r.gather(1, tgt[:, None])[:, 0]
    nll_q = -q.gather(1, tgt[:, None])[:, 0]
    d = nll_q - nll_r
    out = dict(kl_mean=kl.mean().item(), kl_stderr=(kl.std() / len(kl) ** 0.5).item(),
               top1_agree=(r.argmax(-1) == q.argmax(-1)).float().mean().item(),
               nll_diff_mean=d.mean().item(), nll_diff_stderr=(d.std() / len(d) ** 0.5).item(),
               ppl_change_pct=100 * (nll_q.mean().exp() / nll_r.mean().exp() - 1).item(), by_position={})
    for a, b in BUCKETS:
        m = (pos >= a) & (pos < b)
        if m.any():
            out["by_position"][f"{a}-{b}"] = dict(kl=kl[m].mean().item(), kl_stderr=(kl[m].std() / m.sum() ** 0.5).item(),
                                                  nll_diff=d[m].mean().item(), n=int(m.sum()))
    return out


@function(gpu=["RTX5090", "RTX4090"], cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=10800,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=10800))
def long32k(n_tokens: int = 32768, every: int = 8):
    import torch
    from transformers import AutoTokenizer
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = torch.tensor(eval_texts(tok, n_tokens)["wikitext"], device="cuda")
    pos = torch.arange(0, len(ids) - 1, every, device="cuda")
    gold = Golden(MODEL_DIR)
    t0 = time.time()
    ref = run_positions(gold, ids, pos)
    out = dict(kind="exp2_long32k", env=environment(), n_tokens=len(ids), scored_positions=len(pos),
               weights="bf16 (to isolate KV and state effects)", ref_secs=time.time() - t0, configs=[])
    print(f"reference {out['ref_secs']:.0f}s", flush=True)
    for name, kw in LONG32:
        num = Numerics(**kw)
        gold.set_numerics(num)
        t0 = time.time()
        row = dict(name=name, numerics=num.describe(),
                   result=compare_positions(ref, run_positions(gold, ids, pos), ids.cpu(), pos.cpu()))
        row["secs"] = time.time() - t0
        out["configs"].append(row)
        print(name, round(row["result"]["kl_mean"], 5),
              {k: round(v["kl"], 5) for k, v in row["result"]["by_position"].items()}, flush=True)
        torch.cuda.empty_cache()
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("exp2_long32k", out)
    return json.dumps(out)


# ---------------------------------------------------------------- driver
def _store(name, res):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "smoke":
        res = json.loads(gptq_job.remote("gptq nvfp4", 4, True))
        _store("smoke_gptq", res)
        print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1)[:4000])
    elif which == "sweep":
        _store("exp2_sweep", json.loads(sweep.remote()))
    elif which == "long32k":
        _store("exp2_long32k", json.loads(long32k.remote()))
    elif which in ("gptq_a", "gptq_b", "gptq_c", "gptq_d", "gptq_e"):
        fn = {"gptq_a": gptq_a, "gptq_b": gptq_b, "gptq_c": gptq_c, "gptq_d": gptq_d, "gptq_e": gptq_e}[which]
        for res in json.loads(fn.remote()):
            n = res["config"]["name"]
            _store("exp2_" + n.replace(" ", "_").replace("+", "plus"), res)
    elif which == "gptq":
        names = [list(GPTQ)[int(a)] for a in sys.argv[2:]] or list(GPTQ)   # args are indices into GPTQ
        for r in gptq_job.map([(n,) for n in names]):
            if r is None:
                print("one job failed (see Beam logs)", flush=True)
                continue
            res = json.loads(r)
            n = res["config"]["name"]
            _store("exp2_" + n.replace(" ", "_").replace("+", "plus"), res)
            print(n, {k: round(s["kl_mean"], 4) for k, s in res["config"]["sets"].items()}, flush=True)
