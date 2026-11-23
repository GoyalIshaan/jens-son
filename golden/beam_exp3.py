"""Experiment set 3: the last two checks before the PE design is frozen.

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp3.py smoke          # a few problems per task, batched vs one-at-a-time check
    $BEAM beam_exp3.py tables_a       # GPTQ winner with int8 table entries + a same-job control
    $BEAM beam_exp3.py tables_b       # table entries as E4M3, int6, int5
    $BEAM beam_exp3.py tasks <model>  # bf16 | rtn | winner : GSM8K + HumanEval + MBPP, executed

1. Table entries. The winning format looks each 4-bit code up in a 16-entry table learned
   per matrix. Software kept the entries in fp32; the chip stores them in 8 bits so the
   multiplier stays 8-bit x 8-bit. tables_* measure what that rounding costs.

2. Tasks. KL and perplexity are proxies. These jobs check that the quantized model still
   solves maths problems and writes code that passes unit tests. For speed they use
   Hugging Face's batched generate with bf16 arithmetic; only the *weights* differ between
   models (dequantized from our formats into bf16, which adds ~0.2% relative rounding on
   top of a ~10% quantization error). int8 activations / KV were shown to cost <= 0.001 KL
   in experiment set 2, so they are not simulated here.

Results land in golden/results/exp3_*.json. design/15_measurements.md explains them.
"""
import json
import os
import re
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, image, save, weights
from beam_exp2 import ANY_GPU, _gptq_batch, calib_sets, weight_rmse

MODEL_DIR = "/weights/Qwen3.5-9B"
PREFIX = "model.language_model."


# ---------------------------------------------------------------- 1. table entries
TABLES_A = ["gptq lut4_g16_fp8, embed int8, table int8", "gptq lut4_g16_fp8, embed int8, control rerun"]
TABLES_B = ["gptq lut4_g16_fp8, embed int8, table e4m3", "gptq lut4_g16_fp8, embed int8, table int6",
            "gptq lut4_g16_fp8, embed int8, table int5"]


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def tables_a():
    return _gptq_batch(TABLES_A, prefix="exp3_")


@function(gpu=ANY_GPU, cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def tables_b():
    return _gptq_batch(TABLES_B, prefix="exp3_")


# ---------------------------------------------------------------- 2. task benchmarks
TASK_MODELS = {
    "bf16": None,
    "rtn": ("rtn", dict(weights={"default": "int4_g32"})),          # plain int4, scale per 32 (like Q4_0)
    "winner": ("gptq", dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})),
}
TASK_LABEL = {"bf16": "bf16 (published weights)", "rtn": "plain int4_g32, round-to-nearest",
              "winner": "chip format: GPTQ + learned 4-bit table (int8 entries) + E4M3 scale per 16, embed int8"}


def fake_quant(kind, kw):
    """Quantize with our formats, then expand back to bf16 tensors keyed by golden names (CPU)."""
    import torch
    from formats import QTensor, quantize_weight
    from qwen35_golden import Golden, Numerics, load_tensors
    num = Numerics(**kw)
    out, info = {}, {}

    def expand(q):
        return torch.cat([q.dequant_slice(r, r + 16384).to(torch.bfloat16).cpu()
                          for r in range(0, q.shape[0], 16384)])

    if kind == "rtn":
        raw = load_tensors(MODEL_DIR, "cpu")
        err = {}
        for name, t in raw.items():
            fmt = num.weight_format(name)
            if not fmt:
                continue
            q = quantize_weight(t.cuda(), fmt)
            out[name] = expand(q)
            e = err.setdefault(fmt, [0.0, 0.0])
            e[0] += (out[name].float() - t.float()).pow(2).sum().item()
            e[1] += t.float().pow(2).sum().item()
            del q
        info["weight_rel_rmse_after_bf16"] = {k: (a / b) ** 0.5 for k, (a, b) in err.items()}
        del raw
    else:
        from gptq import quantize_model_gptq
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=64, seq_len=512)]
        gold = Golden(MODEL_DIR)
        t0 = time.time()
        quantize_model_gptq(gold, num, calib, log=lambda m: print(m, flush=True))
        info["gptq_secs"] = time.time() - t0
        info["weight_rel_rmse"] = weight_rmse(gold)
        for name, q in gold.W.items():
            if isinstance(q, QTensor):
                out[name] = expand(q)
        del gold
    torch.cuda.empty_cache()
    info["numerics"] = num.describe()
    info["n_replaced_tensors"] = len(out)
    return out, info


def load_hf(fq):
    import torch
    from transformers import AutoModelForImageTextToText
    hf = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()
    if fq:
        sd = hf.state_dict()
        for name, t in fq.items():
            key = name if name == "lm_head.weight" else PREFIX + name
            assert key in sd, key
            assert sd[key].shape == t.shape, (key, sd[key].shape, t.shape)
            with torch.no_grad():
                sd[key].copy_(t)
    return hf


TOKEN_BUDGET = 8192


def generate(hf, tok, prompts, max_new, stops, bs):
    """Greedy, batched (left padding), longest prompts first. Returns completions cut at the first stop."""
    import torch
    lens = [len(tok(p)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lens[i])
    outs, n_gen = [None] * len(prompts), 0
    batches, b = [], 0                       # cap each batch at ~TOKEN_BUDGET prompt tokens (fits 24 GB)
    while b < len(order):
        n = max(1, min(bs, TOKEN_BUDGET // lens[order[b]]))
        batches.append(order[b:b + n])
        b += n
    for k, idx in enumerate(batches):
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = hf.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                            top_k=None, stop_strings=stops, tokenizer=tok, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            new = g[j, enc["input_ids"].shape[1]:]
            n_gen += int((new != tok.pad_token_id).sum())
            text = tok.decode(new, skip_special_tokens=True)
            cut = min([text.find(s) for s in stops if s in text] + [len(text)])
            outs[i] = text[:cut]
        print(f"    batch {k + 1}/{len(batches)} ({len(idx)} prompts)", flush=True)
    return outs, n_gen


def run_code(program, timeout=15):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-c", program], cwd=d, timeout=timeout, capture_output=True)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def run_all_code(programs):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(run_code, programs))


# ---- GSM8K: 8-shot chain of thought, exemplars = first 8 train problems
def gsm8k_task(limit=None):
    from datasets import load_dataset
    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")

    def clean(a):
        body, ans = a.split("####")
        return re.sub(r"<<[^>]*>>", "", body).strip(), ans.strip().replace(",", "")

    shots = ""
    for ex in tr.select(range(8)):
        body, ans = clean(ex["answer"])
        shots += f"Question: {ex['question']}\nAnswer: {body}\nThe answer is {ans}.\n\n"
    rows = list(te) if limit is None else list(te.select(range(limit)))
    prompts = [shots + f"Question: {r['question']}\nAnswer:" for r in rows]
    golds = [clean(r["answer"])[1] for r in rows]
    return prompts, golds


def gsm8k_score(outs, golds):
    def num(s):
        m = re.search(r"The answer is\s*\$?\s*(-?[\d,]*\.?\d+)", s)
        if not m:
            ms = re.findall(r"-?[\d,]*\.?\d+", s)
            if not ms:
                return None
            v = ms[-1]
        else:
            v = m.group(1)
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return [num(o) is not None and abs(num(o) - float(g)) < 1e-6 for o, g in zip(outs, golds)]


# ---- HumanEval: complete the function body, run the official tests
HE_STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert"]


def humaneval_task(limit=None):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = list(ds) if limit is None else list(ds.select(range(limit)))
    return rows, [r["prompt"] for r in rows]


def humaneval_score(rows, outs):
    progs = [r["prompt"] + o + "\n\n" + r["test"] + f"\ncheck({r['entry_point']})\n" for r, o in zip(rows, outs)]
    return run_all_code(progs)


# ---- MBPP: 3-shot (task ids 2-4 as examples, the standard prompt), test ids 11-510
def mbpp_task(limit=None):
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "full")
    allr = {r["task_id"]: r for split in ds.values() for r in split}

    def block(r, with_code):
        s = (f"You are an expert Python programmer, and here is your task: {r['text']} "
             f"Your code should pass these tests:\n\n" + "\n".join(r["test_list"]) + "\n[BEGIN]\n")
        return s + (r["code"] + "\n[DONE]\n\n" if with_code else "")
    shots = "".join(block(allr[i], True) for i in (2, 3, 4))
    rows = [allr[i] for i in range(11, 511)]
    rows = rows if limit is None else rows[:limit]
    return rows, [shots + block(r, False) for r in rows]


def mbpp_score(rows, outs):
    progs = [o + "\n\n" + (r.get("test_setup_code") or "") + "\n" + "\n".join(r["test_list"]) + "\n"
             for r, o in zip(rows, outs)]
    return run_all_code(progs)


def _tasks(model, limit=None, check_batching=False):
    import torch
    from transformers import AutoTokenizer
    ensure_model()
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    out = dict(kind="exp3_tasks", model=model, label=TASK_LABEL[model], env=environment(), limit=limit,
               arithmetic="Hugging Face generate, bf16 weights/activations, greedy, batched with left padding",
               tasks={})
    fq = None
    if TASK_MODELS[model]:
        t0 = time.time()
        fq, info = fake_quant(*TASK_MODELS[model])
        info["secs"] = time.time() - t0
        out["quant"] = info
        print("quantized", {k: v for k, v in info.items() if k != "weight_rel_rmse"}, flush=True)
    hf = load_hf(fq)
    del fq
    out["hf_class"] = type(hf).__name__

    tasks = [
        ("gsm8k", gsm8k_task, 384, ["Question:", "\n\n\n"], 32),
        ("humaneval", humaneval_task, 512, HE_STOPS, 32),
        ("mbpp", mbpp_task, 384, ["[DONE]", "\n[BEGIN]"], 32),
    ]
    for name, fn, max_new, stops, bs in tasks:
        t0 = time.time()
        if name == "gsm8k":
            prompts, golds = fn(limit)
        else:
            rows, prompts = fn(limit)
        print(f"{name}: {len(prompts)} problems", flush=True)
        outs, n_gen = generate(hf, tok, prompts, max_new, stops, bs)
        gen_secs = time.time() - t0
        if name == "gsm8k":
            ok = gsm8k_score(outs, golds)
        elif name == "humaneval":
            ok = humaneval_score(rows, outs)
        else:
            ok = mbpp_score(rows, outs)
        acc = sum(ok) / len(ok)
        out["tasks"][name] = dict(n=len(ok), correct=int(sum(ok)), accuracy=acc,
                                  stderr=(acc * (1 - acc) / len(ok)) ** 0.5, per_problem=[bool(x) for x in ok],
                                  outputs=outs, gen_secs=gen_secs, generated_tokens=n_gen,
                                  max_new_tokens=max_new, stops=stops, batch=bs)
        print(f"{name}: {sum(ok)}/{len(ok)} = {acc:.3f}  ({gen_secs:.0f}s, {n_gen} tokens)", flush=True)
        if check_batching:
            single, _ = generate(hf, tok, prompts[:4], max_new, stops, 1)
            out["tasks"][name]["batched_vs_single_identical"] = [a == b for a, b in zip(outs[:4], single)]
            out["tasks"][name]["single_outputs"] = single
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save(f"exp3_tasks_{model}" if limit is None else f"smoke3_tasks_{model}", out)
    return json.dumps(out)


TASK_KW = dict(cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
               env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
               task_policy=TaskPolicy(ttl=14400, timeout=14400))


@function(gpu=["RTX5090", "RTX4090"], **TASK_KW)
def tasks(model: str, limit: int = 0, check_batching: bool = False):
    return _tasks(model, limit or None, check_batching)


# ---------------------------------------------------------------- driver
def _store(name, res):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "smoke":
        model = sys.argv[2] if len(sys.argv) > 2 else "bf16"
        res = json.loads(tasks.remote(model, 8, True))
        _store(f"smoke3_tasks_{model}", res)
        for k, t in res["tasks"].items():
            print(k, t["correct"], "/", t["n"], "batched==single:", t.get("batched_vs_single_identical"),
                  f"{t['gen_secs']:.0f}s")
            print("  sample:", repr(t["outputs"][0][:400]))
    elif which in ("tables_a", "tables_b"):
        fn = {"tables_a": tables_a, "tables_b": tables_b}[which]
        for res in json.loads(fn.remote()):
            n = res["config"]["name"]
            _store("exp3_" + n.replace(" ", "_").replace("+", "plus").replace(",", ""), res)
    elif which == "tasks":
        model = sys.argv[2]
        _store(f"exp3_tasks_{model}", json.loads(tasks.remote(model)))
