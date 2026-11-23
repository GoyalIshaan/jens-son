"""Experiment set 5: can the activation scale be cheaper? (follows note 24, progression step 1)

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp5.py smoke      # 512 tokens per text, checks it all runs
    $BEAM beam_exp5.py run        # 8,192 tokens of Python code and of English prose

The matrix unit's scale step costs 37% of each row's logic when the activation scale is a full fp32
number (note 24). Two cheaper ideas, both tested against today's format (int8, fp32 scale = amax/127 per 32):
  1. bf16-style scale: keep only an 8-bit mantissa (rounded to nearest and clamped, or rounded up)
  2. power-of-two scale: a shift instead of a multiply (rounded up; and OCP MX style, 2^(floor(log2 amax) - 6))
Each is run with bf16 weights (isolates the activation format) and with the chip's weights
(GPTQ lut4_g16_fp8_t8 + int8 embedding, int8 KV per head). Reported against the bf16 reference model,
and *paired* against today's activation format on the same positions.
"""
import json
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, eval_texts, image, save, strict_fp32, weights
from beam_exp2 import calib_sets, weight_rmse

MODEL_DIR = "/weights/Qwen3.5-9B"
KW = dict(cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
ACT = [("fp32 scale (today)", "int8_g32"),
       ("bf16 scale, nearest", "int8_g32_bf16s"),
       ("bf16 scale, rounded up", "int8_g32_bf16up"),
       ("power-of-two scale, rounded up", "int8_g32_p2"),
       ("power-of-two scale, OCP MX", "int8_g32_mx")]
BASE = "int8_g32"


def per_position(gold, ids, chunk=512):
    """Log-probs for every position, kept on CPU in chunks."""
    import torch
    h = gold.forward(ids, gold.new_cache(), return_hidden=True)
    return [torch.log_softmax(gold.logits(h[c:c + chunk]), -1).cpu() for c in range(0, h.shape[0], chunk)]


def kl_nll(ref, cand, ids):
    """Per-position KL(ref || cand) and NLL of the true next token, as 1-D CPU tensors."""
    import torch
    kl, nll, top1 = [], [], []
    pos = 0
    for r, q in zip(ref, cand):
        r, q = r.cuda(), q.cuda()
        kl.append((r.exp() * (r - q)).sum(-1).cpu())
        top1.append((r.argmax(-1) == q.argmax(-1)).float().cpu())
        tgt = ids[pos + 1: pos + 1 + r.shape[0]].cuda()
        nll.append(-q[:len(tgt)].gather(1, tgt[:, None])[:, 0].cpu())
        pos += r.shape[0]
    return torch.cat(kl), torch.cat(nll), torch.cat(top1)


def summary(kl, nll, top1, nll_ref):
    n = len(kl)
    return dict(kl_mean=kl.mean().item(), kl_stderr=(kl.std() / n ** 0.5).item(), kl_p99=kl.quantile(0.99).item(),
                top1_agree=top1.mean().item(),
                ppl_change_pct=100 * (nll.mean().exp() / nll_ref.mean().exp() - 1).item(), n=n)


def paired(a, b):
    d = a - b
    return dict(mean=d.mean().item(), stderr=(d.std() / len(d) ** 0.5).item())


def _run(n_tokens, smoke):
    import torch
    from transformers import AutoTokenizer
    from gptq import quantize_model_gptq
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = {k: torch.tensor(v, device="cuda") for k, v in eval_texts(tok, n_tokens).items()}
    gold = Golden(MODEL_DIR)
    ref = {k: per_position(gold, v) for k, v in texts.items()}
    nll_ref = {k: kl_nll(ref[k], ref[k], v)[1] for k, v in texts.items()}
    out = dict(kind="exp5_act_scale", env=environment(), smoke=smoke, n_tokens={k: len(v) for k, v in texts.items()},
               formats=ACT, conditions={})

    def evaluate(cond, extra):
        res = {}
        for name, fmt in ACT:
            gold.num = Numerics(**extra, act=fmt)          # weights already in place; only rounding changes
            t0 = time.time()
            res[fmt] = {}
            for k, v in texts.items():
                kl, nll, top1 = kl_nll(ref[k], per_position(gold, v), v)
                res[fmt][k] = dict(s=summary(kl, nll, top1, nll_ref[k]), kl=kl, nll=nll)
            print(cond, name, {k: round(x["s"]["kl_mean"], 5) for k, x in res[fmt].items()},
                  f"{time.time() - t0:.0f}s", flush=True)
        rows = {}
        for name, fmt in ACT:
            rows[fmt] = dict(name=name, sets={})
            for k in texts:
                x, b = res[fmt][k], res[BASE][k]
                rows[fmt]["sets"][k] = dict(**x["s"], kl_vs_today=paired(x["kl"], b["kl"]),
                                            nll_vs_today=paired(x["nll"], b["nll"]))
        out["conditions"][cond] = dict(numerics_extra={k: v for k, v in extra.items()}, rows=rows)

    # 1. bf16 weights, fp32 everything else: only the activation format changes
    evaluate("bf16 weights", {})
    # 2. the chip's weights (GPTQ) and KV
    wkw = dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})
    calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=8 if smoke else 64,
                                                                 seq_len=128 if smoke else 512)]
    t0 = time.time()
    quantize_model_gptq(gold, Numerics(**wkw), calib, log=lambda m: print(m, flush=True))
    out["gptq_secs"] = time.time() - t0
    out["main_weight_rel_rmse"] = weight_rmse(gold)
    evaluate("chip weights + int8 KV", dict(**wkw, kv="int8_head"))
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke5_act_scale" if smoke else "exp5_act_scale", out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], **KW)
def act_scale(smoke: bool = False):
    return _run(512 if smoke else 8192, smoke)


# ---------------------------------------------------------------- tasks: GSM8K + HumanEval per activation format
TASK_FORMATS = [("fp32 scale (today)", "int8_g32"),
                ("power-of-two scale, rounded up", "int8_g32_p2"),
                ("bf16 scale, nearest", "int8_g32_bf16s")]


def patch_linears(hf):
    """Make every Linear of the language model (and the LM head) round its input with the activation format
    in FMT[0] before the matmul, as the chip does. The rounded input is split into a bf16 high part and a bf16
    low part and both are multiplied (fp32 accumulation), so the input reaches the matmul at about fp32
    precision for every format alike. Otherwise casting to bf16 would blur the scale-precision difference."""
    import torch
    import torch.nn.functional as F
    from formats import quant_act
    FMT = [None]
    n = 0
    for name, mod in hf.named_modules():
        if not isinstance(mod, torch.nn.Linear) or "visual" in name:
            continue

        def fwd(x, mod=mod):
            fmt = FMT[0]
            if fmt is None:
                return F.linear(x, mod.weight, mod.bias)
            xq = quant_act(x.float(), fmt)
            hi = xq.to(torch.bfloat16)
            lo = (xq - hi.float()).to(torch.bfloat16)
            y = F.linear(hi, mod.weight) + F.linear(lo, mod.weight)
            return y if mod.bias is None else y + mod.bias
        mod.forward = fwd
        n += 1
    return FMT, n


def _act_tasks(smoke):
    import torch
    from transformers import AutoTokenizer
    from beam_exp3 import (HE_STOPS, fake_quant, generate, gsm8k_score, gsm8k_task, humaneval_score,
                           humaneval_task, load_hf)
    ensure_model()
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    limit = 8 if smoke else None
    wkind = ("rtn", dict(weights={"default": "int4_g32"})) if smoke else \
        ("gptq", dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"}))
    out = dict(kind="exp5_act_tasks", env=environment(), smoke=smoke, weights=wkind[1],
               arithmetic=("Hugging Face generate, bf16 weights (the chip's format, expanded), greedy, batched; "
                           "every Linear input rounded with the activation format, matmul input at ~fp32 precision"),
               formats={})
    t0 = time.time()
    fq, info = fake_quant(*wkind)
    info["secs"] = time.time() - t0
    out["quant"] = {k: v for k, v in info.items() if k != "weight_rel_rmse"}
    hf = load_hf(fq)
    del fq
    FMT, n_lin = patch_linears(hf)
    out["n_patched_linears"] = n_lin
    print("patched", n_lin, "linears", flush=True)
    gp, golds = gsm8k_task(limit)
    hr, hp = humaneval_task(limit)
    for label, fmt in TASK_FORMATS:
        FMT[0] = fmt
        res = dict(label=label, tasks={})
        for name, prompts, max_new, stops in (("gsm8k", gp, 384, ["Question:", "\n\n\n"]),
                                              ("humaneval", hp, 512, HE_STOPS)):
            t0 = time.time()
            outs, n_gen = generate(hf, tok, prompts, max_new, stops, 32)
            ok = gsm8k_score(outs, golds) if name == "gsm8k" else humaneval_score(hr, outs)
            acc = sum(ok) / len(ok)
            res["tasks"][name] = dict(n=len(ok), correct=int(sum(ok)), accuracy=acc,
                                      stderr=(acc * (1 - acc) / len(ok)) ** 0.5, per_problem=[bool(x) for x in ok],
                                      outputs=outs, gen_secs=time.time() - t0, generated_tokens=n_gen)
            print(f"{label} / {name}: {sum(ok)}/{len(ok)} = {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        out["formats"][fmt] = res
        save("smoke5_act_tasks" if smoke else "exp5_act_tasks", out)          # saved after every format
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke5_act_tasks" if smoke else "exp5_act_tasks", out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=21600,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=21600, timeout=21600))
def act_tasks(smoke: bool = False):
    return _act_tasks(smoke)


def _store(name, res):
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "smoke":
        res = json.loads(act_scale.remote(True))
        _store("smoke5_act_scale", res)
        for cond, c in res["conditions"].items():
            print(cond)
            for fmt, r in c["rows"].items():
                print(" ", r["name"], {k: (round(s["kl_mean"], 5), round(s["kl_vs_today"]["mean"], 5))
                                       for k, s in r["sets"].items()})
    elif which == "run":
        _store("exp5_act_scale", json.loads(act_scale.remote(False)))
    elif which == "tasks_smoke":
        res = json.loads(act_tasks.remote(True))
        _store("smoke5_act_tasks", res)
        print("patched", res["n_patched_linears"])
        for fmt, r in res["formats"].items():
            print(r["label"], {k: f"{t['correct']}/{t['n']}" for k, t in r["tasks"].items()})
    elif which == "tasks":
        _store("exp5_act_tasks", json.loads(act_tasks.remote(False)))
