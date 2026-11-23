"""Experiment set 4: the last model-side numbers spec v1 needs.

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp4.py mtp_smoke     # 6 sequences, all variants: checks the MTP head is wired right
    $BEAM beam_exp4.py mtp bf16      # MTP acceptance, published weights
    $BEAM beam_exp4.py mtp chip      # MTP acceptance, exact chip numerics (main GPTQ + MTP 4-bit, act/KV int8)
    $BEAM beam_exp4.py long128k      # KV formats at 128K context (bf16 weights, fp32 state)

1. MTP acceptance. Speculative decoding (D-013) and the 4-token-row matrix unit (D-015) were
   sized on *assumed* acceptance rates (0.6-0.8). Here we measure, on four kinds of text the
   model generated itself (chat, maths, two code sets):
   - how often draft 1, 2, 3 match what the main model would emit (greedy)
   - expected tokens per verify step for k = 1, 2, 3 drafts
   - the same with the draft's LM head trimmed to the 16K / 32K / 64K most frequent tokens
2. 128K context: does int8 KV (one scale per head) stay free, and int4 KV stay flat, 4x past 32K?

Results land in golden/results/exp4_*.json; design/15_measurements.md explains them.
"""
import json
import os
import random
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, eval_texts, image, save, strict_fp32, weights
from beam_exp2 import ANY_GPU, calib_sets, compare_positions, run_positions, weight_rmse

MODEL_DIR = "/weights/Qwen3.5-9B"
KW = dict(cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
K_MAX = 3
VOCAB_SIZES = (16384, 32768, 65536)
VARIANTS = [("emb_first", "post"), ("emb_first", "pre"), ("hid_first", "post"), ("hid_first", "pre")]
N_PER_SET = {"chat": 100, "gsm8k": 100, "humaneval": 164, "mbpp": 100}


# ---------------------------------------------------------------- data
def chat_sequences(tok, n):
    """Alpaca instructions (no input field), chat template, thinking off; bf16 HF greedy, 256 tokens."""
    import torch
    from datasets import load_dataset
    from beam_exp3 import generate, load_hf
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rows = [r for r in ds.select(range(3000)) if not r["input"]][:n]
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    tok.padding_side = "left"
    hf = load_hf(None)
    outs, _ = generate(hf, tok, prompts, 256, ["<|im_end|>"], 32)
    del hf
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return list(zip(prompts, outs))


def task_sequences(tok, n_per):
    """Prompts + the bf16 model's own greedy completions from experiment set 3."""
    from beam_exp3 import gsm8k_task, humaneval_task, mbpp_task
    ex = json.load(open("/weights/results/exp3_tasks_bf16.json"))["tasks"]
    out = {}
    prompts = {"gsm8k": gsm8k_task()[0], "humaneval": humaneval_task()[1], "mbpp": mbpp_task()[1]}
    for k, ps in prompts.items():
        out[k] = list(zip(ps, ex[k]["outputs"]))[:n_per[k]]
    return out


def vocab_ranking(tok):
    """Token frequency on text *not* used for evaluation: WikiText-2 train, stdlib calibration files,
    GSM8K train, MBPP train, Alpaca rows 3000-13000. Special tokens always included."""
    import collections
    from datasets import load_dataset
    cnt = collections.Counter()
    texts = []
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
    texts.append("\n".join(wiki))
    import json as _j
    root = os.path.dirname(os.path.dirname(_j.__file__))
    from beam_exp2 import CALIB_CODE_FILES
    texts += [open(os.path.join(root, f)).read() for f in CALIB_CODE_FILES]
    g = load_dataset("openai/gsm8k", "main", split="train")
    texts += [q + "\n" + a for q, a in zip(g["question"], g["answer"])]
    m = load_dataset("google-research-datasets/mbpp", "full", split="train")
    texts += [t + "\n" + c for t, c in zip(m["text"], m["code"])]
    a = load_dataset("tatsu-lab/alpaca", split="train").select(range(3000, 13000))
    texts += [x + "\n" + y for x, y in zip(a["instruction"], a["output"])]
    for t in texts:
        cnt.update(tok(t)["input_ids"])
    ranked = [t for t, _ in cnt.most_common()]
    special = [i for i in tok.all_special_ids if i not in set(ranked[:1000])]
    return special + ranked, sum(cnt.values())


# ---------------------------------------------------------------- scoring
def score_sequence(gold, mtp, ids, start, vocabs, k_max=K_MAX, variants=None):
    """One sequence. Returns per-vocab accepted-count lists for valid positions (+ variant draft-1 hits)."""
    import torch
    T = len(ids)
    pre, post = gold.forward(ids, gold.new_cache(), return_prenorm=True)
    gmain = torch.cat([gold.logits(post[r:r + 256]).argmax(-1) for r in range(0, T, 256)])  # predicts ids[i+1]
    match = torch.zeros(T, dtype=torch.bool, device=ids.device)
    match[:-1] = ids[1:] == gmain[:-1]
    # valid start i: in the generated region, the k_max tokens after it are what the main model emits
    lo, hi = max(start - 1, 0), T - 1 - k_max
    valid = [i for i in range(lo, hi + 1) if bool(match[i:i + k_max].all())]
    res = dict(n_region=max(0, hi - lo + 1), n_valid=len(valid))
    if not valid:
        return res
    vi = torch.tensor(valid, device=ids.device)
    tgt = torch.stack([gmain[vi + j + 1] for j in range(k_max)])            # [k, V]
    E = T - 1                                                               # entries need ids[i+1]
    Hsel = post if mtp.hidden == "post" else pre
    for name, vocab in vocabs.items():
        d = mtp.drafts(Hsel[:E], ids[1:E + 1], k_max, vocab)[:, vi]         # [k, V]
        hit = d == tgt
        acc = torch.cumprod(hit.int(), 0).sum(0)                            # leading correct drafts
        res[name] = dict(accepted=acc.tolist(), hit_by_depth=hit.float().sum(1).tolist(),
                         in_vocab=(None if vocab is None else
                                   torch.isin(tgt[0], vocab).float().sum().item()))
    if variants:
        res["variants"] = {}
        keep = (mtp.order, mtp.hidden)
        for order, hid in variants:
            mtp.order, mtp.hidden = order, hid
            H = post if hid == "post" else pre
            d1 = mtp.drafts(H[:E], ids[1:E + 1], 1)[0, vi]
            res["variants"][f"{order}/{hid}"] = (d1 == tgt[0]).float().sum().item()
        mtp.order, mtp.hidden = keep
    return res


def summarise(rows, names):
    out = {}
    for name in names:
        acc = [a for r in rows if name in r for a in r[name]["accepted"]]
        n = len(acc)
        if not n:
            continue
        s = dict(n_positions=n)
        hits = [sum(r[name]["hit_by_depth"][j] for r in rows if name in r) for j in range(K_MAX)]
        s["p_draft1"] = sum(a >= 1 for a in acc) / n
        # conditional acceptance of draft j given drafts < j accepted
        s["p_cond"] = [sum(a >= j + 1 for a in acc) / max(1, sum(a >= j for a in acc)) for j in range(K_MAX)]
        s["tokens_per_step"] = {k: 1 + sum(min(a, k) for a in acc) / n for k in range(1, K_MAX + 1)}
        mu = s["tokens_per_step"][K_MAX]
        s["tokens_per_step_k3_stderr"] = (sum((1 + min(a, 3) - mu) ** 2 for a in acc) / (n - 1) / n) ** 0.5
        s["hit_by_depth_unconditional"] = [h / n for h in hits]
        iv = [r[name]["in_vocab"] for r in rows if name in r and r[name]["in_vocab"] is not None]
        if iv:
            s["target_in_vocab"] = sum(iv) / n
        out[name] = s
    return out


def _mtp(cond, smoke=False):
    import torch
    from transformers import AutoTokenizer
    from qwen35_golden import Golden, Numerics
    from mtp import MTP
    strict_fp32()
    ensure_model()
    t_all = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    n_per = {k: 6 if smoke else v for k, v in N_PER_SET.items()}
    if smoke:
        n_per = {"chat": 2, "gsm8k": 2, "humaneval": 2, "mbpp": 2}
    t0 = time.time()
    sets = {"chat": chat_sequences(tok, n_per["chat"])}
    sets.update(task_sequences(tok, n_per))
    ranked, n_freq_tokens = vocab_ranking(tok)
    data_secs = time.time() - t0

    gold = Golden(MODEL_DIR)
    out = dict(kind="exp4_mtp", cond=cond, env=environment(), smoke=smoke, data_secs=data_secs,
               vocab_freq_tokens=n_freq_tokens)
    mtp_fmt = None
    if cond == "chip":
        from gptq import quantize_model_gptq
        wkw = dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})
        calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=8 if smoke else 64,
                                                                     seq_len=128 if smoke else 512)]
        quantize_model_gptq(gold, Numerics(**wkw), calib, log=lambda m: print(m, flush=True))
        out["main_weight_rel_rmse"] = weight_rmse(gold)
        gold.num = Numerics(**wkw, act="int8_g32", kv="int8_head")
        mtp_fmt = "lut4_g16_fp8_t8"
    out["numerics"] = gold.num.describe()
    mtp = MTP(gold, MODEL_DIR, fmt=mtp_fmt)
    out["mtp"] = dict(n_tensors=mtp.n_tensors, matrix_params=mtp.matrix_params, shapes=mtp.shapes,
                      norm_mean=mtp.norm_mean, main_norm_mean=mtp.main_norm_mean, weight_rel_rmse=mtp.weight_error)
    print("mtp", out["mtp"]["matrix_params"], mtp.norm_mean, flush=True)

    vocabs = {"full": None}
    for v in VOCAB_SIZES:
        vocabs[f"top{v // 1024}k"] = torch.tensor(sorted(ranked[:v]), device="cuda")

    # pick the wiring from data: draft-1 hit rate of the 4 variants on the first sequences of each set
    var_rows = []
    for name, seqs in sets.items():
        for prompt, comp in seqs[:3 if smoke else 12]:
            ids = torch.tensor(tok(prompt + comp)["input_ids"], device="cuda")
            r = score_sequence(gold, mtp, ids, len(tok(prompt)["input_ids"]), {}, 1, VARIANTS)
            if r.get("variants"):
                var_rows.append(r)
    tot = sum(r["n_valid"] for r in var_rows)
    out["variants"] = {k: sum(r["variants"][k] for r in var_rows) / max(tot, 1) for k in var_rows[0]["variants"]}
    out["variants_n_positions"] = tot
    best = max(out["variants"], key=out["variants"].get)
    mtp.order, mtp.hidden = best.split("/")
    out["chosen_variant"] = best
    print("variants", out["variants"], "->", best, flush=True)

    out["sets"], all_rows = {}, []
    for name, seqs in sets.items():
        t0 = time.time()
        rows = []
        for prompt, comp in seqs:
            ids = torch.tensor(tok(prompt + comp)["input_ids"], device="cuda")
            rows.append(score_sequence(gold, mtp, ids, len(tok(prompt)["input_ids"]), vocabs))
        all_rows += rows
        out["sets"][name] = dict(n_seq=len(rows), n_region=sum(r["n_region"] for r in rows),
                                 n_valid=sum(r["n_valid"] for r in rows), secs=time.time() - t0,
                                 **summarise(rows, vocabs))
        print(name, json.dumps({k: (round(v["p_draft1"], 3), {a: round(b, 3) for a, b in v["tokens_per_step"].items()})
                                for k, v in out["sets"][name].items() if isinstance(v, dict) and "p_draft1" in v}),
              flush=True)
    out["all"] = summarise(all_rows, vocabs)
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save(("smoke4_mtp_" if smoke else "exp4_mtp_") + cond, out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], **KW)
def mtp_job(cond: str, smoke: bool = False):
    return _mtp(cond, smoke)


# ---------------------------------------------------------------- 128K context
LONG128 = [("kv int8 per-head", dict(kv="int8_head")), ("kv int4_g32", dict(kv="int4_g32"))]
BUCKETS128 = ((0, 8192), (8192, 32768), (32768, 65536), (65536, 98304), (98304, 131072))


@function(gpu="RTX5090", **KW)
def long128k(n_tokens: int = 131072, every: int = 32):
    import torch
    import beam_exp2
    from transformers import AutoTokenizer
    from qwen35_golden import Golden, Numerics
    strict_fp32()
    ensure_model()
    beam_exp2.BUCKETS = BUCKETS128
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = torch.tensor(eval_texts(tok, n_tokens)["wikitext"], device="cuda")
    pos = torch.arange(0, len(ids) - 1, every, device="cuda")
    gold = Golden(MODEL_DIR)
    t0 = time.time()
    ref = run_positions(gold, ids, pos)
    out = dict(kind="exp4_long128k", env=environment(), n_tokens=len(ids), scored_positions=len(pos),
               weights="bf16, fp32 state (isolates the KV format)", ref_secs=time.time() - t0, configs=[])
    print(f"reference {out['ref_secs']:.0f}s", flush=True)
    for name, kw in LONG128:
        gold.set_numerics(Numerics(**kw))
        t0 = time.time()
        row = dict(name=name, numerics=gold.num.describe(),
                   result=compare_positions(ref, run_positions(gold, ids, pos), ids.cpu(), pos.cpu()))
        row["secs"] = time.time() - t0
        out["configs"].append(row)
        print(name, round(row["result"]["kl_mean"], 5),
              {k: round(v["kl"], 5) for k, v in row["result"]["by_position"].items()}, flush=True)
        torch.cuda.empty_cache()
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("exp4_long128k", out)
    return json.dumps(out)


# ---------------------------------------------------------------- one-pass DeltaNet at 32K
@function(gpu=["RTX5090", "RTX4090"], **KW)
def onepass32k(n_tokens: int = 32768, every: int = 8):
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
    out = dict(kind="exp4_onepass32k", env=environment(), n_tokens=len(ids), scored_positions=len(pos),
               ref_secs=time.time() - t0, configs=[])
    for name, kw in (("state fp32, one-pass order (chip)", dict(state="fp32_onepass")),):
        gold.set_numerics(Numerics(**kw))
        t0 = time.time()
        cand = run_positions(gold, ids, pos)
        r = torch.cat(ref)
        q = torch.cat(cand)
        row = dict(name=name, numerics=gold.num.describe(), result=compare_positions(ref, cand, ids.cpu(), pos.cpu()),
                   logprob_max_abs_diff=(r - q).abs().max().item(),
                   top1_identical=(r.argmax(-1) == q.argmax(-1)).float().mean().item(), secs=time.time() - t0)
        out["configs"].append(row)
        print(name, row["result"]["kl_mean"], row["logprob_max_abs_diff"], row["top1_identical"], flush=True)
    save("exp4_onepass32k", out)
    return json.dumps(out)


# ---------------------------------------------------------------- windowed draft head at 32K
WINDOWS = (None, 16384, 4096, 1024, 256)
LONG_CODE_FILES = ["inspect.py", "logging/__init__.py", "threading.py", "os.py", "pickle.py", "doctest.py",
                   "configparser.py", "csv.py", "difflib.py", "imaplib.py", "smtplib.py", "tarfile.py"]


def long_sequences(tok, n):
    """Two prose slices of WikiText-2 test and one stdlib-Python sequence (files used nowhere else), n tokens each."""
    import json as _json
    wiki = eval_texts(tok, None)["wikitext"]
    root = os.path.dirname(os.path.dirname(_json.__file__))
    code = ""
    for f in LONG_CODE_FILES:
        p = os.path.join(root, f)
        if os.path.exists(p):
            code += f"# file: {f}\n" + open(p).read() + "\n\n"
    code_ids = tok(code)["input_ids"]
    return {"prose_a": wiki[:n], "prose_b": wiki[2 * n:3 * n], "code": code_ids[:n]}


def _window_rows(acc_by_w, names):
    """Per-window summary + paired difference vs the full window on tokens/step at k=3."""
    out = {}
    full = acc_by_w["full"]
    for w in names:
        acc = acc_by_w[w]
        n = len(acc)
        s = dict(n_positions=n,
                 p_cond=[sum(a >= j + 1 for a in acc) / max(1, sum(a >= j for a in acc)) for j in range(K_MAX)],
                 tokens_per_step={k: 1 + sum(min(a, k) for a in acc) / n for k in range(1, K_MAX + 1)})
        d = [min(a, 3) - min(b, 3) for a, b in zip(acc, full)]
        mu = sum(d) / n
        s["delta_k3_vs_full"] = mu
        s["delta_k3_stderr"] = (sum((x - mu) ** 2 for x in d) / (n - 1) / n) ** 0.5
        s["identical_to_full"] = sum(a == b for a, b in zip(acc, full)) / n
        out[w] = s
    return out


@function(gpu=["RTX5090", "RTX4090"], **KW)
def mtp_window(n_tokens: int = 32768, start: int = 8192, stride: int = 4, smoke: bool = False):
    """Does the MTP draft head keep its acceptance if its attention only sees the last W positions?
    bf16 weights (experiment set 4 showed the chip format changes acceptance by < 0.01 tokens/step).
    Positions in [start, n) whose next 3 tokens are what the main model emits; every `stride`-th one."""
    import torch
    from transformers import AutoTokenizer
    from qwen35_golden import Golden
    from mtp import MTP
    strict_fp32()
    ensure_model()
    t_all = time.time()
    if smoke:
        n_tokens, start, stride = 4096, 1024, 2
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    seqs = long_sequences(tok, n_tokens)
    ranked, _ = vocab_ranking(tok)
    vocab = torch.tensor(sorted(ranked[:32768]), device="cuda")
    gold = Golden(MODEL_DIR)
    mtp = MTP(gold, MODEL_DIR)
    out = dict(kind="exp4_mtp_window", env=environment(), smoke=smoke, n_tokens=n_tokens, start=start,
               stride=stride, windows=[("full" if w is None else str(w)) for w in WINDOWS], vocab="top32k", numerics=gold.num.describe(),
               seq_tokens={k: len(v) for k, v in seqs.items()}, sets={})

    # check: chunked drafts_at (no window) == drafts() on a short sequence
    ids = torch.tensor(seqs["prose_a"][:1024], device="cuda")
    _, post = gold.forward(ids, gold.new_cache(), return_prenorm=True)
    idx = torch.arange(0, 1020, 3, device="cuda")
    a = mtp.drafts(post[:-1], ids[1:], K_MAX, vocab)[:, idx]
    b = mtp.drafts_at(post[:-1], ids[1:], idx, K_MAX, vocab)
    c = mtp.drafts_at(post[:-1], ids[1:], idx, K_MAX, vocab, window=4096)
    out["check_drafts_at_equals_drafts"] = (a == b).float().mean().item()
    out["check_window_beyond_length_equals"] = (b == c).float().mean().item()
    print("check", out["check_drafts_at_equals_drafts"], out["check_window_beyond_length_equals"], flush=True)

    all_acc = {("full" if w is None else str(w)): [] for w in WINDOWS}
    for name, seq in seqs.items():
        t0 = time.time()
        ids = torch.tensor(seq, device="cuda")
        T = len(ids)
        cache, H = gold.new_cache(), []
        for p0 in range(0, T, 4096):
            H.append(gold.forward(ids[p0:p0 + 4096], cache, return_hidden=True))
        H = torch.cat(H)
        del cache
        torch.cuda.empty_cache()
        gmain = torch.cat([gold.logits(H[r:r + 256]).argmax(-1) for r in range(0, T, 256)])
        match = torch.zeros(T, dtype=torch.bool, device="cuda")
        match[:-1] = ids[1:] == gmain[:-1]
        valid = [i for i in range(start, T - 1 - K_MAX) if bool(match[i:i + K_MAX].all())]
        sel = valid[::stride]
        vi = torch.tensor(sel, device="cuda")
        tgt = torch.stack([gmain[vi + j + 1] for j in range(K_MAX)])
        acc_by_w = {}
        for w in WINDOWS:
            d = mtp.drafts_at(H[:T - 1], ids[1:], vi, K_MAX, vocab, window=w)
            acc_by_w[("full" if w is None else str(w))] = torch.cumprod((d == tgt).int(), 0).sum(0).tolist()
            all_acc[("full" if w is None else str(w))] += acc_by_w[("full" if w is None else str(w))]
        out["sets"][name] = dict(n_region=T - 1 - K_MAX - start, n_valid=len(valid), n_scored=len(sel),
                                 main_top1_match=match[start:].float().mean().item(), secs=time.time() - t0,
                                 **_window_rows(acc_by_w, list(acc_by_w)))
        print(name, {w: round(v["tokens_per_step"][3], 3) for w, v in out["sets"][name].items()
                     if isinstance(v, dict) and "tokens_per_step" in v}, flush=True)
        del H
        torch.cuda.empty_cache()
    out["all"] = _window_rows(all_acc, list(all_acc))
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke4_mtp_window" if smoke else "exp4_mtp_window", out)
    return json.dumps(out)


# ---------------------------------------------------------------- driver
def _store(name, res):
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "mtp_smoke":
        res = json.loads(mtp_job.remote(sys.argv[2] if len(sys.argv) > 2 else "bf16", True))
        _store("smoke4_mtp_" + res["cond"], res)
        print(json.dumps({k: v for k, v in res.items() if k not in ("mtp",)}, indent=1)[:5000])
    elif which == "mtp":
        _store("exp4_mtp_" + sys.argv[2], json.loads(mtp_job.remote(sys.argv[2])))
    elif which == "onepass32k":
        _store("exp4_onepass32k", json.loads(onepass32k.remote()))
    elif which == "long128k":
        _store("exp4_long128k", json.loads(long128k.remote()))
    elif which == "mtp_window_smoke":
        res = json.loads(mtp_window.remote(smoke=True))
        _store("smoke4_mtp_window", res)
        print(json.dumps(res, indent=1)[:4000])
    elif which == "mtp_window":
        _store("exp4_mtp_window", json.loads(mtp_window.remote()))
