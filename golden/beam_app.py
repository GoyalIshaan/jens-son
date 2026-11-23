"""Runs the golden model on a Beam GPU and records every measurement.

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_app.py validate    # our model vs Hugging Face, fp32 both
    $BEAM beam_app.py quality     # every number format vs full precision, prose + code
    $BEAM beam_app.py long        # KV-cache and state formats at 8K context

Each run returns a JSON dict that is saved locally to golden/results/<name>.json
(and on the Beam volume under /weights/results/). design/15_measurements.md
explains every number.

GPU: an RTX 5090 (32 GB), the largest GPU Beam offers serverless here. Hugging
Face's fp32 reference needs ~38 GB, so it runs on the CPU (64 GB RAM).
"""
import json
import os
import sys
import time

from beam import Image, Volume, function

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_DIR = "/weights/Qwen3.5-9B"
GPU = "RTX5090"

image = Image(
    python_version="python3.11",
    python_packages=["torch==2.8.0", "transformers==5.22.2", "safetensors", "huggingface_hub",
                     "accelerate", "datasets", "numpy", "psutil"],
)
weights = Volume(name="qwen35-weights", mount_path="/weights")


# ---------------------------------------------------------------- helpers
def ensure_model():
    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors.index.json")):
        print("downloading", MODEL_ID, flush=True)
        t0 = time.time()
        snapshot_download(MODEL_ID, local_dir=MODEL_DIR,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*",
                                          "vocab*", "merges*"])
        print(f"downloaded in {time.time()-t0:.0f}s", flush=True)
    return MODEL_DIR


def environment():
    import platform
    import psutil
    import torch
    import transformers
    return dict(
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        gpu_mem_GB=torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None,
        cpu_count=os.cpu_count(), host_ram_GB=psutil.virtual_memory().total / 1e9,
        torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
        python=platform.python_version(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model=MODEL_ID,
        model_revision=json.load(open(os.path.join(MODEL_DIR, "config.json"))).get("transformers_version"),
    )


def save(name, result):
    os.makedirs("/weights/results", exist_ok=True)
    with open(f"/weights/results/{name}.json", "w") as f:
        json.dump(result, f, indent=1)


def strict_fp32():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def eval_texts(tok, n):
    """Two domains: English prose (WikiText-2 test) and Python source (stdlib)."""
    import json as _json
    from datasets import load_dataset
    wiki = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    root = os.path.dirname(os.path.dirname(_json.__file__))
    code = ""
    for f in ["json/decoder.py", "json/encoder.py", "textwrap.py", "heapq.py", "bisect.py",
              "fractions.py", "statistics.py", "dataclasses.py"]:
        code += f"# file: {f}\n" + open(os.path.join(root, f)).read() + "\n\n"
    out = {}
    for name, text in (("wikitext", wiki), ("python_code", code)):
        ids = tok(text)["input_ids"]
        out[name] = ids[:n] if isinstance(n, int) else ids
    return out


# ---------------------------------------------------------------- validate
@function(gpu=GPU, cpu=16, memory="64Gi", image=image, volumes=[weights], timeout=7200)
def validate():
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from qwen35_golden import Golden

    strict_fp32()
    ensure_model()
    env = environment()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = {
        "deltanet": ("The Gated DeltaNet layer keeps a fixed-size memory for each head. "
                     "Unlike full attention, its cost per token does not grow with context. "
                     "In one sentence, the main advantage of this design is"),
        "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    }
    N_NEW = 32
    timing = {}

    t0 = time.time()
    gold = Golden(MODEL_DIR)
    timing["golden_load_s"] = time.time() - t0
    G = {}
    for name, text in prompts.items():
        ids = torch.tensor(tok(text)["input_ids"], device="cuda")
        trace = []
        t0 = time.time()
        lg_seq = gold.forward(ids, gold.new_cache(), trace=trace)
        torch.cuda.synchronize()
        timing[f"golden_prefill_{name}_s"] = time.time() - t0
        cache = gold.new_cache()
        t0 = time.time()
        lg_step = torch.cat([gold.forward(ids[i:i + 1], cache) for i in range(len(ids))])
        torch.cuda.synchronize()
        timing[f"golden_stepwise_{name}_s_per_token"] = (time.time() - t0) / len(ids)
        G[name] = dict(ids=ids.cpu(), logits=lg_seq.cpu(), step_logits=lg_step.cpu(), trace=trace,
                       greedy=gold.greedy(ids, N_NEW))
        print(f"golden {name}: {len(ids)} tokens", flush=True)
    timing["golden_gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    del gold
    torch.cuda.empty_cache()

    t0 = time.time()
    torch.set_num_threads(os.cpu_count())
    hf = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.float32, device_map="cpu",
                                                     attn_implementation="eager").eval()
    timing["hf_load_s"] = time.time() - t0
    hf_class = type(hf).__name__

    per_prompt = {}
    for name in prompts:
        g = G[name]
        ids = g["ids"]
        with torch.no_grad():
            t0 = time.time()
            out = hf(input_ids=ids[None], output_hidden_states=True)
            timing[f"hf_cpu_prefill_{name}_s"] = time.time() - t0
            lg_hf = out.logits[0].float()
            hs = [x[0].float() for x in out.hidden_states]
            t0 = time.time()
            hf_new = hf.generate(input_ids=ids[None], max_new_tokens=N_NEW, do_sample=False)[0, len(ids):].tolist()
            timing[f"hf_cpu_generate_{name}_s"] = time.time() - t0

        lg = g["logits"]
        d = (lg - lg_hf).abs()
        lp_hf, lp_g = torch.log_softmax(lg_hf, -1), torch.log_softmax(lg, -1)
        kl = (lp_hf.exp() * (lp_hf - lp_g)).sum(-1)
        top5_hf = lg_hf.topk(5, -1).indices
        top5_g = lg.topk(5, -1).indices
        top5_overlap = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(top5_hf, top5_g)])
        layers = []
        for i in range(min(len(hs), len(g["trace"]))):
            a, b = g["trace"][i], hs[i]
            layers.append(dict(index=i, rel_err=((a - b).norm() / b.norm()).item(),
                               max_abs=(a - b).abs().max().item(), ref_rms=b.pow(2).mean().sqrt().item()))
        per_prompt[name] = dict(
            tokens=len(ids),
            logit_max_abs_diff=d.max().item(), logit_mean_abs_diff=d.mean().item(),
            logit_mean_abs=lg_hf.abs().mean().item(), logit_max_abs=lg_hf.abs().max().item(),
            kl_mean=kl.mean().item(), kl_max=kl.max().item(),
            top1_agree=(lg.argmax(-1) == lg_hf.argmax(-1)).float().mean().item(),
            top5_overlap=top5_overlap.mean().item(),
            prefill_vs_stepwise_max_abs=(g["logits"] - g["step_logits"]).abs().max().item(),
            greedy_tokens_match=g["greedy"] == hf_new,
            greedy_first_mismatch=next((i for i, (a, b) in enumerate(zip(g["greedy"], hf_new)) if a != b), None),
            golden_text=tok.decode(g["greedy"]), hf_text=tok.decode(hf_new),
            hidden_states_compared=len(layers), hf_hidden_states=len(hs), golden_trace=len(g["trace"]),
            per_layer=layers,
        )
        print(name, {k: v for k, v in per_prompt[name].items() if k != "per_layer"}, flush=True)

    result = dict(kind="validate", env=env, hf_class=hf_class, precision="fp32 both; golden on GPU (TF32 off), HF on CPU, eager attention",
                  greedy_new_tokens=N_NEW, timing=timing, prompts=per_prompt)
    save("validate", result)
    return json.dumps(result)


# ---------------------------------------------------------------- quality
BITS = {"int4_g32_mse": 4.5, "int4_g16_fp8": 4.5, "nvfp4_mse": 4.5, "lut4_g32": 4.5, "lut4_g32_mse": 4.5,
        "int4_g32": 4.5, "int4_g128": 4 + 16 / 128, "int3_g32": 3.5, "mxfp4": 4.25, "nvfp4": 4.5}
KV_BYTES = {None: 2.0, "int8_g64": 1 + 2 / 64, "int4_g32": 0.5 + 2 / 32}

EVAL_CONFIGS = [
    ("int4_g32", dict(weights={"default": "int4_g32"})),
    ("int4_g128", dict(weights={"default": "int4_g128"})),
    ("mxfp4", dict(weights={"default": "mxfp4"})),
    ("nvfp4", dict(weights={"default": "nvfp4"})),
    ("int3_g32 everywhere", dict(weights={"default": "int3_g32"})),
    ("ffn int3_g32, rest int4_g32", dict(weights={"default": "int4_g32", "ffn": "int3_g32"})),
    ("int4_g32, LM head kept bf16", dict(weights={"default": "int4_g32", "lm_head": None})),
    ("int4_g32 + act int8_g32", dict(weights={"default": "int4_g32"}, act="int8_g32")),
    ("int4_g32 + kv int8_g64", dict(weights={"default": "int4_g32"}, kv="int8_g64")),
    ("int4_g32 + kv int4_g32", dict(weights={"default": "int4_g32"}, kv="int4_g32")),
    ("int4_g32 + state bf16", dict(weights={"default": "int4_g32"}, state="bf16")),
    ("chip candidate: int4_g32 + act int8 + kv int8 + state fp32",
     dict(weights={"default": "int4_g32"}, act="int8_g32", kv="int8_g64")),
]


def predicted_speed(gold, num, ctx=4096):
    """Bytes per token and tok/s on our card (512 GB/s, 92%) from the formats actually used."""
    from qwen35_golden import weight_kind
    wbytes = 0.0
    for name, t in gold.raw.items():
        if name.endswith("embed_tokens.weight"):
            continue                                               # one row per token
        fmt = num.weight_format(name)
        n = t.numel()
        wbytes += n * (BITS[fmt] / 8 if fmt else (2 if weight_kind(name) != "other" else 0))
    kv = 8 * 4 * 256 * 2 * ctx * KV_BYTES[num.kv]
    state = 2 * 24 * 32 * 128 * 128 * (4 if num.state == "fp32" else 2)
    total = wbytes + kv + state
    return dict(weight_GB=wbytes / 1e9, bytes_per_token_GB=total / 1e9, card_tok_s=0.92 * 512e9 / total)


def run_set(gold, ids, chunk=512):
    """Forward once, return log-probs in CPU chunks (fp32)."""
    import torch
    h = gold.forward(ids, gold.new_cache(), return_hidden=True)
    out = []
    for c in range(0, h.shape[0], chunk):
        out.append(torch.log_softmax(gold.logits(h[c:c + chunk]), -1).cpu())
    return out


def compare_sets(ref_chunks, chunks, ids):
    import torch
    kl, top1, top5, nll_ref, nll = [], [], [], [], []
    pos = 0
    for r, q in zip(ref_chunks, chunks):
        r, q = r.cuda(), q.cuda()
        kl.append((r.exp() * (r - q)).sum(-1).cpu())
        top1.append((r.argmax(-1) == q.argmax(-1)).float().cpu())
        rt, qt = r.topk(5, -1).indices, q.topk(5, -1).indices
        top5.append((rt[:, :, None] == qt[:, None, :]).any(-1).float().mean(-1).cpu())
        tgt = ids[pos + 1: pos + 1 + r.shape[0]].cuda()
        m = len(tgt)
        nll_ref.append(-r[:m].gather(1, tgt[:, None])[:, 0].cpu())
        nll.append(-q[:m].gather(1, tgt[:, None])[:, 0].cpu())
        pos += r.shape[0]
        del r, q
    kl, top1, top5 = torch.cat(kl), torch.cat(top1), torch.cat(top5)
    nll_ref, nll = torch.cat(nll_ref), torch.cat(nll)
    n = len(kl)
    return dict(
        kl_mean=kl.mean().item(), kl_stderr=(kl.std() / n ** 0.5).item(), kl_median=kl.median().item(),
        kl_p99=kl.quantile(0.99).item(), kl_max=kl.max().item(),
        top1_agree=top1.mean().item(), top5_overlap=top5.mean().item(),
        ppl=nll.mean().exp().item(), ppl_ref=nll_ref.mean().exp().item(),
        ppl_change_pct=100 * (nll.mean().exp() / nll_ref.mean().exp() - 1).item(),
        nll_diff_mean=(nll - nll_ref).mean().item(), nll_diff_stderr=((nll - nll_ref).std() / len(nll) ** 0.5).item(),
        kl_by_position={f"{a}-{b}": kl[a:b].mean().item() for a, b in
                        ((0, 1024), (1024, 2048), (2048, 4096), (4096, 8192)) if a < n},
    )


def _quality(n_tokens, configs, label):
    import torch
    from transformers import AutoTokenizer
    from qwen35_golden import Golden, Numerics

    strict_fp32()
    ensure_model()
    env = environment()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = eval_texts(tok, n_tokens)
    if label == "long":
        texts = {"wikitext": texts["wikitext"]}
    ids = {k: torch.tensor(v, device="cuda") for k, v in texts.items()}

    gold = Golden(MODEL_DIR)
    out = dict(kind=label, env=env, n_tokens={k: len(v) for k, v in ids.items()},
               rounding="round-to-nearest, no calibration", configs=[])
    t0 = time.time()
    ref = {k: run_set(gold, v) for k, v in ids.items()}
    ref_num = Numerics()
    out["reference"] = dict(desc="bf16 weights as published, fp32 arithmetic, fp32 KV and state",
                            secs=time.time() - t0, predicted=predicted_speed(gold, ref_num),
                            gpu_peak_mem_GB=torch.cuda.max_memory_allocated() / 1e9)
    print("reference done", flush=True)
    for name, kw in configs:
        num = Numerics(**kw)
        t0 = time.time()
        gold.set_numerics(num)
        qsecs = time.time() - t0
        row = dict(name=name, numerics=num.describe(), quantize_secs=qsecs,
                   weight_rel_rmse={k: dict(format=v[3], rel_rmse=(v[0] / v[1]) ** 0.5, weights=v[2])
                                    for k, v in gold.weight_error.items()},
                   predicted=predicted_speed(gold, num), sets={})
        t0 = time.time()
        for k, v in ids.items():
            row["sets"][k] = compare_sets(ref[k], run_set(gold, v), v.cpu())
        row["eval_secs"] = time.time() - t0
        row["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        out["configs"].append(row)
        print(name, json.dumps({k: (s["kl_mean"], s["top1_agree"], s["ppl_change_pct"])
                                for k, s in row["sets"].items()}), flush=True)
        torch.cuda.empty_cache()
    save(label, out)
    return json.dumps(out)


@function(gpu=GPU, cpu=8, memory="64Gi", image=image, volumes=[weights], timeout=10800)
def quality():
    return _quality(2048, EVAL_CONFIGS, "quality")


ROUND2_CONFIGS = [
    ("nvfp4 (round-1 anchor, should reproduce)", dict(weights={"default": "nvfp4"})),
    ("int4_g32_mse", dict(weights={"default": "int4_g32_mse"})),
    ("int4_g16_fp8", dict(weights={"default": "int4_g16_fp8"})),
    ("nvfp4_mse", dict(weights={"default": "nvfp4_mse"})),
    ("lut4_g32 (learned table)", dict(weights={"default": "lut4_g32"})),
    ("lut4_g32_mse (learned table + scale search)", dict(weights={"default": "lut4_g32_mse"})),
]


@function(gpu=GPU, cpu=8, memory="64Gi", image=image, volumes=[weights], timeout=10800)
def quality2():
    return _quality(2048, ROUND2_CONFIGS, "quality2")


LONG_CONFIGS = [
    ("int4_g32", dict(weights={"default": "int4_g32"})),
    ("int4_g32 + kv int8_g64", dict(weights={"default": "int4_g32"}, kv="int8_g64")),
    ("int4_g32 + kv int4_g32", dict(weights={"default": "int4_g32"}, kv="int4_g32")),
    ("int4_g32 + state bf16", dict(weights={"default": "int4_g32"}, state="bf16")),
    ("kv int4_g32 only (bf16 weights)", dict(kv="int4_g32")),
    ("state bf16 only (bf16 weights)", dict(state="bf16")),
]


@function(gpu=GPU, cpu=8, memory="64Gi", image=image, volumes=[weights], timeout=10800)
def long():
    return _quality(8192, LONG_CONFIGS, "long")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "validate"
    fn = {"validate": validate, "quality": quality, "quality2": quality2, "long": long}[which]
    res = json.loads(fn.remote())
    os.makedirs("results", exist_ok=True)
    with open(f"results/{which}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:6000])
