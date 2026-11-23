"""Measures real GPU speed for Qwen3.5-9B at batch 1 with llama.cpp (CUDA).

This replaces the scaled estimates in design/07_competitors_and_goal.md with a
measurement on Qwen3.5-9B itself (DeltaNet layers included).

    cd golden
    ~/.local/share/uv/tools/beam-client/bin/python beam_bench.py gguf     # once: build Q4_0 file
    ~/.local/share/uv/tools/beam-client/bin/python beam_bench.py bench    # run on each GPU

Bandwidth utilization = tok/s x bytes read per token / peak bandwidth.
Bytes per token = GGUF file size minus the token-embedding table (only one row
of it is read per token), plus KV cache at the tested depth (small here).
"""
import json
import sys

from beam import Image, Volume, function

MODEL_DIR = "/weights/Qwen3.5-9B"
GGUF_DIR = "/weights/gguf"

image = Image(
    base_image="nvidia/cuda:12.8.1-devel-ubuntu22.04",
    python_version="python3.11",
    python_packages=["numpy", "sentencepiece", "protobuf", "safetensors", "huggingface_hub",
                     "transformers==5.22.2", "torch==2.8.0"],
    commands=[
        "apt-get update && apt-get install -y git cmake build-essential libcurl4-openssl-dev",
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        "cd /llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='86;89;120' "
        "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && cmake --build build -j 16 --target llama-bench llama-quantize",
    ],
)
weights = Volume(name="qwen35-weights", mount_path="/weights")

# peak DRAM bandwidth from each vendor's spec sheet, GB/s
PEAK_BW = {"A10G": 600.0, "RTX4090": 1008.0, "RTX5090": 1792.0}


def sh(cmd):
    import subprocess
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    return r


@function(gpu="A10G", cpu=8, memory="64Gi", image=image, volumes=[weights], timeout=5400)
def gguf():
    """Convert the HF checkpoint to GGUF and quantize to Q4_0 (same size as our int4_g32)."""
    import os
    os.makedirs(GGUF_DIR, exist_ok=True)
    bf16 = f"{GGUF_DIR}/qwen35-9b-bf16.gguf"
    q4 = f"{GGUF_DIR}/qwen35-9b-Q4_0.gguf"
    commit = sh("cd /llama.cpp && git rev-parse HEAD").stdout.strip()
    if not os.path.exists(bf16):
        r = sh(f"python /llama.cpp/convert_hf_to_gguf.py {MODEL_DIR} --outtype bf16 --outfile {bf16}")
        if r.returncode != 0:
            return json.dumps(dict(ok=False, step="convert", llama_cpp=commit, err=r.stderr[-2000:]))
    if not os.path.exists(q4):
        r = sh(f"/llama.cpp/build/bin/llama-quantize {bf16} {q4} Q4_0")
        if r.returncode != 0:
            return json.dumps(dict(ok=False, step="quantize", llama_cpp=commit, err=r.stderr[-2000:]))
    return json.dumps(dict(ok=True, llama_cpp=commit, files={f: os.path.getsize(f"{GGUF_DIR}/{f}") for f in os.listdir(GGUF_DIR)}))


def _bench(gpu_name):
    import os
    import sys
    q4 = f"{GGUF_DIR}/qwen35-9b-Q4_0.gguf"
    commit = sh("cd /llama.cpp && git rev-parse HEAD").stdout.strip()
    gpu = sh("nvidia-smi --query-gpu=name,memory.total,clocks.max.memory --format=csv,noheader").stdout.strip()
    r = sh(f"/llama.cpp/build/bin/llama-bench -m {q4} -ngl 99 -fa 1 -p 512 -n 128 -d 0,4096,32768 -r 3 -o json")
    if r.returncode != 0:
        return json.dumps(dict(ok=False, gpu=gpu, err=r.stderr[-2000:]))
    rows = json.loads(r.stdout)

    # bytes per token, from the real tensor table in the GGUF file: every tensor is read once per
    # token except the token-embedding table (one row). DeltaNet state (fp32) is read and written,
    # and the f16 KV cache grows with depth.
    sys.path.insert(0, "/llama.cpp/gguf-py")
    from gguf import GGUFReader
    rd = GGUFReader(q4)
    sizes = {t.name: int(t.n_bytes) for t in rd.tensors}
    types = {}
    for t in rd.tensors:
        k = t.tensor_type.name
        types.setdefault(k, [0, 0])
        types[k][0] += 1
        types[k][1] += int(t.n_bytes)
    file_bytes = os.path.getsize(q4)
    emb_bytes = sizes.get("token_embd.weight", 0)
    weights_read = sum(sizes.values()) - emb_bytes
    kv_per_tok = 8 * 4 * 256 * 2 * 2                      # f16 K and V, 8 full-attention layers
    state_rw = 2 * 24 * 32 * 128 * 128 * 4                # fp32 DeltaNet state, read + write
    out = []
    for row in rows:
        tps = row["avg_ts"]
        depth = row.get("n_depth", 0)
        is_tg = row.get("n_gen", 0) > 0
        res = dict(test="tg" if is_tg else "pp", n=row.get("n_gen") or row.get("n_prompt"), depth=depth,
                   tok_s=tps, tok_s_sd=row.get("stddev_ts"))
        if is_tg:
            bytes_tok = weights_read + state_rw + kv_per_tok * (depth + 64)
            res["bytes_per_token_GB"] = bytes_tok / 1e9
            res["utilization"] = tps * bytes_tok / (PEAK_BW[gpu_name] * 1e9)
        out.append(res)
    return json.dumps(dict(ok=True, gpu=gpu, peak_bw_GBs=PEAK_BW[gpu_name], llama_cpp=commit,
                           q4_file_GB=file_bytes / 1e9, weights_read_GB=weights_read / 1e9, embedding_GB=emb_bytes / 1e9,
                           tensor_types={k: dict(count=v[0], GB=v[1] / 1e9) for k, v in types.items()},
                           lm_head_type=next((t.tensor_type.name for t in rd.tensors if t.name == 'output.weight'), None),
                           results=out, raw=rows))


@function(gpu="RTX5090", cpu=4, memory="32Gi", image=image, volumes=[weights], timeout=1800)
def bench_5090():
    return _bench("RTX5090")


@function(gpu="A10G", cpu=4, memory="32Gi", image=image, volumes=[weights], timeout=1800)
def bench_A10G():
    return _bench("A10G")


@function(gpu="RTX4090", cpu=4, memory="32Gi", image=image, volumes=[weights], timeout=1800)
def bench_4090():
    return _bench("RTX4090")


if __name__ == "__main__":
    import os
    which = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if which == "gguf":
        res = json.loads(gguf.remote())
        os.makedirs("results", exist_ok=True)
        json.dump(res, open("results/gguf.json", "w"), indent=1)
        print(json.dumps(res, indent=1))
    else:
        res = {}
        for name, fn in (("A10G", bench_A10G), ("RTX4090", bench_4090), ("RTX5090", bench_5090)):
            try:
                res[name] = json.loads(fn.remote())
            except Exception as e:  # one GPU type being unavailable shouldn't sink the rest
                res[name] = dict(ok=False, err=repr(e))
            print(name, json.dumps(res[name], indent=1), flush=True)
        os.makedirs("results", exist_ok=True)
        json.dump(res, open("results/bench.json", "w"), indent=1)
