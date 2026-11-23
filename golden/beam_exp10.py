"""Experiment set 10: what the chunked DeltaNet changes inside the model, layer by layer (note 30, part 2).

    cd golden
    BEAM=~/.local/share/uv/tools/beam-client/bin/python
    $BEAM beam_exp10.py smoke
    $BEAM beam_exp10.py run

All on the chip's full numerics plus experiment set 8's chip-exact pieces. Reference: the sequential DeltaNet.
1. Probe (2,048 tokens each of prose and code): for every layer, the relative change of the mixer (DeltaNet or
   attention) output, the FFN output and the residual stream, and the share of int8 codes that change at the
   inputs of the mixer's and the FFN's matrices. Versions: the chip's own decode order of the sequential rule
   ("fp32_onepass", a reordering already known to be harmless, as a yardstick), chunked int8, chunked 3-pass split
   with fp32 scales (D-025 as proposed), and the no-new-hardware version (fp16 scales applied after the product).
2. KL of the no-new-hardware version, 8K prose/code and 32K prose, paired against sequential.
"""
import json
import sys
import time

from beam import TaskPolicy, function

from beam_app import ensure_model, environment, eval_texts, image, save, strict_fp32, weights
from beam_exp2 import calib_sets, compare_positions, run_positions
from beam_exp5 import kl_nll, paired, per_position, summary

MODEL_DIR = "/weights/Qwen3.5-9B"
ALL = dict(sfu=True, rope=True, softmax=True, dnq=True)
NONE = dict(sfu=False, rope=False, softmax=False, dnq=False)


def _run(smoke):
    import torch
    from transformers import AutoTokenizer
    import beam_exp2
    import qwen35_golden as G
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
    gold = G.Golden(MODEL_DIR)
    G.CHIP.update(NONE)
    ref = {k: per_position(gold, v) for k, v in texts.items()}
    nll_ref = {k: kl_nll(ref[k], ref[k], v)[1] for k, v in texts.items()}
    t0 = time.time()
    ref_long = run_positions(gold, long_ids, pos)
    out = dict(kind="exp10_dn_layers", env=environment(), smoke=smoke, n_tokens={k: len(v) for k, v in texts.items()},
               long=dict(n_tokens=len(long_ids), every=every, ref_secs=time.time() - t0), rows=[])
    attn = dict(attn_q=CHIP_ACT, attn_p=CHIP_ACT)

    def evaluate(cond, label, num, chip, base=None, do_long=False):
        G.CHIP.update(NONE)
        G.CHIP.update(chip)
        gold.num = num
        t0 = time.time()
        row = dict(cond=cond, label=label, numerics=num.describe(), chip=dict(G.CHIP), sets={})
        raw = {}
        for k, v in texts.items():
            kl, nll, top1 = kl_nll(ref[k], per_position(gold, v), v)
            s = summary(kl, nll, top1, nll_ref[k])
            if base is not None:
                s["kl_vs_base"] = paired(kl, base[k][0])
                s["nll_vs_base"] = paired(nll, base[k][1])
            row["sets"][k] = s
            raw[k] = (kl, nll)
        if do_long:
            row["long"] = compare_positions(ref_long, run_positions(gold, long_ids, pos), long_ids.cpu(), pos.cpu())
        row["secs"] = time.time() - t0
        print(cond, label, {k: round(x["kl_mean"], 6) for k, x in row["sets"].items()},
              "long", round(row["long"]["kl_mean"], 6) if do_long else "-", f"{row['secs']:.0f}s", flush=True)
        out["rows"].append(row)
        return raw

    # the chip's full numerics + experiment set 8's chip-exact pieces
    wkw = dict(weights={"default": "lut4_g16_fp8_t8", "embed": "int8_g32"})
    calib = [torch.tensor(c, device="cuda") for c in calib_sets(tok, n_seq=8 if smoke else 64,
                                                                 seq_len=128 if smoke else 512)]
    G.CHIP.update(NONE)
    gold.num = G.Numerics()
    t0 = time.time()
    quantize_model_gptq(gold, G.Numerics(**wkw), calib, log=lambda m: print(m, flush=True))
    out["gptq_secs"] = time.time() - t0
    chip = dict(**wkw, act=CHIP_ACT, kv="int8_head", **attn)
    G.CHIP.update(ALL)
    # 1. layer-by-layer probe
    n_probe = 256 if smoke else 2048
    out["probe"] = {}
    for name, ids in texts.items():
        ids = ids[:n_probe]
        gold.num = G.Numerics(**chip)
        gold.probe = dict(mode="record", ref={})
        gold.forward(ids, gold.new_cache(), return_hidden=True)
        pref = gold.probe["ref"]
        out["probe"][name] = {}
        for st in ("fp32_onepass", "chunked_int8", "chunked_split", "chunked_lite"):
            gold.num = G.Numerics(**chip, state=st)
            gold.probe = dict(mode="compare", ref=pref, out={})
            gold.forward(ids, gold.new_cache(), return_hidden=True)
            out["probe"][name][st] = gold.probe["out"]
            r = gold.probe["out"]
            print(name, st, "residual rel at last layer", round(r["residual"][-1]["rel"], 6),
                  "ffn_in code change (mean over layers)",
                  round(sum(x["code_change"] for x in r["ffn_in"]) / len(r["ffn_in"]), 6), flush=True)
        gold.probe = None
        del pref
    # 2. quality of the no-new-hardware version
    base = evaluate("chip", "chip numerics, sequential DeltaNet", G.Numerics(**chip), ALL, do_long=True)
    evaluate("chip", "chip numerics, chunked_lite", G.Numerics(**chip, state="chunked_lite"), ALL, base=base,
             do_long=True)
    out["total_secs"] = time.time() - t_all
    out["gpu_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    save("smoke10_dn_layers" if smoke else "exp10_dn_layers", out)
    return json.dumps(out)


@function(gpu=["RTX5090", "RTX4090"], cpu=4, memory="64Gi", image=image, volumes=[weights], timeout=14400,
          env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false"},
          task_policy=TaskPolicy(ttl=14400, timeout=14400))
def dn_layers(smoke: bool = False):
    return _run(smoke)


def _store(name, res):
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    res = json.loads(dn_layers.remote(which == "smoke"))
    _store("smoke10_dn_layers" if which == "smoke" else "exp10_dn_layers", res)
    for r in res["rows"]:
        print(r["cond"], r["label"], {k: round(s["kl_mean"], 6) for k, s in r["sets"].items()},
              "long", round(r["long"]["kl_mean"], 6) if "long" in r else "-")
