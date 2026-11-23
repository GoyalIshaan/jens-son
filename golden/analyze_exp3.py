"""Summarise experiment set 3 (run locally, no torch needed):  python3 analyze_exp3.py

Tasks: for each quantized model vs bf16, a *paired* comparison on the same problems:
  - lost: bf16 solved it, quantized did not; gained: the reverse
  - delta = (gained - lost) / n, with the paired standard error
  - exact two-sided sign-test p-value on the discordant pairs (McNemar)
  - how often the generated text was byte-identical to bf16's
"""
import glob
import json
import math
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sign_test_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def tables():
    print("== table entry precision (GPTQ lut4_g16_fp8 + embed int8)")
    files = sorted(glob.glob(f"{R}/exp3_gptq_*.json")) + [f"{R}/exp2_gptq_lut4_g16_fp8,_embed_int8.json"]
    for f in files:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        c = d["config"]
        s = c["sets"]
        full = c["with_act_int8_kv_int8"]["sets"]
        print(f"- {c['name']}  [{d['env']['gpu']}]")
        for k in ("python_code", "wikitext"):
            x = s[k]
            print(f"    {k:11s} KL {x['kl_mean']:.4f} ±{x['kl_stderr']:.4f}  top1 {x['top1_agree']:.4f}  "
                  f"top5 {x['top5_overlap']:.4f}  ppl {x['ppl_change_pct']:+.2f}%  "
                  f"nll {x['nll_diff_mean']:+.4f}±{x['nll_diff_stderr']:.4f}  p99 {x['kl_p99']:.3f}  "
                  f"| +act8/kv8 KL {full[k]['kl_mean']:.4f} ppl {full[k]['ppl_change_pct']:+.2f}%")
        if c.get("table_stats"):
            print("    table stats:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c["table_stats"].items()})
        print(f"    predicted {c['predicted']['card_tok_s']:.1f} tok/s, weights {c['predicted']['weight_GB']:.3f} GB")


def tasks():
    print("\n== task benchmarks")
    runs = {}
    for f in glob.glob(f"{R}/exp3_tasks_*.json"):
        d = json.load(open(f))
        runs[d["model"]] = d
    for m, d in sorted(runs.items()):
        print(f"- {m}: {d['label']}  [{d['env']['gpu']}, {d['total_secs'] / 60:.0f} min]")
        for k, t in d["tasks"].items():
            print(f"    {k:9s} {t['correct']:4d}/{t['n']:<4d} = {100 * t['accuracy']:.2f}% ±{100 * t['stderr']:.2f}  "
                  f"({t['gen_secs'] / 60:.1f} min, {t['generated_tokens']} tokens)")
        if "quant" in d:
            q = d["quant"]
            print("    quant:", {k: v for k, v in q.items() if k not in ("weight_rel_rmse", "numerics")})
    if "bf16" not in runs:
        return
    ref = runs["bf16"]
    for m, d in sorted(runs.items()):
        if m == "bf16":
            continue
        print(f"\n  paired vs bf16: {m}")
        for k, t in d["tasks"].items():
            r = ref["tasks"][k]
            a, b = r["per_problem"], t["per_problem"]
            n = len(a)
            lost = sum(x and not y for x, y in zip(a, b))
            gained = sum(y and not x for x, y in zip(a, b))
            diff = [int(y) - int(x) for x, y in zip(a, b)]
            mu = sum(diff) / n
            se = math.sqrt(sum((v - mu) ** 2 for v in diff) / (n - 1) / n)
            same = sum(x == y for x, y in zip(r["outputs"], t["outputs"])) / n
            print(f"    {k:9s} delta {100 * mu:+.2f} pp ±{100 * se:.2f}  (lost {lost}, gained {gained}, "
                  f"sign-test p={sign_test_p(lost, gained):.3f})  identical text {100 * same:.1f}%")


if __name__ == "__main__":
    tables()
    tasks()
