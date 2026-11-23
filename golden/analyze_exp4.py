"""Summarise experiment set 4 (run locally, no torch needed):  python3 analyze_exp4.py

- MTP acceptance per set and draft vocabulary, bf16 vs chip format
- KV formats at 128K, KL by position bucket
- one-pass DeltaNet order at 32K
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SETS = ("chat", "gsm8k", "humaneval", "mbpp")
VOCABS = ("full", "top64k", "top32k", "top16k")


def load(n):
    p = f"{R}/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def mtp():
    runs = {c: load(f"exp4_mtp_{c}") for c in ("bf16", "chip")}
    for c, d in runs.items():
        if d is None:
            continue
        print(f"== MTP acceptance: {c}  [{d['env']['gpu']}]  numerics {d['numerics']}")
        if c == "bf16":
            print(f"  wiring variants (draft-1 hit, {d['variants_n_positions']} positions): "
                  + ", ".join(f"{k} {v:.4f}" for k, v in d["variants"].items()) + f" -> {d['chosen_variant']}")
        if d["mtp"].get("weight_rel_rmse"):
            e = d["mtp"]["weight_rel_rmse"]
            print(f"  MTP weight rel RMSE: mean {sum(e.values()) / len(e):.4f}, max {max(e.values()):.4f}")
        for s in SETS:
            x = d["sets"][s]
            print(f"  {s:9s} {x['n_seq']} seqs, {x['n_valid']}/{x['n_region']} positions valid")
            for v in VOCABS:
                y = x[v]
                pc = ", ".join(f"{p:.3f}" for p in y["p_cond"])
                tps = ", ".join(f"k={k} {t:.3f}" for k, t in y["tokens_per_step"].items())
                tiv = f"  target in vocab {y['target_in_vocab']:.3f}" if "target_in_vocab" in y else ""
                print(f"    {v:7s} draft1 {y['p_draft1']:.3f}  p_cond [{pc}]  tokens/step {tps} "
                      f"±{y['tokens_per_step_k3_stderr']:.3f}{tiv}")
    if all(runs.values()):
        print("== chip minus bf16, tokens per step at k=3")
        for s in SETS:
            print(f"  {s:9s} " + "  ".join(
                f"{v} {runs['chip']['sets'][s][v]['tokens_per_step']['3'] - runs['bf16']['sets'][s][v]['tokens_per_step']['3']:+.3f}"
                for v in VOCABS))
        print("== pooled over the four sets (equal weight), chip format")
        for v in VOCABS:
            pc = [sum(runs["chip"]["sets"][s][v]["p_cond"][j] for s in SETS) / 4 for j in range(3)]
            t3 = sum(runs["chip"]["sets"][s][v]["tokens_per_step"]["3"] for s in SETS) / 4
            print(f"  {v:7s} p_cond [{', '.join(f'{p:.3f}' for p in pc)}]  tokens/step k=3 {t3:.3f}")


def long128k():
    d = load("exp4_long128k")
    if d is None:
        return
    print(f"\n== KV at 128K  [{d['env']['gpu']}]  tokens {d.get('n_tokens')}, scored {d.get('scored_positions')}, "
          f"reference {d.get('ref_secs', 0):.0f}s")
    for c in d["configs"]:
        r = c["result"]
        print(f"  {c['name']}: KL {r['kl_mean']:.4f} ±{r['kl_stderr']:.4f}  top1 {r['top1_agree']:.4f}  "
              f"ppl {r['ppl_change_pct']:+.2f}%  nll {r['nll_diff_mean']:+.4f}±{r['nll_diff_stderr']:.4f}")
        if "by_position" in r:
            print("     KL by position:", r["by_position"])


def onepass():
    d = load("exp4_onepass32k")
    if d is None:
        return
    print(f"\n== one-pass DeltaNet order at 32K  [{d['env']['gpu']}]  scored {d['scored_positions']}")
    for c in d["configs"]:
        r = c["result"]
        print(f"  {c['name']}: KL {r['kl_mean']:.2e}  top1 {r['top1_agree']:.4f}  max |dlogprob| "
              f"{c['logprob_max_abs_diff']:.5f}  argmax identical {c['top1_identical']:.4f}  "
              f"ppl {r['ppl_change_pct']:+.5f}%")


def window():
    d = load("exp4_mtp_window")
    if d is None:
        return
    print(f"\n== MTP draft head with a limited attention window  [{d['env']['gpu']}]  {d['n_tokens']} tokens per sequence, "
          f"positions >= {d['start']}, every {d['stride']}th valid one, {d['vocab']} draft vocab")
    print(f"  checks: drafts_at == drafts {d['check_drafts_at_equals_drafts']:.4f}, "
          f"window past the end changes nothing {d['check_window_beyond_length_equals']:.4f}")
    for name, x in list(d["sets"].items()) + [("all", d["all"])]:
        head = "" if name == "all" else (f" ({x['n_scored']} scored of {x['n_valid']} valid / {x['n_region']}; "
                                         f"main model matches the text {x['main_top1_match']:.3f})")
        print(f"  {name}{head}")
        for w in d["windows"]:
            y = x[str(w)]
            print(f"    window {str(w):>6s}: p_cond [{', '.join(f'{p:.3f}' for p in y['p_cond'])}]  "
                  f"tokens/step k=3 {y['tokens_per_step']['3']:.3f}  vs full {y['delta_k3_vs_full']:+.3f} "
                  f"±{y['delta_k3_stderr']:.3f}  identical {y['identical_to_full']:.3f}")


if __name__ == "__main__":
    mtp()
    window()
    long128k()
    onepass()
