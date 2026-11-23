"""Progression step 2: on-chip SRAM budget, special-function counts, LM head. Run: python3 design/calc/budgets_v2.py

Built from the decided formats (D-005, D-019): int8 activations with a power-of-two scale per 32 (1 byte of
exponent per 32 values), int8 KV with one fp16 scale per 256-value head, fp32 DeltaNet state, fp32 residual.
Dataflow choices (note 27):
  - matrix tiling: for each tile of N_T outputs, stream the input in 1,024-wide chunks, so one chunk of
    activations stays in the lanes for N_T beats (SRAM reads ~1 KB/cycle) and the outputs sit in
    N_T x rows fp32 accumulators;
  - DeltaNet and attention projections are made head by head, so only one head's q/k/v/z lives on chip;
  - FFN gate and up rows are interleaved, so SiLU(gate)*up is made and rounded to int8 as it streams out;
  - the speculative lazy-commit log, the conv history and the draft head's KV ring buffer live in DRAM.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_budget as B  # noqa: E402

KB, MB = 1024, 1024 * 1024
H, FFN = B.HIDDEN, B.FFN
ACT = 1 + 1 / 32                       # int8 + 1-byte power-of-two exponent per 32
LANES = 1024


def sram(mode, R=64, C=64, N_T=64, residual_on_chip=True, k=3):
    """mode: "decode" (1 token), "verify" (k+1 tokens), "prefill" (chunks of C tokens). Returns [(item, bytes)]."""
    T = {"decode": 1, "verify": k + 1, "prefill": C}[mode]
    rows = min(T, R)
    items = [
        ("weight stream FIFO (Little's law: 100 ns x 512 GB/s, x2)", 2 * 512e9 * 100e-9),
        ("matrix-unit activation chunk, double-buffered (rows x 1,024 int8 + exps)", 2 * rows * LANES * ACT),
        (f"output accumulators ({N_T} outputs x {rows} rows fp32)", N_T * rows * 4),
        (f"residual stream ({T} tokens x 4,096 fp32)" if residual_on_chip else
         f"residual staging (kept in DRAM; {N_T} outputs x {T} tokens fp32, double-buffered)",
         T * H * 4 if residual_on_chip else 2 * N_T * T * 4),
        (f"normed layer input ({T} x 4,096 int8 + exps)", T * H * ACT),
        (f"FFN intermediate ({T} x 12,288 int8 + exps)", T * FFN * ACT),
        (f"one DeltaNet head's q, k, v, z, beta, g for {T} tokens (fp32)", T * (128 * 4 + 2) * 4 * 2),
        ("DeltaNet resident fp32 state (D-023)", (2 if mode == "prefill" else 4) * 128 * 128 * 4),
        ("conv history staging, one layer (3 x 8,192 fp32), double-buffered", 2 * 3 * 8192 * 4),
        (f"one query head's q + gate for {T} tokens (int8 q, fp32 gate)", T * 256 * (ACT + 4) * 2),
        ("KV block, 256 positions x 256 dims x K and V int8, double-buffered", 2 * 256 * 256 * 2),
        (f"attention scores tile ({min(T * 4, R)} query rows x 256 positions fp32)", min(T * 4, R) * 256 * 4),
        (f"attention output + softmax state ({min(T * 4, R)} rows x 256 fp32)", min(T * 4, R) * (256 + 2) * 4),
        ("lazy-commit log staging (one layer's k+1 tokens, on its way to DRAM)",
         (k + 1) * (16 * 128 + 32 * 128 + 64) * 4 if mode == "verify" else 0),
        ("LM head: running best value + index per row", rows * 8),
        ("instruction memory (to be sized by ISA v1)", 256 * KB),
    ]
    return items


def show(mode, **kw):
    items = sram(mode, **kw)
    tot = sum(b for _, b in items)
    print(f"\n== SRAM, {mode} {kw} ==")
    for nm, b in items:
        if b:
            print(f"  {nm:78s} {b / KB:8.1f} KB")
    print(f"  {'total':78s} {tot / MB:8.2f} MB")
    return tot


def special_function_counts():
    """Per token, by function: (count per token at context L) for the decode and prefill sizing."""
    full, lin = B.N_FULL, B.N_LIN
    def per_token(L):
        return dict(
            exp=full * B.Q_HEADS * L + lin * B.LIN_V_HEADS,                      # softmax + DeltaNet decay
            silu=B.LAYERS * FFN + lin * 8192 + lin * B.LIN_V_HEADS * B.LIN_DIM,  # FFN + conv + gated norm
            sigmoid=full * B.Q_HEADS * B.HEAD_DIM + lin * B.LIN_V_HEADS,         # attention gate + beta
            softplus=lin * B.LIN_V_HEADS,
            rsqrt=(2 * B.LAYERS + 1) + full * (B.Q_HEADS + B.KV_HEADS) + lin * (2 * B.LIN_V_HEADS + B.LIN_V_HEADS),
            recip=full * B.Q_HEADS,
        )
    print("\n== Special functions per token ==")
    for L in (4096, 32768, 131072, 1_010_000):
        c = per_token(L)
        print(f"  context {L:>9,d}: " + ", ".join(f"{k} {v:,}" for k, v in c.items()))
    # throughput needed (per 1 GHz cycle) if each kind must keep pace with the phase it belongs to
    print("  rate needed, decode at 1M context, k=3 verify (about 45 ms per 4.3-token step):")
    c = per_token(1_010_000)
    step = 4 * 1.0 / 68.5            # 4 tokens verified per step at ~68.5 tok/s (note 21, 64 rows)
    for kname in ("exp", "silu", "sigmoid"):
        print(f"    {kname:8s} {4 * c[kname] / (step * 1e9):8.2f} per cycle averaged over the step")
    print("  rate needed, prefill of 32K tokens in about 6.8 s (average context 16K):")
    c = per_token(16384)
    for kname in ("exp", "silu", "sigmoid"):
        print(f"    {kname:8s} {32768 * c[kname] / (6.8 * 1e9):8.2f} per cycle averaged over the prompt")


def lm_head():
    print("\n== LM head and next-token choice ==")
    w = B.VOCAB * H
    print(f"  main LM head: {B.VOCAB:,} x {H:,} = {w / 1e9:.3f} B weights, {w * 9 / 16 / 1e6:.0f} MB per pass "
          f"({100 * w / B.P['read_per_token']:.1f}% of the weights read per token)")
    print(f"  each logit takes {H // LANES} beats; the unit produces {LANES // H * 1.0 if H < LANES else 1 / (H // LANES):.2f} "
          f"logits per row per cycle -> greedy = one fp32 compare per logit, a running (best, index) per row")
    print("  ties: the lowest token id wins (bit-exact rule). No logits are stored.")
    print(f"  draft head: top 32,768 tokens, rows stored in that order; the winning row's position maps to a token id "
          f"through a 32,768 x 3-byte table in DRAM ({32768 * 3 / KB:.0f} KB; one 3-byte read per draft)")
    print("  verify: compare main greedy tokens at positions i..i+k with the k drafts; accepted = leading matches;")
    print("          then move the KV end pointer back and replay the accepted DeltaNet updates from the log (D-013)")


if __name__ == "__main__":
    show("decode")
    show("verify", k=3)
    for R, C in ((64, 64), (64, 128), (32, 64)):
        show("prefill", R=R, C=C)
    show("prefill", R=64, C=128, residual_on_chip=False)
    extra = 2 * 128 * H * 4 * B.LAYERS            # read + write the residual once per layer, per 128-token chunk
    w = B.P["read_per_token"] * 9 / 16
    print(f"  keeping the residual in DRAM adds {extra / 1e6:.0f} MB of traffic per 128-token chunk, "
          f"{100 * extra / w:.1f}% of the chunk's {w / 1e9:.2f} GB of weights")
    special_function_counts()
    lm_head()
