# 11. How much memory: on the chip, on the card, and the 1M-token context

*Started 2026-11-10. Status: under consideration. The requirement to handle the full context length is the user's (D-011). The memory size is proposed (D-012). The numbers come from `long_context()` and `sram_budget()` in `calc/qwen35_budget.py`.*

"Memory" means two different things for us, and they're sized for different reasons:
- **On-chip SRAM** is a few MB, sized by how much data we need close at hand while we work.
- **Card DRAM** (the GDDR7 chips next to the die) is GBs, sized by how much the model and its context need to store.

## What "the entire context length" means for Qwen3.5-9B

From the model card and `config.json`:
- **Native context:** 262,144 tokens (`max_position_embeddings`).
- **Extended context:** "extensible up to 1,010,000 tokens" with YaRN, a way of stretching the position encoding.
- **The card warns** that static YaRN can hurt quality on short texts, and recommends setting the stretch factor to fit the typical length.

So 1M context is a mode you switch on, not the default. Supporting it needs no extra hardware for YaRN. It only changes the rotation angles used in RoPE, which come from a table or a few multiplies. **The real costs of 1M context are memory capacity, decode speed and prefill time.**

## What grows with context

Only the KV cache of the 8 full-attention layers grows. The DeltaNet state is a fixed 50.3 MB (fp32), however long the context gets. Per token of context, the KV cache holds:
- 8 layers × 4 KV heads × 256 values × 2 (K and V) = **16,384 values**

At 1,010,000 tokens, that's 16.5 billion values. The format decides everything:
- **bf16 KV:** 33.1 GB
- **int8 KV with a scale per 64 values:** 17.1 GB
- **int4 KV with a scale per 32:** 9.3 GB

Add the stored weights. That's **5.04 GB including the 0.57 GB embedding table.** Our earlier 5.03 GB figure for 32K context missed the embedding table and was slightly low. The total needed at 1M:
- bf16 KV: **38.2 GB**
- int8 KV: **22.2 GB**
- int4 KV: **14.4 GB**

## The three costs of 1M context

### 1. Capacity

Our design point has 8 GB. GDDR7 gives us these options on the same 128-bit bus at the same 512 GB/s:
- **4 × 2 GB:** 8 GB
- **4 × 3 GB:** 12 GB (3 GB chips exist)
- **8 × 2 GB in clamshell:** 16 GB. In clamshell mode, two chips share each 32-bit channel, each using 16 pins. That doubles capacity at the same bandwidth. The RTX 5060 Ti 16GB is built this way.
- **8 × 3 GB in clamshell:** 24 GB

So **24 GB with int8 KV holds the full 1M context**, with about 2 GB to spare. 16 GB holds it only with int4 KV. bf16 KV at 1M doesn't fit on any 128-bit GDDR7 board.

### 2. Decode speed at long context

Every decode step reads the whole KV cache once. At 512 GB/s and 92% utilization:
- **At 32K:** 92 tok/s with int8 KV (97 with int4)
- **At 128K:** 70 tok/s (82)
- **At 262K:** 52 tok/s (68)
- **At 1M:** **22 tok/s** with int8 KV, **34 tok/s** with int4 KV, 12.5 with bf16

Our floor is 30 tok/s. **At 1M, only int4 KV keeps us above it.** So int4 KV isn't just a capacity trick, it's needed for speed at full context.

### 3. Prefill time

Attention work grows with the square of the context. Reading a 1M-token document means each new token attends to everything before it. At our 8.2 TOPS prefill array, fully busy:
- **32K tokens:** 1.2 minutes
- **128K:** 6.5 minutes
- **262K:** 17.6 minutes
- **1M:** about **2.8 hours**

That's slow, but it's honest: an RTX 5060 can't hold 1M context at all in 8 GB. And a person typically builds a long context up over a conversation, not by pasting 1M tokens at once. **We should say this plainly in the blog** rather than claim "1M context" without the timing.

## Ways we considered to reach 1M, and what we think of each

- **Make every card 24 GB.**
  - What it does: unit cost goes from about $158 to about $350, and tokens per dollar at normal context falls from 0.321 to 0.145. That's below the RTX 5060's 0.273.
  - Verdict: **rejected as the default.** It makes the main score lose, to pay for a mode most people rarely use.
- **Keep the KV cache in host memory and stream it over PCIe.**
  - What it does: streaming 17 GB per token over PCIe Gen4 x4 (8 GB/s) takes 2.1 s per token.
  - Verdict: **rejected.**
- **Add a second, cheaper memory such as LPDDR5X just for the KV cache.**
  - What it does: adds a second PHY and controller, and the KV reads still need bandwidth.
  - Verdict: **rejected** for now. It's complexity for one mode.
- **Sparse attention** (read only the most relevant KV blocks, as in Quest or DeepSeek's NSA).
  - What it does: cuts bytes a lot.
  - Verdict: it changes the model's output, and our goal is to run the exact model. **Parked** as an optional mode that the public ISA can express. Users can try it, but we won't count it in our score.
- **The same chip, different boards.**
  - What it does: the chip supports up to 24 GB (clamshell mode and enough address bits) plus int8 and int4 KV formats. The base card ships with 8 GB for the score. A "long-context" card ships with 16 or 24 GB.
  - Verdict: **our leaning.** Capacity becomes a board decision, not a chip decision.

With that leaning, the scored base card at 8 GB holds about 170K tokens with int8 KV, or about 315K with int4. That covers the native 262K window with int4 KV. Working: (8 − 5.04 − 0.05) GB ÷ 16,384 values ÷ 1.03 bytes is about 172K tokens. Using 0.5625 bytes for int4 gives about 316K. Treat both as rough: buffers and firmware need some room too.

What the chip needs for this:
- **A GDDR7 controller that supports clamshell (x16) mode.** That needs a check with the PHY and controller vendor.
- **Address space for 24 GB,** plus a page table or simple base-and-bound for the KV cache so it can grow.
- **int8 and int4 KV formats** in the attention data path, with the scales.
- **The YaRN position table** loaded by firmware.

A side benefit: the embedding table (0.57 GB) only gives one 8 KB row per token, so it can stay in host memory and be sent over PCIe. That frees 0.57 GB on the card for about 34K more tokens of int8 KV.

## On-chip SRAM

The opposite question is how little we can get away with. At batch 1, on-chip memory holds only what's being worked on right now. A first guess at the budget (the tile sizes are our choices):
- **The weight stream FIFO:** 102 KB. By Little's law, 512 GB/s × 100 ns of DRAM latency is 51 KB in flight. We doubled it for headroom.
- **The DeltaNet state for one head, fp32, double-buffered:** 131 KB. Each of the 32 heads per layer is independent, so we stream the state head by head (64 KB each) instead of holding a whole layer (2.1 MB).
- **The lazy-commit log for speculative decoding:** 99 KB. See `12_speculative_decoding.md`.
- **Activations for a 64-token prefill tile:** 524 KB in, 1.57 MB of FFN intermediate, 1.05 MB of fp32 residual.
- **A KV tile:** 256 positions, double-buffered, 262 KB.
- **Instruction memory:** 256 KB (a guess).

The total is about **4 MB**, roughly 2 mm² at a rough 0.5 mm² per MB on an N6-class node. That density figure is an estimate to verify. It's small, which fits the "small silicon" lever.

**The rejected alternative:** keep the whole DeltaNet state on chip.
- It would take 50.3 MB of fp32, about 25 mm², and would save 2.3% of bytes per token.
- That's about 2 tok/s for about a third more die. **Rejected.** We record it because it's an obvious idea, and the earlier batch-1 section of the script shows it too.

The 4 MB budget gets checked against the performance model once it exists. The prefill tile size (64 tokens) is the biggest item and the easiest to change.

## Summary

- **On chip:** about 4 MB of SRAM.
- **On the card:** 8 GB for the scored base card. The chip supports 16 and 24 GB boards for long context.
- **1M context:** needs 24 GB with int8 KV, or 16 GB with int4 KV. Decode at 1M is 22 tok/s with int8 KV and 34 with int4, so int4 KV is the long-context mode. Prefilling a full 1M prompt takes hours, and we'll say so.
