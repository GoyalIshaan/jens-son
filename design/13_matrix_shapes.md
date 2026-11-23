# 13. The shapes of the matrices

*Started 2026-11-10. Status: facts from the config, plus a first idea of how to map them onto hardware (under consideration). The numbers come from `shapes()` in `calc/qwen35_budget.py`. The weights listed add up to the same 7.936 B read per token that every other note uses.*

## The weight matrices

Each entry gives the input size, output size, how many layers have it, the total weights, and the share of bytes read per token.

**Full-attention layers (8 of them):**
- **q plus output gate:** 4096 → 8192, 268 M, 3.4%. The output is twice the 16 × 256 query width because Qwen3.5's attention has an output gate (`attn_output_gate: true`).
- **k:** 4096 → 1024, 34 M, 0.4%. 4 KV heads × 256.
- **v:** 4096 → 1024, 34 M, 0.4%.
- **o:** 4096 → 4096, 134 M, 1.7%.

**Gated DeltaNet layers (24 of them):**
- **q, k, v and gate z, fused:** 4096 → 12288, 1,208 M, **15.2%**. That's q 2048, k 2048, v 4096 and z 4096.
- **beta and decay a:** 4096 → 64, 6 M, 0.1%. One of each per V head. A tiny, awkward shape.
- **out:** 4096 → 4096, 403 M, 5.1%.

**FFN, in all 32 layers (SwiGLU):**
- **gate:** 4096 → 12288, 1,611 M, **20.3%**
- **up:** 4096 → 12288, 1,611 M, **20.3%**
- **down:** 12288 → 4096, 1,611 M, **20.3%**

**Once per token:**
- **LM head:** 4096 → 248,320, 1,017 M, **12.8%**. The biggest single matrix. Its output is a vector of 248,320 scores, and we need the top one (or top few for sampling).

**Not weights, but still matrix maths, done per head:**
- **Attention:** with 4 query heads sharing each KV head (GQA 4:1), each KV head does q[4 × 256] · Kᵀ[256 × L] for scores, then scores[4 × L] · V[L × 256]. L is the context length. This is a tiny matrix times a long one.
- **DeltaNet:** each of 32 V heads per layer has a state S of 128 × 128. Per token it does a few matrix-vector products with S (S·k, S·q) and one rank-1 update (S += outer product of two 128-vectors). It's small and fixed in size whatever the context.

## What the shapes tell us

- **Only two input widths matter:** 4,096 (almost everything) and 12,288 (FFN down). So each output is a dot product of length 4,096 or 12,288. Fixed lengths make control simple.
- **Every size is a multiple of 64,** including the vocabulary: 248,320 = 512 × 485. Tiles of 64, 128 or 256 never leave ragged edges. The one exception is beta/a with only 64 outputs. It's 0.1% of the work, so it can run slowly without mattering.
- **The FFN is 61% of the bytes,** and DeltaNet's projections are another 20%. So "make the FFN path perfect" is the optimization that counts. Attention projections are under 6%.
- **Matrices that share an input can be fused.** Gate and up both read the same input vector, so treat them as one 4096 → 24576 matrix. The same goes for q/k/v. One long pass over a bigger matrix keeps DRAM streaming smoothly instead of stopping and starting.
- **At batch 1, every weight multiply is a matrix times a vector:** [1 × 4096] · [4096 × N]. Each weight is used exactly once per token. That's why a classic weight-stationary systolic array (load a weight tile, reuse it many times) is a poor fit for decode. It pays to load weights it then barely reuses.

## A first mapping onto hardware

The numbers:
- At peak, 512 GB/s and 1 GHz deliver 512 bytes per cycle. At 0.5625 bytes per int4 weight including scales, that's **910 weights per cycle.** At 92% utilization, it's 837.
- For decode, each weight needs one MAC. So we need about **1,024 weight lanes:** a multiplier for each weight arriving that cycle, with some headroom over 910.
- The input vector (4,096 values) sits in a small buffer and is **broadcast**. Each lane takes its weight from the DRAM stream and multiplies it by the matching activation. Adder trees then sum the lanes that belong to the same output row.

Now give each lane **4 multipliers instead of 1,** one for each of 4 token rows:
- **Decode:** uses 1 row.
- **Speculative verification** of up to 4 tokens: uses all 4, at the same weight bandwidth (`12_speculative_decoding.md`).
- **Prefill:** uses all 4.

That's 1,024 × 4 = **4,096 MACs**, which is exactly the size our prefill floor asked for (D-007, the "about 64 × 64" array). So the shape is **weights along one axis, tokens along the other:** 1,024 × 4, not 64 × 64.

A catch for prefill:
- With 4 token rows, each weight read is used 4 times. So prefill runs at 4× decode speed: about 4 × 102, roughly 410 tok/s. That's 2,048 tokens in about **5.0 s**, which misses our 4 s floor.
- **The fix:** hold each weight in its lane for 2 cycles and run 8 tokens past it. Then the array is fully busy and prefill is compute-bound. 2,048 tokens × 7.936 B MACs ÷ 4,096 MACs per cycle ÷ 1 GHz is about **3.97 s**.
- That just meets the floor, with no margin, and ignores the attention part. So either the floor gets loosened, the array gets about 25% bigger, or the clock goes up. **Open.**

This is a sketch. The real check is the performance model in `06_build_and_verify_plan.md`.

## The awkward bits

- **The LM head's 248,320 outputs.** The maths is ordinary, but picking the top-k from 248K scores needs a small streaming top-k unit: keep the best few as scores go by. Don't write all 248K out and sort them.
- **beta/a at 4096 → 64.** Too narrow to fill 1,024 lanes. Fuse it into the q/k/v/z pass, making one 4096 → 12352 matrix.
- **Attention over long context.** At 1M context, the KV reads (17 GB with int8) are 4× the weight reads, so the attention path must stream at full bandwidth too. That means it either shares the main lanes (K and V tiles treated like weights) or gets its own fast path.
