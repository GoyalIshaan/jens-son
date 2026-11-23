# Matmul shape

This chapter answers the question you asked directly: how do designers reason about matmul shape? It is the most important chapter in the guide.

::viz cube

## Naming the three dimensions

A matrix multiply has three dimensions. This guide uses standard GEMM names throughout.

`C[M × N] = A[M × K] · B[K × N]`

- **M**: rows of activations. Tokens times batch.
- **K**: the reduction depth. How many products sum into one output.
- **N**: output features. Columns of the weight matrix.

Roune uses N for rows and M for columns, so swap them when you read his document. His "N rows of LHS" is our M.

Picture the computation as a box of size M × N × K. Each small cube inside is one multiply-accumulate. An array of R × C processing elements covers a slab of that box each cycle. Everything about utilization comes from how well slabs tile the box. The lab above shows exactly this.

## The three dimensions are not equal

This is the insight that separates good reasoning from bad.

**M is usually fillable.** Rows do not interact. You can raise M with batching, with sequence length during prefill, or with speculative tokens during decode.

**N is usually fillable.** More output features or more experts or more heads give you more columns.

**K is not fillable.** If a dot product only has 64 terms, you cannot find more. Roune calls it the more fundamental limit. [R] When K is smaller than the array's reduction depth, those PEs have nothing to do.

## Mismatch math

Consider a width-W array running a problem of width w, with w < W.

| Situation | Utilization |
| --- | --- |
| Naive | (w/W)² |
| Block-diagonal, independent sub-blocks | w/W |

A 64-wide problem on a 128-wide array runs at 25%, or at 50% with block-diagonal packing. [R] The trick needs no hardware feature, since independent matmuls can occupy the upper-left and lower-right blocks of the array at the same time.

Rounding up also costs. A dimension just over a multiple is the worst case: 257 on a width of 256 uses two tiles, so it can run at half speed. [R] This is why you hear rules like "powers of 2 are right, otherwise your code's a blight". [R]

## Where shapes come from

| Operation | M | N | K | Notes |
| --- | --- | --- | --- | --- |
| Feed-forward, prefill | Tokens | 4 × d_model | d_model | All large, a systolic array's friend |
| Feed-forward, decode | Batch × speculated tokens | 4 × d_model | d_model | M is the problem |
| Attention scores QKᵀ, prefill | Tokens | Tokens | head_dim | K is small |
| Attention scores, decode | Queries per KV group | Context | head_dim | Both M and K small |
| Attention output PV, decode | Queries per KV group | head_dim | Context | K large, M small |
| First image layer | Pixels | Channels | 3 | K=3, do it on a CPU |
| Depthwise 3×3 convolution | Many | Many | 9 | Poor fit for arrays, use vector units |

Attention K is typically 16 to 128. Feed-forward K can be 8,192 or more. [R] A chip that serves both faces a 100× spread in its most important dimension.

## The rows-needed equation

For a weight-stationary array with R reduction rows, loading a new weight tile takes time. While it loads, the array must stream enough activation rows to stay busy. With double buffering:

`M_min = R ÷ ρ`

ρ is the bandwidth available for weights, relative to activations. [R] With equal bandwidth, a 128×128 array needs 128 rows. One row gets 1/128 = 0.8%.

Doubling weight bandwidth halves the required rows, but you must provision that bandwidth in SRAM and HBM too. Taking this to the limit gives a batch=1 chip that needs R times as much bandwidth per operation. At that point it is no longer really a systolic array, and it needs SRAM in place of HBM. [R] Groq is this kind of design. It makes a deliberate trade: a much higher price for much lower token latency.

So there are three separate questions for a decode chip:

1. How many rows can I afford to gather? This is limited by latency and KV cache.
2. How much weight bandwidth can I pay for? This is limited by cost.
3. What array width follows from those two?

## Non-square arrays

Square arrays are the easiest for software because all vectors have one width. But each dimension can be examined separately. [R]

**Raising K alone** is roughly two arrays plus a vector add, without the vector add. It helps feed-forward and not attention, and it reduces the maximum parallelism: model K of 16,384 with array K of 1,024 allows at most 16-way within-layer splitting.

**Raising N alone** is roughly two arrays sharing their activation input. It saves some on-chip bandwidth. Vectors become mismatched, which complicates software.

**Lowering the required M** by adding weight bandwidth reduces batch and latency without changing throughput. It therefore helps responsiveness, and does nothing for tokens per dollar.

A cleverly shaped array is a way of matching your hardware to the box you expect. It also fixes your assumptions into silicon, for the two to three years it takes to design, build and deploy a chip. [J]

## Working through a real shape

Say you serve a model with d_model = 8192 on a 256×256 weight-stationary array.

**Feed-forward up-projection in decode.** M = 64 users × 4 tokens = 256, N = 32768, K = 8192.

- K tiles: 32. N tiles: 128. Each tile sees 256 rows, which matches the 256 needed.
- Result: full utilization in steady state. This is why Roune speaks of batch 256 or batch 64 with speculation for a 256-wide array. [R]

**Attention scores in decode, head_dim 128, 8 KV groups, 128 heads.** Queries per group = 128 ÷ 8 × 4 speculated = 64. So M = 64, K = 128.

- K of 128 on a 256-deep array: half the rows are idle unless you pack block-diagonally.
- M of 64 against 256 needed: one quarter.
- Result: around 12.5% to 25%, which may be fine because attention is limited by memory anyway. [D]

Now you can see Roune's argument for smaller attention cores: width 64 would run this same work near full utilization.

## Questions to ask of any shape

1. What are the smallest K and M I must run well?
2. Which dimension am I relying on batching to fill, and what does that cost in latency and KV cache?
3. What happens at non-multiples of the array width?
4. Can independent sub-problems share the array block-diagonally?
5. Is this operation compute-bound anyway? If it is memory-bound, low array utilization may cost nothing.

That last question matters. Utilization is not a virtue in itself. An array that sits idle while waiting on a saturated memory system is simply the correct outcome of a memory-bound workload.

## Test yourself

Use the lab above to answer these. Set the array to 128×128.

- With M=1, N=4096, K=4096, what limits you, and why does a 16×16 array do better on utilization and worse on time?
- With K=64, how much does raising array size hurt, and what do you lose by shrinking it instead?
- With M=257, watch what padding does, and compare with M=256.
