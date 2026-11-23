# Vector units and everything else

Most operations in a transformer run inside matmul. But the ones that do not can still decide your speed. Vector units are the sixth of Jouppi's six key decisions. [J]

## What runs here

| Operation | Why the array cannot do it | Hardware it wants |
| --- | --- | --- |
| Softmax | Exponentials, row max, row sum | Exp approximation, reductions |
| LayerNorm, RMSNorm | Mean, variance, reciprocal square root | Reduction, 1/√x |
| GELU, SiLU | Non-linear functions | Polynomial or lookup |
| RoPE | Pairwise rotation | Elementwise multiply-add |
| Residual add | Elementwise | Adder, ideally fused |
| Quantize, dequantize | Scale, round, clip | Scale multiplier, saturation |
| Top-k, sampling | Sorting, randomness | Selection network |
| Optimizer step (training) | Per-weight math | Heavy vector throughput |

Your PLAN.md lists exactly these: requant, LayerNorm, softmax, GELU, residual add. The encyclopedia has a block for each. [E]

## Sizing the vector unit

Use Amdahl's law with overlap. Suppose a tile takes T_array cycles on the array, and its follow-on vector work takes T_vector cycles.

- If they run in series, total = T_array + T_vector.
- If software pipelined, total = max(T_array, T_vector).

You want T_vector ≤ T_array, so vector work hides completely. Vector throughput needed then follows from how many vector operations each array output needs.

**Worked example.** A 256×256 array in steady state produces 256 outputs per cycle. If each output needs 4 vector operations for scale, bias, activation and quantization, you need 1,024 lane-operations per cycle. [D] Ironwood's move to four general ALUs per lane, and vector registers going from 8×128 to 16×256, is this calculation made real. [J]

Attention is harder: for each score element you need a subtract, an exp, a sum and a multiply, and K is small so the array finishes quickly. This is where vector units become the critical path. FlashAttention-style tiling keeps that work on chip, and the exp unit often sets the limit. [F]

## TPU's vector unit

TPU v2 has 128 lanes, each with an 8-way sublane dimension, so one operation covers 8 × 128 elements. Each lane reads and writes its own local slice of VMEM. [J] That locality matters: no crossbar is needed for the common case.

Two things evolved. ALUs grew from two restricted units per lane, each able to do some operations, to four general-purpose ones. Registers doubled in both dimensions. [J]

Note the design pattern. Restricted ALUs are cheap but constrain the compiler. As logic got cheaper, generality won.

## Nonlinear function strategy

You rarely build exact hardware. Three options [E]:

- **Lookup table**: fast and simple. Size grows with accuracy.
- **Polynomial with multiply-add**: reuses vector ALUs. Needs several cycles.
- **Interpolated table**: a small table plus one multiply-add. A good compromise.

For softmax, subtract the row max before exponentiating, for numerical range. Exponentials in base 2 reduce to a shift plus a small polynomial on the fractional part. A reciprocal unit lets you multiply rather than divide.

For fixed-point designs like yours, define the approximation identically in Python and RTL, since numerical matching is a stated risk in your plan. *Nonlinear units and MoE* works all of this through for your chip, with measured table sizes and errors.

## Transpose and permute

Training needs high-throughput transposes in the backward pass. You could do them on the array, but a transpose unit costs far less power and area. [R] Inference can usually avoid them. Compute Q·Kᵀ directly with an array mode that takes the second operand transposed, as TPUs do, and transpose weights offline. [R] Roune recommends keeping a permute unit anyway, for head layout. He says he is less sure on transpose, and notes that you will sleep better with it.

## SparseCore

SparseCore is a small dataflow engine for embeddings, taking about 5% of die area and 5% of power. Each unit has 16 tiles, each with an HBM channel, a Fetch Unit, an 8-wide SIMD unit, a Flush Unit, and a slice of 2.5 MiB Sparse Vector Memory. [J]

Its name misleads: it has nothing to do with structured sparsity. It is "sparse" in the sense of accessing HBM at finer granularity. [R] Watch how its role changed over time. Built for recommendation embeddings, it became an offload for all-reduce, all-gather, reduce-scatter, broadcast, Top-K and small sparse decode work. [J] Roune suggests a plain CPU core could do the same job more generally. [R]

The general lesson: a small second engine for irregular work is cheap insurance against a shifting workload.

## Silent data corruption

Vector units are now also a reliability feature. Ironwood's hardware replay unit randomly samples vector bundles and reruns odd-lane operations on even lanes in idle VLIW slots. It adds no performance overhead, it is invisible to the compiler, and it has caught defective chips that passed every other screen. [J]

## Tuning reference

| Knob | Effect | Cost |
| --- | --- | --- |
| Lanes | Throughput on elementwise work | Area, register file |
| ALUs per lane | Operations per cycle | Area, VLIW slots |
| ALU generality | Compiler freedom | Area per ALU |
| Register tile shape | Fewer instructions per vector | Tiling constraints |
| Dedicated exp unit | Softmax throughput | Area for one function |
| Numerics | FP32 is easiest here | Modest, since this block is small |
| Second engine | Irregular work off the critical path | About 5% area in Google's case |
