# Reduction trees

A dot product is multiplies followed by a sum. How you lay out that sum is one of the deepest structural choices in a matrix engine. Most of this chapter is **[O]** and **[D]**, because the PDFs define these blocks without analyzing them. The reasoning here connects those definitions to array design.

::viz reduction

## Two ways to add N numbers

**A chain** adds one number at a time. It needs N−1 adders, depth N−1, and only local wires.

**A tree** adds pairs, then pairs of pairs. It also needs N−1 adders, but depth is log₂N. It needs wires that get longer at each level.

| Inputs | Chain depth | Tree depth |
| --- | --- | --- |
| 4 | 3 | 2 |
| 16 | 15 | 4 |
| 256 | 255 | 8 |
| 4096 | 4095 | 12 |

The adder count is identical. You are choosing between latency and wire complexity.

## An array column is a chain in space

In a weight-stationary array, a column of R PEs is a pipelined adder chain. Each PE adds its product to the partial sum coming from above and passes it down. The depth of R adders does not hurt throughput, because each stage works on a different row. It adds R cycles of latency, and that is your fill time.

So a systolic array chooses the chain, and trades latency for short wires. With pipelining, the throughput is the same as a tree, and every wire is one PE long.

A tensor core does the opposite inside. It takes a group of products and sums them through a small tree in one step. Roune treats both as systolic arrays of different sizes, because at the chip level they play the same role. [R] At the circuit level they are different structures, and understanding the difference tells you what each is good at.

## The grouped processing element

Here is the design knob that bridges them. Rather than one multiply per PE, put G multipliers in each PE and sum them with a small tree.

| G | MACs per PE | Pipeline flops per MAC (int8, OS) | Tree depth | Clock pressure |
| --- | --- | --- | --- | --- |
| 1 | 1 | 48 | 0 | Lowest |
| 2 | 2 | 32 | 1 | Low |
| 4 | 4 | 24 | 2 | Medium |
| 8 | 8 | 20 | 3 | High |

The arithmetic: flops per MAC = 16 + 32/G, because each multiplier needs 16 operand-register bits and the whole group shares one 32-bit accumulator. [D] Fewer flops per MAC means less area, less clock power and less wire traffic. In exchange, the critical path gets longer, and a wider operand bus needs to reach each PE.

Real designs land between the extremes, because a pure chain is register-heavy and a large tree is wire-heavy.

## Carry-save: why trees are cheaper than they look

Adding two numbers fully requires a carry to ripple across all the bits. Adding many numbers does not need that at each step. A carry-save adder takes three numbers and outputs two, with no carry propagation, in the delay of one full adder. [E]

So a multi-operand sum runs like this:

1. Compress rows with carry-save adders, 3 to 2, again and again. Wallace and Dadda trees are two ways to arrange this. [E]
2. Do one real carry-propagate addition at the end, using a fast parallel-prefix adder like Kogge-Stone or Brent-Kung. [E]

Inside a multiplier, the same structure reduces partial products. Booth recoding halves how many partial products you start with. [E]

This has a useful implication. In a grouped PE, you can feed all G products into one compressor tree and only pay for a single carry-propagate add. Better still, keep the accumulator in carry-save form across cycles, and resolve it once at readout. That shortens the critical path of a MAC considerably. The cost is roughly twice the accumulator flops.

## Accumulator width

Integer accumulation has an exact bit-growth rule.

`accumulator bits = a_bits + b_bits + ⌈log₂ K⌉`

| Operands | K | Bits needed [D] |
| --- | --- | --- |
| int8 × int8 | 64 | 22 |
| int8 × int8 | 256 | 24 |
| int8 × int8 | 4096 | 28 |
| int8 × int8 | 65536 | 32 |

Your `ACC_W = 32` handles K up to 65,536 with no overflow. For your tiny GPT-2, the largest K is 256, so 24 bits would do. That cuts 8 flops from every PE, or about 17% of the PE's flops. Roune adds that sums can be made to saturate, which allows narrower accumulators with a defined failure mode. [R]

Accumulator width is a major hidden cost at low precision. As multipliers get smaller, the accumulator becomes a larger share of the PE. At FP4, the sum can cost more than the product. [R]

## Floating-point accumulation

Floating-point addition needs an exponent compare, a mantissa align, an add, a normalize and a round. [E] A tree of full FP adders is expensive. So dot-product hardware usually does this instead:

1. Find the largest exponent among the G products.
2. Shift all mantissas to align with it.
3. Add them as integers in a tree.
4. Normalize once at the end.

This is much cheaper, but it changes rounding, and results depend on the group size. Reproducibility and numerical behavior therefore become hardware design decisions. The TPU requirement of strict deterministic repeatability shows up at this level. [J]

## Reduction happens at every level

The same structure repeats at every scale, and the same chain-versus-tree trade applies.

| Level | Structure | Example |
| --- | --- | --- |
| Inside a multiplier | Compressor tree | Partial products |
| Inside a PE | Small adder tree | Grouped dot product |
| Across a column | Pipelined chain | Weight-stationary partial sums |
| Across K tiles | Accumulator memory | Summing tile results |
| Across lanes | Log-depth shuffle tree | Softmax max and sum, LayerNorm mean |
| Across chips | Ring or tree collective | All-reduce |

## K-splitting: using trees to fix a shape problem

When M is tiny, like decode at batch 1, a normal array leaves most rows idle. There is another option. Split K across the array, so that each row or PE group computes a partial dot product of the same output, then combine them with a reduction tree at the edge.

This turns a GEMV into full-width work. It requires bandwidth to supply K-sliced weights to every PE, since no weight gets reused across rows. That is the design point of low-batch chips: wide memory plus reduction trees, rather than an array with deep reuse.

For your embedded GPT-2, this matters, as the *Your chip* chapter shows.

## Vector reductions

Softmax needs a max and a sum across a row. LayerNorm needs a mean and a variance. These run on lanes, so you need cross-lane reduction. Options:

- A log-depth shuffle network across lanes. Fast, needs shuffle wiring.
- A dedicated reduction unit at the array edge. TPUs have units for row reductions, transposes and permutes beside the array. [J]
- Reusing the matrix unit by multiplying by a ones vector. Works, but wasteful.

## Network reductions

All-reduce is the top-level reduction. Precision matters here too, because partial sums cross the network. Eight bits is probably too few, 32 is more than you need, and 16 is probably fine with care. Choosing 32-bit sums doubles network bandwidth, and thus network cost. [R]

Ironwood's SparseCores now offload collectives such as all-reduce, plus Top-K. [J] Your factbook notes TPU 8i adds a dedicated Collectives Acceleration Engine, which shows that reductions keep earning dedicated hardware. [F]

## Tuning reference

| Knob | Effect | Cost |
| --- | --- | --- |
| Group size G | Fewer flops per MAC | Longer critical path, wider buses |
| Carry-save accumulators | Shorter MAC path | About 2× accumulator flops, resolve step |
| Accumulator width | Fewer flops | Risk of overflow, needs saturation |
| Saturating sums | Narrower safe datapath | Extra logic, changed numerics |
| Reduced FP alignment | Cheaper FP dot product | Rounding differs |
| Edge reduction tree | Decode efficiency | Area, weight bandwidth |
