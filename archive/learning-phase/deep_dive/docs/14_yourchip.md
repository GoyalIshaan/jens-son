# Your chip

This chapter applies everything to your repo: `pe.sv`, `sys_array.sv`, `tile_engine.sv` and the embedded GPT-2 in `PLAN.md`. These are observations and suggestions. I have not changed your RTL, and I have not run synthesis. Every cycle count below comes from the analytical model, which matches `TOTAL_CYCLES` in your tile engine but is not a simulation of it.

## What you have built, in this guide's terms

| Attribute | Your design |
| --- | --- |
| Dataflow | Output-stationary |
| PE | int8 × int8, 32-bit accumulator, registers for A and B |
| Array | Parametrized R × C, default 2×2 |
| Tile cost | K + R + C − 2 cycles, plus handshake |
| Memory | Tile registers in `tile_engine` |
| Control | Three-state FSM |
| Output | Full R × C × 32-bit parallel bus |

## What you got right

**Parametrization.** Roune calls this a brilliant move he has not seen used anywhere: define everything by width, so you test at 2×2 and then scale. [R] `ROWS`, `COLS`, `INNER_K`, `DATA_W` and `ACC_W` already do this. Keep it up everywhere, including in your tests.

**Output-stationary for a small design.** Narrow moving data, simple dataflow, no weight-load path.

**Wide accumulator with sign extension.** `ACC_W=32` is safe to K = 65,536.

**A cycle model you can check.** Your `TOTAL_CYCLES` comment matches the theory exactly.

## The shape of your workload

The plan: 2 layers, d_model 64, 2 heads of width 32, FFN 256, vocab 256, context up to 32.

| Operation | M (decode) | M (prefill) | N | K |
| --- | --- | --- | --- | --- |
| QKV projection | 1 | 32 | 192 | 64 |
| Attention scores, per head | 1 | 32 | context | 32 |
| Attention values, per head | 1 | 32 | 32 | context |
| Output projection | 1 | 32 | 64 | 64 |
| FFN up | 1 | 32 | 256 | 64 |
| FFN down | 1 | 32 | 64 | 256 |
| LM head | 1 | 32 | 256 | 64 |

Totals [D]: about 123,000 MACs per decoded token at context 32, and 114,688 bytes of weights plus embeddings. All of it fits on chip. So you are building a tiny version of a Groq-style machine: no DRAM, weights in on-chip memory, fully predictable. Your hardcoded FSM is literally static scheduling.

## What array size should you pick?

The model's estimate of a full forward pass, with tiles running back to back: [D]

| Array | Decode cycles/token | Decode util | Prefill (32 tokens) cycles | Prefill util |
| --- | --- | --- | --- | --- |
| 2×2 | 63,104 | 48.7% | 1,009,664 | 97.4% |
| 4×4 | 33,216 | 23.1% | 265,728 | 92.5% |
| 8×8 | 18,272 | 10.5% | 73,088 | 84.1% |
| 16×16 | 10,800 | 4.4% | 21,600 | 71.1% |
| 32×32 | 7,064 | 1.7% | 7,064 | 54.4% |

This table holds the main lesson.

**Decode at M=1 wastes all but one row.** From 8×8 to 16×16 you buy 4× the PEs and get 1.7× the decode speed. You are paying for rows with nothing to do.

**Prefill behaves well.** Larger arrays help a lot, because M=32 fills them.

**Fill and drain dominate at small K.** With K=64 on 32×32, the 62-cycle fill nearly doubles every tile. Overlapping tiles takes the 8×8 prefill from 84% to 99.6%.

So there is a real architectural choice here, and your plan says to test it. Report tokens per second, and report PE-count-normalized numbers too.

## Fixing decode's M=1 problem

You have three options, and each teaches a concept from this guide.

**Option 1: Split K across rows.** Let each row of the array compute a partial dot product over a different K slice of the same output, then add those at the edge with a small reduction tree. This turns GEMV into full-array work. Your weights are on chip, so the bandwidth is cheap.

**Option 2: Add a dedicated GEMV path.** A row of MACs fed directly from wide SRAM, with a reduction tree. Roune's heterogeneous argument, at miniature scale.

**Option 3: Batch or speculate.** If you decode 4 streams at once, M=4 fills a 4×4 array. This changes your product, so it is a decision for you.

## Things that will not scale past small arrays

None of these are bugs at 2×2. They become problems as `ROWS` and `COLS` grow.

**1. The parallel output bus.** `ACC` exposes R × C × 32 bits at once. At 16×16 that is 8,192 wires into whatever consumes them, which is the crossbar wiring problem from the fabric chapter. A real array shifts results out along columns, over R cycles. If you double-buffer the accumulator, that drain can overlap the next tile. This is probably your largest structural change.

**2. Tile registers.** `A_reg` and `B_reg` store the whole tile in flops. Replace them with banked SRAM feeding row and column FIFOs. For synthesis, flops cost several times what SRAM bits cost.

**3. The skew mux.** Each row selects `A_reg[row][k_idx]` through a mux that grows with `INNER_K`, driven by a shared counter. The usual structure is a triangular shift-register skew at the array edge, which needs no mux at all.

**4. Global control.** `enable` and `clear_acc` reach every PE in one cycle. At 16×16 that is a high-fanout net; at 256×256 it cannot meet timing. Pipeline these alongside the data, so that each PE gets its control when the wavefront arrives. A `clear` bit traveling with the first operand also removes the need to wait for a drain.

**5. No tile overlap.** The FSM runs IDLE → RUN → DONE → IDLE, so each tile pays fill, drain and handshake. To overlap, you need the next tile's skewed operands to follow immediately, with a double-buffered accumulator.

**6. Accumulator width.** Your largest K is 256, so 24 bits is enough, saving 8 flops of 48 per PE. Keep it as a parameter, and test it. That also shows you what saturation logic costs.

## The surrounding blocks you will need

From your plan, and in this guide's terms:

| Block | Chapter | Note |
| --- | --- | --- |
| `matmul_engine` | Shape | Loop order decides what stays stationary. Keep accumulators across K tiles with `init_acc=0`, which you already support |
| Weight SRAM | Memory | 115 KB. Bank width should match array edge per cycle |
| KV cache | Memory | 2 layers × 2 × 32 × 64 = 8 KB at 1 byte per element |
| Requant | Reduction | Scale, shift, round, clip at the array edge, so 32-bit values never get stored |
| Softmax | Vector | Over at most 32 values. Use a small exp table, a reciprocal, and subtract the max first |
| LayerNorm | Vector | Needs reciprocal square root. Consider a small table plus a Newton step |
| GELU | Vector | A lookup table on int8 input has only 256 entries, which is exact and cheap |
| Controller | Control | Your static schedule is a strength. Add performance counters from day one |

*Nonlinear units and MoE* designs these blocks in detail, and covers adding experts.

The GELU row is worth noticing. With int8 inputs, a 256-entry table is the exact function, and requires no approximation work at all. This trick does not exist at FP32.

## Suggested experiments

Each answers a question this guide raises, with data you can produce in Verilator.

1. Sweep R, C from 2 to 32. Reproduce the table above from RTL, and see how close the model is.
2. Measure tile handshake overhead by comparing real cycles with K + R + C − 2.
3. Add column-shift output and compare wiring and cycles.
4. Add overlap, and check that you reach about 99% on 8×8 prefill.
5. Try K-split for decode, and compare against the 10.5% baseline.
6. Synthesize a PE with Yosys at 8, 16, 24 and 32-bit accumulators, and see where area goes.
7. Sweep group size G (two or four multipliers per PE), and measure area per MAC.
8. Build 1×64 with `x` broadcast and 1×16 with four multipliers per PE, and check the 2,048 and 1,920 beats per token from *Systolic arrays*.

Experiment 6 would give you real numbers for your target library. That would replace the [O] estimates in this guide with facts about your own chip.
