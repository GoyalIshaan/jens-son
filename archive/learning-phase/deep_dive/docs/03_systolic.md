# Systolic arrays

"It's all systolic arrays." [R] This is the engine of the chip. This chapter covers how one works, how to size it, and every knob you can tune.

A note on sources. Roune states that his document is not an explanation of systolic arrays, and he skips skew on purpose. [R] So your PDFs give the reasons and the big decisions, while most of the beat-level mechanics here are **[O]**, drawn from the published designs listed at the end, or **[D]**, checked with a beat-by-beat simulator in `deep_dive/model/sim_array.js`.

::viz systolic

## Why matmul wins

A matrix multiply has more arithmetic than data. Multiplying two N×N matrices takes about 2N³ operations on 3N² numbers, so each number can be reused about N times. A systolic array physically implements this reuse. Each value enters at the edge once, then passes between neighbors, doing a multiply at every stop. Only the edge touches memory.

Systolic arrays were invented in 1978. They became important because AI suddenly created demand for them. [R] They are the first of the six features Jouppi ranks as most important in training accelerators. [J]

## Anatomy of a processing element

Your `pe.sv` is a good reference. Each PE holds:

- A multiplier: 8×8 gives a 16-bit product.
- An adder and accumulator: 32 bits in your design.
- Pipeline registers that pass operands to neighbors.

The cost breakdown surprises people. In an 8-bit output-stationary PE you have 8 + 8 + 32 = 48 flip-flops for one multiply. A flip-flop costs several gates, so storage and clocking are a large share of the PE. This drives the *grouped PE* idea in the next chapter.

## The three dataflows

What stays still decides what moves. Name the dataflow before you use any formula, because array dimensions mean different things in each. [R]

| Dataflow | Stationary | Moving | Good for | Costs |
| --- | --- | --- | --- | --- |
| Output-stationary | Accumulators | A rows, B columns | Long K, narrow data wires | Need a readout path, no weight reuse across tiles |
| Weight-stationary | Weights | Activations, partial sums | Many rows per weight tile | Wide partial sums move every hop, weight load time |
| Input-stationary | Activations | Weights, partial sums | Rarely the best for LLMs | Same as weight-stationary, mirrored |

TPUs are weight-stationary. Your `sys_array.sv` is output-stationary. Neither is universally right.

**Wire traffic comparison for int8.** An output-stationary PE moves 8 + 8 = 16 bits per cycle between neighbors. A weight-stationary PE moves 8 activation bits and 32 partial-sum bits, for 40 bits, once the weights are loaded. [D] Output-stationary is cheaper to run in steady state. Weight-stationary avoids a readout network and pairs naturally with accumulators outside the array.

## What a beat is

A **beat** is one tick of the array's clock. On every beat, each value moves one hop to its neighbor, and each PE does its work on whatever just arrived. The word systolic comes from the heartbeat for this reason: the whole array pumps together.

Everything in this chapter comes down to three questions about a beat:

- How many beats until every PE has work? That is fill.
- How many beats until the last PE has finished? That is drain.
- How much work does one PE do in one beat? That is packing.

## The timing that matters

For an output-stationary R×C array computing an R×C tile with reduction depth K, operands are skewed so element k reaches PE(r,c) at beat k + r + c.

`tile cycles = K + R + C − 2`

This is exactly `TOTAL_CYCLES` in your `tile_engine.sv`. Utilization for one isolated tile is `K / (K + R + C − 2)`.

### Watch a small one run

Take a 3×3 array with K = 4. PE(r,c) works during beats r+c through r+c+3.

| Beat | PEs working | Out of 9 | What is happening |
| --- | --- | --- | --- |
| 0 | 1 | 11% | Only the corner has data |
| 1 | 3 | 33% | Filling |
| 2 | 6 | 67% | Filling |
| 3 | 8 | 89% | Near the peak, and the corner is on its last value |
| 4 | 8 | 89% | The corner has finished |
| 5 | 6 | 67% | Draining |
| 6 | 3 | 33% | Draining |
| 7 | 1 | 11% | Only the far corner is left |

The total is 8 beats, which is 4 + 3 + 3 − 2. The array did 36 MACs in 72 slots, so it ran at 50%. With K this short, all nine PEs are never busy at once.

**Fill and drain are one tax.** Each PE works for exactly K beats, and sits idle for the other R+C−2. A PE near the first corner idles at the end, one near the far corner idles at the start, and one in the middle does some of each. So the array does not pay for fill and then again for drain. It pays R+C−2 once, split between the two ends. [D]

### Weight-stationary timing

The same diagonal appears, with three differences. Time for T tiles of M activation rows each, once weights are in place: [D]

`beats = T × M + R + C − 2`

- Weights must first shift in, one row per beat, so R beats per tile. TPU v1 quotes this: 256 cycles to shift a tile in. [O]
- Partial sums now travel down the column, so the last result leaves the bottom row R−1 beats after its activation entered. This is drain, and it is real wire traffic rather than idle time.
- The stream length is M, not K. So at decode, with M = 1, there is almost no stream to hide anything behind. This is why the rows-needed equation in *Matmul shape* exists.

Simulators often count one more thing. SCALE-Sim's standard estimate is `2R + C + T − 2`, which is this chapter's formula plus R beats to carry results out. [O] Both are right. They stop the clock at different moments, so check which one a paper is using.

### What isolated tiles cost

| Array | K=16 | K=64 | K=256 | K=4096 |
| --- | --- | --- | --- | --- |
| 8×8 | 53% | 82% | 95% | 100% |
| 32×32 | 20% | 50% | 80% | 98% |
| 128×128 | 6% | 20% | 50% | 94% |
| 256×256 | 3% | 11% | 33% | 89% |

Those are the numbers for isolated tiles. [D] They look terrible for big arrays with short K, and most of the next section is about not paying them.

**Rule of thumb.** Fraction lost = (R+C−2) ÷ (stream + R+C−2). For about 90% utilization, the stream must be roughly ten times R+C. [D]

## Cutting fill and drain

There are only four strategies. You can pay the tax less often, make the tax smaller, put other work in the gap, or make the idle beats cheap.

::viz beat

| Trick | Fill on 256×256 | What it costs | Source |
| --- | --- | --- | --- |
| Nothing | 510 per tile | Nothing | [D] |
| **Pay less often:** run tiles back to back | 510 once per stream | Shadow registers, tags that travel with data | [R, O] |
| **Pay less often:** order loops for long streams | 510 once per longer stream | More accumulators or more SRAM | [D] |
| **Shrink:** four 128×128 arrays | 254 | Four sets of control, twice the edge bandwidth | [R, D] |
| **Shrink:** 2×2 MACs per register stage | 254 | Longer wires, fanout of 2, lower top clock | [O, D] |
| **Shrink:** broadcast one operand | 255 | Long buffered wire across every row | [O, D] |
| **Shrink:** broadcast both operands | 0 | No longer really systolic, wires limit the clock | [O, D] |
| **Shrink:** feed from the diagonal | 255 | Feed wires into the array interior | [O] |
| **Shrink:** split columns with an adder tree (WS drain) | R÷S + log₂S | S output taps per column, tree area | [O, D] |
| **Fill the gap:** independent work | Unchanged, but used | A scheduler, a second stream | [R, D] |
| **Make it cheap:** clock-gate idle PEs | Unchanged in time | One gating cell, enable timing | [E] |

### Pay less often: run tiles back to back

This is the big one, and it comes first because nothing else gets close. Roune describes the idea directly: feed the next B while processing the current A and outputting the prior C, so the left, top and right edges are all busy on every cycle. [R]

Start tile 2 one beat behind tile 1, like cars in a convoy. Each PE finishes its last value of tile 1 and gets its first value of tile 2 on the very next beat. The diagonal band never goes away, and the ramp is paid once for the whole stream.

`T back-to-back tiles = T × K + R + C − 2`

For K = 256, here is what that buys. [D]

| Tiles in a row | 128×128 | 256×256 |
| --- | --- | --- |
| 1 | 50% | 33% |
| 4 | 80% | 67% |
| 16 | 94% | 89% |
| 64 | 99% | 97% |

This needs real hardware changes, covered in *What overlap actually needs* below.

**What overlap cannot do.** It only works when the next tile does not depend on this one. Tiles in one matmul are independent. The next layer is not, because it needs this layer's last output before it can finish, and in a Transformer a LayerNorm or softmax usually sits between them. So you pay the ramp at least once per dependent stage, and this is the number to count at batch 1. [D]

### Pay less often: choose a long stream

The stream is whatever you loop over while the same thing stays in place. For output-stationary, keep the accumulators and walk all of K before you read out. Your `init_acc=0` already allows this: tiles that continue the same sum need no readout between them. For weight-stationary, hold weights while as many rows as possible pass through. [D]

### Shrink: put more MACs behind each register

Fill is counted in hops. If a value crosses two PEs per beat, there are half as many hops. Gemmini exposes this as a generator setting: its array is a mesh of tiles, with pure combinational logic inside a tile and pipeline registers between tiles. [O] This is the packing section's Tm and Tn, seen from the timing side.

### Shrink: broadcast instead of hop

A row can receive one operand on a wire that reaches all its PEs in the same beat. Then a PE's start time depends only on its row, and fill falls from R+C−2 to R−1. Broadcast both operands, and fill is zero.

This gives up the thing that makes an array systolic. A hop is short wire and tiny fanout, whatever the array size. A broadcast's delay grows with length, at the order of 100 ps per mm from *Area, clock, power*. [O] So broadcast is fine over a short span and hopeless across a big one. A 64-column int8 row is under 1 mm at 5 nm, on the rough cell sizes in that chapter. A 256-wide array is about 3 mm, which uses most of a 500 ps clock at 2 GHz before fanout or logic. [D]

Treat it as a dial, not a switch: broadcast within a group of PEs, and hop between groups.

### Shrink: feed from the diagonal

Axon injects operands at the PEs on the main diagonal, and sends them outward in both directions. A row's operand starts at PE(r,r) and a column's starts at PE(c,c), so they meet at PE(r,c) at beat |r−c|. Both arrive together with no skew at all, and fill falls to max(R,C)−1, half of what a square array normally pays. [O]

The paper reports up to 2× runtime on shapes where fill dominates, and about 0.2% area overhead at 16×16 in a 7 nm library. Its average gain at 256×256 is 1.76×. [O] It was implemented for output-stationary. Weight-stationary needs extra muxes. Treat these as one paper's results, not as established practice.

### Shrink: end the column chain early (weight-stationary)

A 256-long chain takes 256 beats to drain. Cut it into S segments, then add the S partial sums at the bottom with a small tree. Drain becomes R÷S + log₂S beats. With S = 4 that is 66, not 256. [D] The extreme version is a full adder tree per column. The DiP paper notes that the Meissa design did this and that large trees hurt scaling, through wiring congestion and deep pipelines. [O] *Reduction trees* covers the trade.

### Fill the gap with other work

If a gap is unavoidable, use it.

- **Block-diagonal packing.** Two independent small matmuls can occupy the upper-left and lower-right of the array at once, with no hardware feature. [R]
- **Two arrays, staggered.** While one array is at a dependency stall, the other runs a different stream. [D]
- **Overlap attention with feed-forward.** Roune notes this needs deeper pipelines or a model change that runs them in parallel. [R]
- **A second request.** Two independent users can alternate, so one's stage boundary hides behind the other's work. At batch 1 you do not have this option. [D]

### Make it cheap

Clock gating does not give back a single beat, but it means idle PEs burn little dynamic power. [E] Time and energy are separate budgets here: at batch 1 you may accept the beats and care only about the battery.

## What overlap actually needs

Your `tile_engine.sv` runs IDLE → RUN → DONE, one tile at a time. Three things stop you from just feeding tiles back to back.

**1. Clear must travel with the data.** Your `clear_acc` fires at one moment for every PE. That is safe today only because the array is empty when it fires. With overlap, PE(0,0) starts tile 2 while PE(7,7) is still on tile 1, so a global clear would wipe a running sum. The fix is one extra bit that rides beside the first operand, so each PE clears on the beat tile 2 reaches it. TPU v1 describes the same idea for weights: they take effect "with the advancing wave". [O]

| Tag | Rides with | Tells a PE to |
| --- | --- | --- |
| `first` | First K element | Start a new sum rather than add |
| `last` | Final K element | Hand the sum to the shadow register |
| `flip` | First row of a tile (WS) | Switch to the preloaded weights |
| `valid` | Every operand | Hold still, since this slot is a bubble |

**2. The finished sum needs somewhere to go.** On the very next beat, the accumulator is needed for tile 2. So each PE holds a second register, called a shadow. Gemmini's PE works this way: two registers, `c1` and `c2`, with a travelling `propagate` bit picking which one accumulates and which one is read out. In weight-stationary mode the same two registers hold active and next weights. [O]

This is the price of overlap: 32 more flops per PE, taking an int8 PE from 48 to 80 flops. Packing K, below, shrinks this cost a lot.

**3. Results must leave as fast as they arrive.** Each column shifts one value per beat, so a tile takes R beats to leave, while the next tile takes K beats to finish. [D]

`tile time = max(K, R)`

Our beat simulator confirms the edge exactly: K = R runs with no collisions, and K = R−1 overwrites results. There is a catch. A plain shift-enable for the whole column is not enough, because the next tile's first results land while the old ones are leaving, and they would get dragged along. It takes one tag bit per slot, so that only the old tile moves. [D]

The weight-stationary version is the same rule in another costume. New weights follow one beat behind the flip wave, deepest row first, so they need R beats: M ≥ R, which is the rows-needed equation at ρ = 1. The simulator confirms this as well: M = R works, and M = R−1 swaps too early. [D]

TPU v1 shows what a full design looks like: one 64 KiB weight tile active, one for double-buffering to hide the 256 shift cycles, and a Weight FIFO four tiles deep behind them. [O]

## More than one element per beat

So far each PE does one MAC per beat. A PE can do more, and this is one of the most useful knobs you have, because it changes the array's shape without changing how many MACs you pay for.

Call a PE doing several MACs a **cell**. A cell has three packing factors, one for each matmul dimension:

- **Tk**: reduction elements per beat. This is the grouped PE from *Reduction trees*, where it is called G.
- **Tm**: output rows per cell.
- **Tn**: output columns per cell.

For an array of Rc × Cc cells, these formulas hold. The simulator checks them for several cell shapes, with real values. [D]

| Quantity | Formula |
| --- | --- |
| MACs | Rc × Cc × Tm × Tn × Tk |
| Output tile | (Rc × Tm) × (Cc × Tn) |
| Beats per tile | ⌈K ÷ Tk⌉ + Rc + Cc − 2 |
| Operands in per beat | (Rc × Tm + Cc × Tn) × Tk |
| Flops per MAC, int8 | 8(Tm + Tn) ÷ (Tm × Tn) + 32 ÷ Tk |

The last formula matches 16 + 32/G when Tm = Tn = 1. The shadow register adds another 32 ÷ Tk.

### Packing K: a small dot product per beat

The cell takes Tk values of A and Tk values of B, multiplies them pairwise, and sums them with a small tree into **one** accumulator.

For Tk = 4, one beat does `acc += a0·b0 + a1·b1 + a2·b2 + a3·b3`.

- **Good:** K takes Tk× fewer beats. One accumulator serves Tk multipliers, so accumulators, shadows, and readout traffic all shrink by Tk.
- **Bad:** The tree sits in the critical path, at log₂Tk adder levels. Operand wires get Tk× wider. Reuse falls, because the array is smaller for the same MACs.
- **Watch out:** K rounds up to a multiple of Tk. K = 3 on Tk = 4 wastes a quarter of the multipliers.
- **Watch out:** Isolated-tile utilization can get **worse**, because K shrinks by Tk while fill only shrinks by about √Tk. Packing K helps latency, not the utilization of short tiles.
- **Watch out:** The readout rule becomes K ÷ Tk ≥ R.

Nvidia's tensor cores sit far toward this end, which is why they can act as systolic arrays at chip level, as Roune treats them, while differing at circuit level. [R, O]

### Packing M and N: a small outer product per beat

The cell takes Tm values of A and Tn values of B and does every pairing, into Tm × Tn accumulators. A 2×2 cell does four MACs from four operands, where four separate PEs would need eight operand registers.

- **Good:** Fill halves, with the same output tile and the **same** edge traffic. Fewer operand flops.
- **Bad:** Each operand drives Tm or Tn multipliers in the same beat, and has to cross a wider cell. The cost is wire delay and fanout, not logic depth.
- **Unchanged:** Accumulators stay at one per MAC.

This is the Gemmini tile from earlier. Taken to its limit, one giant cell has no hops at all: you have an outer-product engine, the structure behind Arm's SME that a reader raises with Roune. His answer is that systolic arrays remain the most area- and power-efficient choice for dedicated AI. [R]

### Packing bits: several narrow numbers in one wide lane

A multiplier can often split. An 8×8 multiplier is built from partial products, and it holds four 4×4 multipliers inside it, since (16·a₁ + a₀)(16·b₁ + b₀) has four terms. The same 8-bit wires carry two 4-bit values each. Gate the carries between the pieces, and one PE does more narrow MACs per beat. [O]

The reverse also works, and TPU v1 did it: the matrix unit ran at half speed with one 16-bit operand, and quarter speed with two, building wide products out of 8-bit multipliers over several beats. [O]

| Example | Packing | Source |
| --- | --- | --- |
| TPU v1 | 16-bit operands take 2 or 4 passes on 8-bit hardware | [O] |
| Ironwood | Described as four 256×256 BF16 or four 512×512 FP8 arrays | [J] |
| RTX Blackwell | FP4 at double the FP8 rate | [O] |

Read the Ironwood line with care. A 512 side implies four FP8 MACs per BF16 cell, but the published peak is only 2× BF16. *Source audit* covers this mismatch, so the paper does not settle the real packing factor. [J, D]

Roune's packing notes are the software side of this. Narrow formats need several rows packed into one vector register, so that each register stays full, which forces row counts to be a multiple of that factor. [R]

### Doing fewer multiplies

Two more tricks change the arithmetic itself. They are niche, but they show the design space is wider than it looks.

- **Fast inner product.** Winograd's 1968 identity gets two products from one multiply, using cheap pre-additions: `(a₀ + b₁)(a₁ + b₀)` contains `a₀b₀ + a₁b₁`, plus correction terms that depend on only one matrix. Pogue and Nicolici's FFIP puts this into systolic arrays for about half the multipliers at fixed-point precision. [O]
- **Shared-multiplier packing.** Place two weights in a wide multiplier with a gap between them, and one multiply by a shared activation gives both products. FPGA designs use this with wide DSP multipliers. [O]

### Same 64 MACs, six ways

All of these are 64 MACs, at K = 64, 16 tiles. [D]

| Array × cell | Output tile | Fill | Beats per tile | Isolated | Back to back | Operands per beat | Flops per MAC, with shadow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8×8, plain | 8×8 | 14 | 64 | 82% | 98.7% | 16 | 80 |
| 4×4 of 2×2×1 | 8×8 | 6 | 64 | 91% | 99.4% | 16 | 72 |
| 4×4 of 1×1×4 | 4×4 | 6 | 16 | 73% | 97.7% | 32 | 32 |
| 2×2 of 2×2×4 | 4×4 | 2 | 16 | 89% | 99.2% | 32 | 24 |
| 1×16 of 1×1×4 | 1×16 | 15 | 16 | 52% | 94.5% | 68 | 32 |
| 1×64, plain | 1×64 | 63 | 64 | 50% | 94.2% | 65 | 80 |

Flops are the pipeline and accumulator registers only. Fill here assumes hops, so a row with a broadcast operand has zero fill.

Read down the columns. Packing M and N (row 2) is close to free, if wires allow it. Packing K (row 3) cuts flops per MAC by more than half, and pays for it with twice the operand traffic. Combining both (row 4) has the shortest fill and fewest flops, and pays with the widest cell. The two rows at the bottom pay more edge traffic because they have no reuse in one direction. That is the price of M = 1, not a design flaw.

### Choosing what to pack

| If this is your situation | Do this |
| --- | --- |
| M is small | Do not pack Tm. Pack Tk, or use a row |
| K is short or odd | Keep Tk small, since rounding wastes MACs |
| Fill hurts and wires are short | Pack Tm and Tn |
| Flops or clock power dominate | Pack Tk, since one accumulator serves several MACs |
| SRAM bandwidth binds | Keep the array large and square, and avoid Tk |
| Clock is the priority | Keep cells small, with one MAC per register |
| You need overlap but shadows cost too much | Pack Tk, which divides shadow cost by Tk |

## Skew: the hidden triangle

Skew means row r starts r beats late, so that operands meet at the right PE on the right beat. Roune skips it, saying it is not relevant to his point. [R] It matters if you build one.

Row r needs r delay stages, so one edge needs R(R−1)/2 registers. [D]

| Side | Delay stages per edge | Bits for int8 in on two edges | Bits for 32-bit deskew out |
| --- | --- | --- | --- |
| 8 | 28 | 448 | 896 |
| 128 | 8,128 | 130,048 | 260,096 |
| 256 | 32,640 | 522,240 | 1,044,480 |

The output column only applies to weight-stationary arrays whose consumer needs aligned rows. Your options:

| Option | How it works | Cost |
| --- | --- | --- |
| Triangle of flops or FIFOs | The textbook edge | The table above |
| Staggered SRAM addresses | Each row's bank reads with its own offset | No delay registers, needs a bank per row |
| Index logic | Your `k_idx = clock_count − row` | A mux per row that grows with K |
| Let the consumer stay skewed | Downstream lanes also run a beat apart | Skew must be planned chip-wide |
| Diagonal feed | Axon needs no skew | Feed wires into the interior [O] |
| Permuted weights | DiP moves inputs diagonally and permutes the weights, removing the FIFOs | New dataflow, one paper's result [O] |

DiP reports up to 1.49× throughput and up to 2.02× energy efficiency per area over its weight-stationary baseline in a 22 nm flow. [O] Again, that is one paper against its own baseline.

## Reading results out

Fill gets the attention, but readout is often the real problem. An R×C tile of 32-bit sums is a lot of data.

| Method | Beats to empty | Wires | Good for |
| --- | --- | --- | --- |
| Parallel bus, like your `ACC` | 1 | R × C × 32 | Tiny arrays only |
| Column shift, array waits | R, exposed | C × 32 | Simple small designs |
| Column shift with shadow | R, hidden if K ≥ R | C × 32 | Output-stationary at scale |
| Sums flow out the bottom (WS) | Hidden in the stream | C × 32 | TPUs, with accumulators outside |
| Requantize at the edge | As above, narrower memory | C × 32, then C × 8 | Inference |

TPU v1 used the fourth: 4 MiB of accumulators below the array, as 4,096 entries of 256 × 32 bits. Google sized it from the roofline knee of about 1,350 operations per byte, rounded up to 2,048, and doubled for double-buffering. [O] Notice the method: a memory sized from a bandwidth ratio, not from a guess.

## Bubbles and stalls

A global stall has the same problem as a global clear: no signal reaches every PE in one beat. So large arrays do not freeze. They let bubbles flow. A `valid` bit rides with each operand, and a PE simply holds its registers when the bit is low. Gemmini's PE does this. [O]

To keep bubbles rare, you need the feed to be reliable. TPU v1's array stalls if activations or weights are not ready, and it stages weights through that four-deep FIFO to reduce the chance. [O]

Feeding the array on every beat is what Roune calls cycle tyranny, and he knows of a chip that could never reach full utilization because its controlling core could not keep up. [R] His remedies are 8×128 two-dimensional registers, hardware queues, unrolled loops, slower array clocks, and DMAs that stream straight into the array. [R] *Control and feeding* covers them. Packing interacts with this directly: Tk multiplies the operands needed per beat, while the array issues the same MACs in fewer beats.

## Repair and test

A regular array makes a defect cheap to repair. Ironwood added a redundant row to the MXU, and it runs a Functional Built-In Self-Test engine inside it, to catch bad chips in manufacturing and in the field. [J] To skip a dead row, operands and partial sums need a bypass mux around each row. [O] A longer path through that mux lands on the hop you most want to keep short, so plan for it from the start.

## One PE, two dataflows

You do not have to choose forever. Gemmini can be built with output-stationary, weight-stationary, or both, picked at runtime. In "both" mode, the two registers in each PE serve as double-buffered accumulators in one mode, and as active and next weights in the other. [O] The price is muxes in the most replicated cell in the chip, plus an accumulator memory outside the array for weight-stationary mode.

## Your chip

Your workload has 13 dependent stages per token: six in each of two layers, plus the LM head. [D] Overlap cannot hide those boundaries, so the ramp is paid 13 times per token, and batch 1 gives you no second stream to fill it. These estimates come from the model, not from your RTL.

| 64-MAC design | Work beats per token | Ramp, 13 stages | Total | Accumulators |
| --- | --- | --- | --- | --- |
| 1×64 row, operand hops | 2,048 | 819 | 2,867 | 64 |
| 1×64 row, operand broadcast | 2,048 | 0 | 2,048 | 64 |
| 1×16 of Tk=4, operand hops | 1,920 | 195 | 2,115 | 16 |
| 1×16 of Tk=4, operand broadcast | 1,920 | 0 | 1,920 | 16 |

Three points follow. Broadcasting `x` removes a 40% slowdown, and at this size the wire is short, so that is a clear win. [D] The `ROWS=1, COLS=64` setting of your current RTL is the first row of this table.

The Tk=4 row is worth a look. It is 6% faster, because attention's narrow N of 32 wastes half of a 64-wide row. It also has a quarter of the accumulators, which is 1,536 fewer flops at 32 bits, and it needs only 16 readout lanes. The price is a two-level adder tree in the PE and a 4-byte `x` bus. At your clock that tree is probably affordable, but only synthesis can confirm it. Every K in your model is a multiple of 4, apart from attention values at odd context lengths, where the last group is partly empty.

Weight traffic stays at 64 bytes per beat in all four rows. That is the real limit, and no packing changes it at M = 1.

## Why bigger is better

Compare one 2N×2N array with four N×N arrays. They do the same math per cycle. The large one needs one control stream rather than four, and its operand vectors are only twice as wide, rather than four times the total. [R]

| Going from N to 2N | Change |
| --- | --- |
| Math per cycle | 4× |
| Edge bandwidth needed | 2× |
| Scalar work per unit of math | ÷4 |
| Vector work per unit of math | ÷2 |
| Bytes moved per MAC | ÷2 |

From a 4×4 array to a 128×128 array is five doublings, so 1024× less scalar work and 32× less vector work. With TPU's 8×128 registers the scalar saving reaches 8192×. [R] Roune attributes much of the TPU team's ability to keep up with Nvidia to this one fact.

Bigger arrays also make the whole chip easier to build. Arrays and caches are simple and keep getting more area. The complex CPU-like logic shrinks as a fraction of the die, so it needs less optimization. [R]

## Why not build it infinitely large

Four forces limit array size. [R]

**1. Shape mismatch.** An N×N array cannot efficiently do a smaller matmul. This is the main force, covered in the next chapter.

**2. Batch and latency.** Large arrays need more rows per weight tile, which means larger batches, more KV cache and more token latency.

**3. The rest of the chip.** Scalar, vector and memory blocks need space. TPU v1 was nearly one big array, but it needed an external CPU to drive it, so that area moved to another chip you still had to buy.

**4. Heat density.** Concentrated computing concentrates heat, though Roune says this can be managed and should not limit size.

Roune's reading is that attention is what holds array size down. Without it, we might have 1024×1024 arrays today.

## Replication

Once an array hits its size limit, you add more of them. Replication multiplies all costs and benefits, with small savings from sharing. [R] Ironwood's move from 128² arrays to 256² arrays fits this logic. My reading is that feed-forward efficiency outweighed attention utilization for a workload that was 74% Transformer by 2026. [J, D]

**Decomposable arrays** split into smaller ones on demand, as Etched and MatX describe. [R, F] The catch: to run in small mode you must already have paid for the on-chip bandwidth and control that small mode needs, so you lose the area saving of a large array. You can still gain some power. [R]

**Heterogeneous arrays** are Roune's suggestion: one big array for feed-forward and several small ones for attention, identical except for `VECTOR_WIDTH`. Whatever ratio you pick is better than the 100% to one side that uniform designs give you. [R]

## Numerics: making each cell cheap

Multiplier area grows roughly with the product of operand widths, so going narrower has compounding benefits: less multiplier area, less energy, half the memory traffic, half the capacity.

| Format | Bits (sign, exponent, mantissa) | Key property |
| --- | --- | --- |
| FP32 | 1, 8, 23 | Original baseline, expensive |
| FP16 | 1, 5, 10 | Needs loss scaling for training |
| BF16 | 1, 8, 7 | FP32 range, drop-in for training, cheaper than FP16 |
| FP8 E4M3 | 1, 4, 3 | Forward pass |
| FP8 E5M2 | 1, 5, 2 | More range, used for gradients |
| Int8 | 8-bit integer | Simplest hardware, enough for inference |
| FP4 | 1, 2, 1 | Barely usable, weights easier than activations |
| MXFP4 and NVFP4 | 4 bits plus shared scale | About 4.25 bits, for models that nearly work in 4 |

The key design rules:

- **Multiply narrow, accumulate wide.** Multiplies tolerate low precision. Sums do not. BF16 multiplies accumulate in FP32. [J]
- **Range beats precision.** BF16 was the first format to give the exponent more bits than the fraction, and later narrow formats followed that lead. [J]
- **Diminishing returns below 8 bits.** Accumulators stop shrinking, and at FP4 the sum may cost more than the product. An FP4 product fits exactly in int8 fixed point, so integer summation could be cheaper. [R]
- **Bandwidth gains do not diminish.** FP4 still halves memory traffic even when the compute saving tails off. [R]
- **Every format costs area in the most replicated cell.** Nvidia's many formats must carry a real price. Google avoids paying for FP16. [R]
- **Array numerics can differ from vector numerics.** You can have int8 arrays and FP32 vectors, since the vector unit is small. [R]

How low can you go? Eight bits is always enough for inference if quantization is done right. Five bits is usually enough. Four is sometimes enough. [R] TPU v1 was pure int8 and suffered in adoption because in 2015 few people knew how to quantize. That barrier has gone, so an int8-only inference chip is a real contender. [R]

## Sparsity

Unstructured sparsity does not fit a regular array. Structured A:B sparsity does. In 2:4 sparsity, each group of 4 holds at most 2 non-zeros. Nvidia has supported it since Ampere for about 2× throughput at modest area, with metadata of 3 bits per 4 entries. [R]

Roune proposes going further with 1:2 sparsity and a 7-bit integer: an 8-bit format with one position bit and seven value bits. It is simple and SIMD-friendly, but unproven.

Be careful with this one. Roune is an advocate, and the same document records a practitioner's objection that accuracy drops are too large and that almost nobody uses 2:4 at scale. [R] Sparsity also applies mainly to weights. It does not speed up attention's activation-times-activation matmuls, and it does nothing for KV cache. Treat it as a bet that depends on your model team.

## Systolic tuning reference

| Knob | Raise it to | Pay with | Watch for |
| --- | --- | --- | --- |
| Side length | Cut control and vector work, cut bytes per MAC | Padding on small K, larger batch | K of attention heads |
| Array count | Add throughput, keep shape flexible | Duplicated edge and control | Area, SRAM ports |
| K-depth only | Fewer overheads for feed-forward | Vector width mismatch, less parallelism | Limit on within-layer split |
| Clock | Add throughput with less area | Voltage, heat, feeding pressure | Cycle tyranny |
| Weight load bandwidth | Lower required rows, lower batch | SRAM and HBM bandwidth | Expensive everywhere upstream |
| Double buffering | Hide load and drain | Registers in every PE | Area of 65,536 extra bytes |
| Tile overlap | Pay fill once per stream | Travelling tags, shadow registers | Needs K ≥ R, or M ≥ R for weights |
| Tk (pack K) | Fewer beats, accumulators and shadows | Tree in the critical path, wider buses | Rounding of K, readout rule |
| Tm, Tn (pack M, N) | Shorter fill, fewer operand flops | Wire delay, fanout | Padding when M or N is small |
| Broadcast span | Shorter fill, down to zero | Long wires, top clock | Delay grows with length |
| Diagonal feed | Half the fill, no skew | Interior feed wires | Published results only |
| Column split | Shorter weight-stationary drain | Output taps, adder tree | Congestion at large S |
| Skew method | Fewer delay registers | SRAM banks, planning | Triangles grow with the square of side |
| Readout width | Faster emptying | Wires | Tile time is max(K, R) |
| Precision | Throughput, bandwidth, capacity | Accuracy, format zoo | Accumulator share of area |
| Sparsity support | Up to 2× on weights | Muxes in the array, model work | Real adoption |
| Spare rows | Yield | Small area, repair muxes | Fuse and test flow |

## Further reading

The papers behind the [O] items in this chapter. These sit outside your four PDFs. I read the relevant passages directly, apart from FFIP, where I read only the abstract and introduction.

- Jouppi et al., [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760). TPU v1 double-buffering, advancing wave, accumulators, 16-bit speeds.
- Genc et al., [Gemmini](https://github.com/ucb-bar/gemmini). Open-source generator. Read `PE.scala` for `c1`, `c2` and `propagate`, and the Chipyard docs for tiles and meshes.
- Nayan et al., [Axon](https://arxiv.org/abs/2501.06043). Diagonal feed and half the fill.
- Abdelmaksoud et al., [DiP](https://arxiv.org/abs/2412.09709). Removing the synchronization FIFOs.
- Samajdar et al., [SCALE-Sim](https://arxiv.org/abs/1811.02883). The standard analytical cycle model.
- Pogue and Nicolici, [Fast Inner-Product Algorithms and Architectures](https://arxiv.org/abs/2311.12224). Half the multipliers.
