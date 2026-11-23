# Raw material for blog part 2: what was hard, what broke, and what it taught

*Not a draft. A collection of the intricacies and blockers from the ISA, the functional simulator and the first RTL, kept in one place so part 2 can be written from it. Each item points to the note with the full story. Started 2026-11-21; add to it as the work goes on.*

## The theme

- **Timing a program is not running it.** The ISA program was timed for days (note 28) and looked right. The first run on real data found 6 bugs in decode and 14 groups in prefill. Several of them made the chip look *faster* than it is.
- **Each new level of reality found what the previous one couldn't:**
  - the performance model counted bytes and operations
  - the timing simulator added the program's real control flow, queues and semaphores
  - the functional simulator added values
  - the RTL added gates, and with them latency
- **A yardstick for "close enough".** "Within 3%" means nothing alone. It means something once two equally valid versions of the golden model are shown to be 2–3% apart too: the rounding floor (notes 30, 31).

## Bugs only real data could find (note 31)

- **The conv history never advanced in plain decode.** The register that moves it was set only by the speculative path. Every step re-read the same 3 rows.
- **The per-matrix weight scale was never applied.** The descriptor field had been left at 1.0, and a field per matrix would have overflowed the descriptor memory (about 1,000 matrices). Now it travels in a 32-byte header with each matrix.
- **A semaphore off by one at the prefill chunk boundary.** Each chunk's embedding added a spare "residual ready" count, so in chunk 2 every layer could start before the previous one finished. **The race made prefill look 0.2 ms per chunk faster.** Timing can't see a race; values can.
- **A DMA fence that passes too early.** A fence in a DMA queue passes once the reads ahead of it are *requested*, about 150 ns before the bytes land. Signal from the last DMA itself instead.
- **Sign errors in the chunked DeltaNet:** v + γ·Sᵀk instead of v − γ·Sᵀk, and +A where the solve needs −A. Fixed with a one-bit `neg` flag on the vector ops.
- **Stale padding that wipes out real products.** Short matrix operands are padded to whole segments. Stale exponents left in the padding raised the segment's shared exponent and rounded the real products away. Clear the padding, exponent −126.
- **An off-by-one in the triangular solve's column register.** It showed up as a 4.4% error in Δ, and was found by probing one tensor at a time.
- **Two 9B coincidences.** The 9B's hidden size happens to equal its DeltaNet value width (4,096), and its q-projection width happens to equal 2 × hidden. The program used the wrong one in each place, and only a second model (Qwen3.5-0.8B) exposed it.

## The golden model has to match the chip, not the other way round

- **KV rounding:** the chip stores fp16 scales; the golden model used fp32. With only a few positions, one flipped int8 code in the softmax moves the output by 1%. Now golden follows the chip's rule (`int8_head_chip`).
- **SiLU returned −0 and kept subnormals,** against the chip's own rule. The RTL test found it.
- **The golden model's chunked DeltaNet uses a library solve and float matrix products.** Fine for quality, but it stops prefill's layer checks at int8 flips (about 10⁻³) instead of fp32 rounding. A bit-exact golden for that piece is open.

## Bit-exact RTL (note 32)

- **"Bit-exact" includes torch's quirks.** Torch's `ldexp` goes through 2ⁿ, so any result of 2¹²⁷ or more becomes ∞. Found as a 1% mismatch, all above x = 88.03; now part of the contract.
- **One FMA is every elementwise op.** ADD, SUB and MUL are FMAs with a 1 or a 0, with the same bits. So the vector lane has one arithmetic unit.
- **The FMA passed 20,000 cases on the first try,** because the rounding argument was written down before the code.
- **Verilog traps:**
  - `small` is a reserved word
  - mixing signed and unsigned values in one expression makes the whole expression unsigned (every int8 code came out 0)
  - driving and reading an FMA in one combinational block loops forever in the simulator (a hang, not an error)
- **A table index that needs a division:** 1/√x's index depends on an fp32-rounded (u − 1)/3, so the hardware needs an exactly rounded divide-by-3.

## What the RTL revealed about performance (note 32)

- **The special functions cost more than they should:** exp is 3.8 FMAs of area and sigmoid 6.6, because the contract was written as fp32 steps. A fixed-point contract would be smaller; that's a golden change plus a quality re-check.
- **The FMA needs about 8 pipeline stages at 1 GHz** by synthesis, and the slow parts are the shifters, not the adders. Kogge-Stone carry networks didn't help.
- **Deep pipelines need independent work.** The DeltaNet step is a chain of 32 dependent FMAs per lane. With an 8-cycle FMA, the speculative step loses 11%. Interleaving 4 heads in the DN unit recovers it, at +128 KB of SRAM, with no change to the arithmetic. The timing model hadn't seen it because it counted operations, not dependencies.

## Blockers and open threads (as of 2026-11-21)

- **Instruction memory:** the lite prefill fills 95% (7,780 of 8,192 bundles), because the peeled attention block is repeated per KV head. Being looped.
- **Synthesis time:** the int8 quantizer (32 variable shifters) didn't finish in 10 minutes. It needs a cheaper structure.
- **Not yet run on values:** the speculative step (MTP drafts and replay). In progress, along with a race check using random unit speeds.
- **Costs** (wafer, PHY, board) and **power:** deferred by choice.

## From a verified score to a returned token (note 31, part 3)

The speculative verifier produced all 18 rows of logits exactly, yet the host saw repeated tokens. On a multi-token acceptance, the DMA wrote successive host addresses while rereading the first SRAM word. Comparing the host stream exposed a stride bug that a logit-only test would have missed. After the fix the host received all 18 ordinary-decode tokens. The independent golden greedy run agreed on 13.

Looping over KV heads reduced the lite 9B program from 7,780 to 5,516 bundles in an 8,192-slot memory. The two-chunk functional rerun on 0.8B still agreed with the golden model on the first token and six following decode steps. Four schedules of a four-layer prefill also produced identical hidden, KV, recurrent and convolution memory regions. The full 24-layer prefill produced matching bytes in all four regions under two schedules.

## The multiply inside the FMA (note 32, part 2)

Giving the FMA eight pipeline stages still left a full significand multiply on one stage. Splitting that multiply into four 12-bit partial products moved the delay-oriented ASAP7 estimate from 2.272 ns to 0.919 ns (the area-oriented path fell from 1.737 ns to 1.091 ns). The 20,000 arithmetic results stayed identical. Routing and register timing still have to fit inside the clock period.

The quantizer now finds a candidate group exponent per lane and limits how far each significand must shift. The restructured 32-value block mapped to 548 µm² in the area-oriented run, with a 2.871 ns path. That gives the next timing iteration an actual bottleneck to measure.

## Correct is not fast (note 32, part 4)

- **Every command passed bit-exact while running at a small fraction of its intended speed.** The first full vector unit (Codex, while Claude was unavailable) was stop-and-wait: issue a beat, wait the FMA's 8 cycles, write, repeat.
  - elementwise ops: about 47 elements per clock against 512
  - special functions: about 0.2 SiLU per clock against 64
  - the softmax tile: about 4,070 clocks per 1,024-score row against 10
  - Correctness tests can't see this. A timing run with each command at its RTL cost can: as built, the 9B would decode at 36.6 tok/s instead of 90, and a 32K prompt would take 484 s instead of 5.7.
- **The DeltaNet controller's slowness went into the performance model.** Its measured cycles were real, but they measured a controller that waited after every op, even independent ones. Folding them into the model lowered the target from 315 to 280 tok/s. Re-scheduling the same operations by dependency gave 302, with identical bits.
  - **The trick that made it fit:** compute d·S while S is being read for S·k anyway, and write it back in place. The update pass then needs one independent FMA per row group.
- **The old model had a mistake too:** it counted 3 passes over the DeltaNet state per token, but the rule needs 4 FMA operations per element. Writing the RTL is what exposed it.
- **Freeze, don't buffer.** When a streaming unit's write stalls, the standard answer is to freeze the whole pipeline with a clock enable. A buffer for a 136-stage special-function pipeline would need 136 beats of storage; the clock enable is one gate.
- **The 8-beat wave.** For CONV4 (4 chained taps) and RoPE (2), issue tap 0 of 8 beats, then tap 1 of the same 8. Each beat's previous result leaves the 8-stage FMA exactly when it's needed again, so no accumulator registers are needed.
- **Small bugs, big symptoms:**
  - a 25-bit field declared as 24 bits dropped the "last" flag, and no special-function result was ever written
  - the cycle a command finishes has to count as idle, or the next command is dropped
  - a tree depth hard-coded for 128-wide heads broke at the 16-wide test size
- **Simulation time became a constraint:** replicating the special-function units put hundreds of pipelined FMAs in one bench (9 minutes in Icarus), and Verilator can't link on this Mac's current command-line tools.

## Blockers and open threads (updated 2026-11-21, evening)

- **The softmax tile** now issues 128 positions per clock within a row. Overlapping rows is still the largest timing opportunity in prefill.
- **Row commands run one row at a time,** each waiting out its function's latency. The modeled 31 s to 9 s prefill improvement assumes the tile and reductions already overlap rows.
- **Short rows waste box beats** (128-wide rows in a 512-lane beat); QKV and QUANT carry per-command latency.
- **Instruction memory is solved** (5,516 of 8,192 bundles after looping over KV heads).
- **The quantizer's 2.9 ns path** needs pipelining for 1 GHz.
- **Toolchain:** Verilator needs a working C++ SDK on this Mac.

## The softmax tile after the first streaming pass (note 32, part 5)

- The 128-position pipeline cut a 1,024-position row to 196 clocks with ready SRAM ports. A 256-position prefill row still takes 184 clocks because the two passes and reduction tree have fixed latency. At 1 GHz the timing model projects 88.6 tok/s decode and 86.1 s for a 32K prompt. Giving only the tile its architectural rate drops that prompt to 35.8 s.
- Port masks stop unused lanes from reading SRAM, so the words on those lanes can retain data from the previous beat. The softmax tile needs a second mask inside its arithmetic. A nonexistent score contributes negative infinity to the max and zero to the R1 sum. The test deliberately leaves a stale positive score on an unused lane.
- The box command's streaming read port used the mask of the beat being written eight FMA clocks later. At startup that mask was unknown, so the first decoded ADD fetched no operands and wrote unknown bytes. The block bench had supplied every lane regardless. It now honors the read mask, and the SRAM frontend uses separate issue and write masks.
- EXP produced the right value and still never finished. Before the first row command launched, an uninitialized dispatch-valid bit entered its tag delay. That bit arrived at writeback and made the in-flight beat count unknown. Resetting validity and gating dispatch on the streaming state keeps the counter defined. The exec bench now starts with a cold EXP.
- The sub-beat count depended on the SFU pool width, but its continuous assignment passed only the column to the function. Switching pools while that column stayed at zero could retire the beat early and leave old values in its tail. The count function now receives the rate and row width explicitly.
- Icarus spent minutes inferring sensitivity for lane muxes that referred to full buses or unpacked port arrays. Naming the lane words before selecting operands let elaboration move on. Explicit 16-bit write columns also eliminated warnings that repeated once per lane.
- Row interleaving hides dependent FMA latency when there is a ready operation from another row. The reduction still needs issue slots on the shared bank for each accumulation beat and each tree level. The one-row-per-clock projection needs a schedule that accounts for both.

## Scheduling softmax across rows (note 32, part 6)

- Issue capacity depends on lane placement. At a 16-clock period, full-width subtraction and V multiplication occupy complementary sets of clocks because their 72-clock offset is 8 modulo 16. The 256-lane first sum level then has no clock with both quarters free. A 17-clock calendar leaves room for it while preserving the nine tree levels.
- The two buffers that arrive together have different lifetimes. Scaled scores can be released at subtraction. V scales have to wait for the matching probability to leave exp. Per-row V scales would use six full-width slots at period 17. The 9B layout broadcasts them, so one FP16 scale vector can serve the whole descriptor.
- Write acceptance and ISA completion can occur on different clocks. While the bank and exp pipeline are frozen on a port stall, input and output staging must remember transfers the SRAM has already accepted. The completion signal waits for write visibility.
- Serializing a descriptor with shifted aliases can still change a future row's input. The functional simulator reads the descriptor's inputs before it writes, so an overlapping row needs the old bytes preserved. Same-row max and sum updates are safe when the row addresses are disjoint.
- The implemented calendar shares the exp and quantizer with the serial fallback. Icarus aborted while elaborating one wide concatenated constant tie-off in the serial bench. Individual tie-offs compiled cleanly.
- At full width, a read and write can be due on the same calendar tick. If only one SRAM port accepts, the machine remembers that transfer and freezes the calendar until the other port accepts. This keeps backpressure from duplicating a transaction or overwriting the early read payload.
- With the softmax tile overlapped and ordinary row streaming modeled, the midpoint chunk spends 21.47 ms in serial RSUMSQ and 9.94 ms in SMTILE. Four reduction contexts would reduce that RSUMSQ time to 5.21 ms in the ready-port timing model. The 32K projection then falls to 11.38 s.

## Streaming rows and interleaving reductions (note 32, part 7)

- **Tag the beat, not the controller.** Row streaming worked once every beat carried its own (plane, row, column) tag through the same delay lines as its data. The writer never asks the reader which row it is on, because by the time a SiLU result emerges 112 clocks later, the reader is several rows ahead.
- **A value computed for a row can outrun that row's writes.** QKV computes each row's fp16 scale after a scan pass, and the next row's scan finishes while this row's codes are still in the bank. A single scale register would have stamped row r's last beats with row r + 1's scale. The scale now rides in the tag too.
- **A scheduled shortcut is still bound by the ISA.** MATCH streamed rows as happily as the others. FSim defines MATCH on one row, though, so the RTL would have written extra results the reference never makes. It is pinned to one row.
- **The 9-clock calendar.** An 8-stage FMA means a reduction row can issue only every 9th clock. Four rows on slots 0–3 of that calendar keep the arithmetic order bit for bit and cut 16 rows of 4,096 values from about 2,500 clocks to 620. It costs four 2 KiB accumulators. Two contexts would miss the 32K prefill target (12.7 s), and eight would buy 0.7 s for another 8 KiB.
- **The last second was hiding in a 16-clock command.** After everything big was streamed, prefill sat at 12.8 s. It wasn't the softmax tile or the reductions. The program issues 14,016 QKV commands per chunk to quantize K and V heads into the cache, and each one paid about 11 clocks of launch and drain per row. Streaming QKV brought a 32K prompt to 11.4 s and a 20K prompt to 6.5 s, both inside D-008.
- **Testing ran at Icarus speed.** Each decoded check against the functional simulator took 12–17 minutes of compile and simulation, so new cases went in small, focused groups (`--rows-only`, `--red-only`) rather than into the full 40-minute suite.

## Blockers and open threads (updated 2026-11-22)

- **The vector unit now meets D-008 in simulation.** Prefill is 11.4 s for 32K and 6.5 s for 20K; decode is 90.0 tok/s, or 296 with three accepted drafts.
- **Settled:** partial overlap between a command's inputs and outputs is illegal; outputs are exactly in place or disjoint (D-027).
- **The whole chip is in RTL and matches FSim byte for byte** on the real decode program (1 and 3 tokens) and the real prefill chunk, on a 4-layer Qwen3.5-shaped model (note 33).
- **Open:**
  - the vector unit runs prefill at half its architectural rate (11.4 s vs 5.7 s for 32K); R1, QUANT width and SMTILE are the levers
  - the banked SRAM arbiter isn't written (conflicts measured at under 2% on decode)
  - D-029's matrix-unit buffers add about 1.2 mm² (without them, 32K prefill would take 32.6 s instead of 11.7)
  - prefill SRAM bank demand is 2.6× what 64 one-line banks serve when every unit runs at once; the arbiter decides what that costs
- **Next:** synthesis of the whole units for area and 1 GHz timing.

## The whole chip in RTL (note 33)

- **The descriptors had no bits yet.** ISA v1 defined descriptors as Python dataclasses, and the simulators read the objects directly. Building the hardware meant fixing a 64-byte encoding: DMA 272 bits, MM 292, Vec 495. The vector descriptor only fits because the op-specific parameters share one 96-bit union: no command uses both DeltaNet's head counts and the softmax tile's offsets.
- **A default is part of the contract.** FSim treats a missing CAUSAL step as 1. In a binary union, missing means 0. CAUSAL got a presence bit.
- **A register's meaning depends on the opcode.** r0–r3 normally offset the four bases. CAUSAL's r2 and r3 are a position and a limit; DeltaNet's r1 is a head number, and its token rows are absolute addresses. The decoder copies FSim's exceptions one by one.
- **The first whole-chip mismatch was the test's fault.** The smoke program's X wrote DRAM inside the matrix that W was streaming into the weight FIFO. FSim reads all of W's bytes the moment it starts; the RTL streams them over thousands of clocks. Five beats saw new bytes. Different DMA queues are not ordered against each other, so a program needs a semaphore between them, exactly as the ISA says.
- **The simulator was the bottleneck, and the fix was two environment settings.** Verilator couldn't link on this Mac: the linker can't read the newest SDK's library stubs (an unknown arm64e.x1 architecture), and Homebrew's binutils `ar` writes archives Apple's linker rejects. Building against the 26.5 SDK with Apple's `ar` fixed both.
- **Sizing the SRAM from measurements.** Prefill averages 1,766 bytes per clock across all units against 4 KB of bank bandwidth. The vector commands that want 4–6 KB per clock are already held back by their own arithmetic, so capping them at 4 KB, or even 3, changes prefill by 0.02 ms per 44 ms chunk. The ISA's 64 banks stay.
- **The DMA engine was 6× too slow, and nothing was wrong with it.** It moved 64 bytes per clock, one SRAM line, while DRAM delivers 512. The tiny decode took 292,501 clocks against the model's 48,045. Widening each engine to 512 bytes per clock and timing the testbench's DRAM like the model's (one shared 471-byte-per-clock server, 150 clocks of latency) brought it to 57,443.
- **The timing model had a hidden assumption: the matrix unit's bookkeeping is free.** It charges a matrix command its beats only. The RTL also loaded each chunk's activations into the stationary buffer and drained results through a 16-output epilogue between beats. Decode has 1–4 rows, so this cost 2%. Prefill has 128 rows: a 16K chunk would take 116 ms instead of 22.4. The fix is the standard one: double-buffer everything so loading, computing and draining overlap (D-029), plus an epilogue wide enough, 256 outputs per clock, that the attention blocks stream. That brings it to 23.9 ms.
- **"Needs 49,152 accumulators" was really "needs an output buffer".** Prefill's q·K and P·V blocks are one 256-input segment. Each output is final after one beat, so nothing accumulates. They only looked big because the first RTL held every output until the end of the command. Streaming them in 16-output tiles leaves 8,192 words per tile: 64 outputs × 128 rows.
- **The vector unit's pipeline is long, and the model charges 8 clocks for it.** SFU commands and reductions take 110–145 clocks beyond their beats in the RTL. Decode barely notices (+0.3%).
- **A 3.6× miss that was already in the books.** The prefill chunk at the chip's real 64 rows per clock took 867,985 clocks against the plain model's 238,399. Almost all of it was the vector unit, and it was exactly what the RTL-rate model from the vector-unit work predicted. That model charges each command what the RTL measurably does: 4 reduction rows per 99 clocks, row ops at 128 lanes. Stacking it with the matrix unit's new costs predicts 789,565, within 10% of the RTL. The lesson for the blog: keep two timing models, the architectural one for "what could this chip do" and a measured-rate one for "what does this RTL do". Only compare the RTL against the second.
- **"Done" had two meanings.** The ISA says a DMA signals when its last byte lands. The first RTL also took that as the moment the engine could start the next copy. The timing model had quietly assumed the next copy starts requesting the moment the last request leaves. Between those two moments sits a whole DRAM latency, so the weight stream lost 150 clocks per matrix and ran at 87% speed. Splitting "free to issue" from "done" with a last-piece bit on each request brought the tiny decode from 57,314 to 50,705 clocks, with the same semaphore order.
- **A transpose is a million tiny copies.** Writing V transposed for attention is a DMA of 1-byte pieces, and the DMA moved one piece per clock: 32,769 clocks where the model charged 528. A 64 × 64-byte tile buffer in the SRAM-to-SRAM engine reads rows and writes columns at 32 bytes per clock. It took 4 KB of flip-flops and one test with ragged edges.
- **Python's `id()` lied in my own measuring script.** It marked commands already charged by their `id()`. The simulator frees command objects, and new ones reuse the same addresses, so most commands were skipped and the overhead looked 400× too small. Holding a reference to the command fixed it.
- **Small tool traps:** `packed` is a SystemVerilog keyword, so a wire can't be called that. Icarus crashes (an internal assertion) on a variable index into a two-dimensional array inside a continuous assign; flattening the array to one dimension avoids it.

## Prefill at full width, and the first 1 GHz pass (note 32, part 8)

- **The three named levers weren't the whole gap.** Reductions, quantizer width and the softmax tile at their ideal still left 8.3 s against 5.7. The other half was short rows: a per-head row is 64–256 values, and a 512-lane beat carried one. Packing several rows into a beat took 32K prefill from 11.4 to 9.8 s on its own.
- **A reduction tree fits in one clock of the bank it already has.** The pairwise tree's levels use 256 + 128 + ... + 1 = 511 lanes, one fewer than the bank has. So one clock can do level 1 of the newest row, level 2 of the row before, and so on. Alternating that clock with a read clock gives full-rate reductions with no new adders, and the unit never reads and writes in the same clock.
- **Skipping the levels that add zero is exact, but only for a reason.** A packed 128-value row is done after 7 levels; the golden order then adds +0 twice. That's the identity only because no sum here is ever −0 (every lane starts at +0) or subnormal (every value is an FMA result, already flushed). Worth writing down, because it's the kind of argument that breaks when a format changes.
- **A row every 3 clocks, from residues.** The softmax tile's read port needs 3 transfers per row (state, two score beats) and its write port 3 (two code beats, state), so 3 is the floor at 256 positions. The calendar is a puzzle in time mod 3. The alpha subtraction rides lane 511 on the one residue the scaling leaves free. The final FMA shares lane 383 with the tree's last level on another residue.
- **Synthesis numbers lie without buffering.** The first survey said the FMA was 2.7 ns. The real path was one clock-enable net driving 1,103 flops through a single gate. With buffering it's 478 ps. In silicon a clock enable is a clock gate, not a wire to every flop.
- **`N / 3` is a divider.** The softplus and rsqrt glue divided a 50-bit number by 3, and synthesis built a full divider, about 1 ns alone. N is a 24-bit mantissa times 2^26, and 2^26 is 1 mod 3. So it's one constant multiply plus a constant per remainder, checked over all 2^24 mantissas.
- **Timing fixes cost latency, and the model has to hear about it.** Every register added for 1 GHz moved a function's latency: rsqrt 40 → 48 clocks, sigmoid 104 → 118, the quantizer +3. That moved the tile's write slots and the row path's delay lines. The 7.2 s projection includes all of it.

### Blockers and open threads (added 2026-11-22, evening)

- **The SRAM arbiter deadlocks on prefill.** All three bank-count runs (64, 128, 64 with skew) stalled at bundle 697 from about clock 6,000 and sat there for 2 hours. Decode and the fuzz passed with the arbiter on. This is the first thing to look at when the SRAM work resumes; the bank count can't be decided until prefill runs.
- **The matrix unit computed a whole beat in one clock** in `chip_mxu`: weight decode, 1,024 multiplies, the reducer, an fp32 add, and two chained FMAs in the drain. The standalone group was already pipelined (624 ps); the whole-chip unit wasn't. It's being pipelined now: a 7-stage row (bit-exact on 2,291 random beats), a pipelined accumulate, and a pipelined drain and argmax.

### The matrix unit pipelined, and QUANT / DEQ / QKV packed (2026-11-22, late)

- **Pipelining broke the end of a command, not the arithmetic.** The pipelined matrix unit failed its fuzz on all 3 seeds, wrong from the first output byte. The 8-stage row was already bit-exact on its own. The bug was in the controller. After the last tile, the unit waits 17 clocks (DWAIT) for the drain pipeline to empty. Two things were only allowed in RUN: reading dst for the acc epilogue, and updating argmax's running best. A small command's whole drain fits in those 17 clocks, so both were skipped. Plain f32 writes were already allowed in DWAIT, which is why some outputs came out right. Two lines fixed it; all 3 seeds now match every byte.
- **The lesson:** around a deep pipeline, the "wait for it to empty" state has to allow everything still in flight. Making the unit faster moved work later in time, past the point where the controller stopped allowing it.
- **Widening QUANT to 512 lanes was worth 0.1 s, not the 0.5 s expected.** Prefill's QUANT, QKV and DEQ rows are mostly 64 or 128 values, so a 512-lane beat still carried one row and most lanes idled. Measured over the whole prefill chunk, the softmax tile at 256 positions and the short-row QKV and QUANT were the biggest excesses over the model: 5.4M, 4.4M and 3.5M clocks.
- **528 lanes into 512.** A full QUANT beat makes 512 codes and 16 group exponents, one more write lane per 32 than the port has. Codes are bytes, so 4 go in each 32-bit lane: 128 lanes for codes, 16 for exponents. This only works when the codes are stored contiguously, so the wide path checks the stride and otherwise falls back to 128 lanes.
- **Packing short rows works for QUANT because a group of 32 never straddles two rows.** Rows of 32 to 256 values are whole groups, so each slot's exponents are exactly its own row's. QUANT on 128 rows of 64 now takes 30 clocks instead of about 140.
- **QKV reads every beat twice.** Each row needs its maximum before anything can be quantized. Scans take even clocks and streams odd ones. A stream is exactly the scan from 5 clocks earlier, coming out of a delay line, so there's no second cursor to keep in sync. The row maximum is of |a|, and non-negative floats sort like their bit patterns as integers, so the max tree needs no float logic.
- **A 9 ns unit nobody had timed.** The QKV scale (`vec_kv_scale`) did two real divisions in one clock: the row max by 127, and 1 over the fp16 scale. It came out at 9,060 ps. The old 128-lane QKV path had it too; it just wasn't on the survey list.
- **Neither division needed a divider.** round(N / 127) is floor((N + 63) / 127), because 127 is odd so there are no ties. floor(X / 127) is (X × 0x81020409) >> 38, six shifted adds, checked over all 16.7M possible inputs. The reciprocal's input is an fp16 significand, only 1,024 values, so a table holds the exactly rounded answers. In 3 stages it's 652 ps and half the area (462 against 947 µm²), and it matched the old unit on 54.3M inputs with random stalls.
- **The pipelining was free on the wide path.** Its 3 stages fit inside the 5 clocks QKV already waits between scan and stream. On the 128-lane path it costs 3 clocks per row, which is why that fallback now projects worse (7.6 s) than before.
- **The packed QKV's first bug:** the stream reads took their row from the scan cursor, which had already moved 2–3 beats ahead. The fix reads the rows the scan tags carry out of the same delay line as the scales.
- **Where prefill stands:** 6.6 s projected for 32K, from 7.2. Measured: 16 packed beats take 27 clocks for DEQ, 30 for QUANT, 47 for QKV, against about 140 and 395 at 128 lanes. The softmax tile at 256 positions is the next biggest piece (it would make it 6.1 s), and its period-3 floor comes from the read and write ports.

### The SRAM, revisited (note 34; 2026-11-22, evening)

- **The prefill "deadlock" was a livelock in the arbiter model.** The vector unit's read and write were separate requesters, and each forgot its progress when it completed. The unit only moves when both are ready in one clock. A read and a write needing 2 clocks each, one clock apart, finish on alternate clocks forever. The fix makes them ready together and keeps a finished half's progress. It's applied, and a prefill with the arbiter on is running to confirm.
- **Packing moved the bottleneck to the SRAM.** D-028 kept 64 banks because the vector unit rarely asked for more than 4 KB per clock. A packed FMA beat asks for 8 KB.
- **Bandwidth is the smaller part.** Holding each vector command to 4 KB per clock costs 0.1–0.3 s of the 6.6 s. Bank collisions on the real addresses cost 1.2 s (7.8 s), with the FMA over 128 rows of 256 floats alone at 2.3M clocks per chunk.
- **Most collisions come from layout, not bank count.** Buffers used together start at the same offset in a 4 KB stripe, so they hit the same 32 banks. A column read at a 512-byte stride hits only 8 of 64 banks. Perfect placement with 64 banks reaches 6.85 s, better than 128 banks with today's layout (7.3 s).
- **My first bank model overcounted.** It charged each beat's write against its own reads. In the pipeline, the write belongs to a beat about 11 clocks earlier. Fixing that took the 64-bank figure from 8.4 to 7.8 s.
- **With the livelock fixed, prefill finished but came out wrong** (2.67M bytes), and decode had 18 wrong bytes. That decode passed with the arbiter on before today's changes. The 18 bytes led straight to the cause: two floats in 3 rows, top byte right and mantissa wrong. They came from an RMAX whose rows 2 and 3 were wrong, and those rows were the second beat of a packed CAUSAL mask.
- **The bug: a request that depended on its own answer.** The pass path (COPY, MAX, CAUSAL) asked for its next beat only if its write was ready. The arbiter decides at the clock's midpoint from the requests it sees, and a requester not asking is "ready" by default. So when the write turned ready, the read request appeared after the decision, met that default, and the unit took stale data as a real read. The core's code already had this rule written down ("the request does not depend on its answer"); the pass path broke it.
- **Why no test caught it:** the vector-unit tests drive ready from the clock count, never from the requests, so this can't happen there. The matrix fuzz runs one unit at a time, so it never overlaps units.
- **The fix, and a guard:** the pass path now always asks for its next beat and moves only when the read and the held write both go, the pattern the rest of the vector unit uses. The SRAM model now stops with an error if any request changes between the arbiter's decision and the clock edge, so this kind of bug fails loudly at once.

### The matrix unit's cost per command (2026-11-22, night)

- **The short-context prefill found a 3-second problem the model didn't have.** The tiny prefill took 408,084 clocks against the RTL-rate model's 336,842. The vector unit was within 6%. The matrix unit's 584 small attention multiplies (MMS) took 166 clocks each against 49. A 9B chunk has 105,216 of them, so 32K prefill would be about 9.7 s, not 6.6.
- **Where one command's 166 clocks went:** 28 loading activations, 49 of beats (all the model charges), 17 in the accumulate pipeline, 49 draining, 20 for the drain pipeline and the end wait.
- **The drain was supposed to hide behind the beats, and didn't.** These tiles are 16 outputs, 8 beats each, and the accumulator had 2 halves. Tile t+2 had to wait for tile t's 17-clock pipeline and 8-clock drain, so 2 tiles took about 25 clocks instead of 16. The accumulator holds 16K words and a small tile needs 2,048, so it now keeps as many tiles as fit: up to 8 slots. Bit-exact on the fuzz, all 3 seeds.
- **The rest is between commands.** The unit finished each command completely before taking the next. The fix is the DMA's lesson again: split "can take the next command" from "done". A front (loader and beats) takes the next command once the last beat issues, while a back (drain, end wait, argmax) finishes the old one. Each command keeps its own copy of what the back needs, and each beat carries its own accumulator address. Otherwise the last beats of one command land after the next has changed the registers they read.
- **Small layout fixes beat big hardware ones.** Staggering every buffer's start did almost nothing, because the colliding operands were inside one buffer or in place. The worst one was the attention accumulator: OA = OA × alpha + OT, where OA and OT both had 1 KB rows at the same position in the 4 KB stripe, and ping-pong slot 1 lined up exactly. Rounding the slab to whole stripes and starting OA 2 KB off OT costs 9 KB.
