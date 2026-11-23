# 33. The whole chip in RTL: core, queues, DMA, matrix unit, SRAM (2026-11-22)

*Progression step 6, second half. The vector unit was note 32. This note puts every other ISA v1 block in RTL (`rtl/chip/`), links them into one top level, and checks the whole chip against the functional simulator, byte for byte, on real programs.*

## Part 1. What was built

- **`chip_core.sv`, the scalar core.**
  - One 256-bit bundle per clock, in order: S, D, M, V slots.
  - 32 registers of 48 bits; all scalar arithmetic is unsigned 48-bit, as in `isa/sim_v1.py`.
  - The S slot: ADD … LI, LW/SW (32-bit SRAM words), BEQ/BNE/BLT/J, LOOP, WAITS/SIGS, HALT.
  - **Hardware loops:** a 4-entry stack. At the end of a body the innermost loop branches back while iterations remain, and several loops may end on one bundle. This is the simulator's rule, written as a combinational check from the innermost entry outward.
  - **Six command queues,** 16 deep: W, R, X, L, M, V. An entry holds the opcode, the descriptor index, the five register values read at issue, and the two waits and two signals.
  - **Issue stalls** on a full queue, or on WAITS when its semaphore is 0.
  - **63 counting semaphores (16 bits).** A queue's head starts when its unit is idle and its semaphores are positive (2 if it waits twice on one).
    - Several starts in one clock are granted in unit order, against the counts the earlier grants left. The scalar WAITS goes last. So two units can never both take the last count.
    - A unit's done increments its command's signal semaphores.
  - **A slot is present when its word is nonzero.** An M or V NOP that waits or signals is a fence; an all-zero word is an empty slot.
- **`desc_v1.py` and `desc_fields.vh`, the binary descriptor format.** ISA v1 had defined descriptors only as Python dataclasses. Now each one is a 64-byte word, and the encoder generates the RTL's field offsets, so there is one definition.
  - DMA: 272 bits. MM: 292 bits. Vec: 495 bits.
  - Vec's op-specific parameters share one 96-bit union: the QUANT exponent bias; ROPE's rot; CAUSAL's step (with a presence bit, since FSim's default step is 1); the CONV4 window; DN's T, commit, work, heads, lv, lqk; SMTILE's vs, e and mlb.
  - **DN stores log2(lv / lqk),** so the key head is a shift. The encoder rejects a ratio that isn't a power of two.
- **`chip_dma.sv`, one engine per DMA queue.** It does the 3-D copy (n0 bytes × n1 × n2, signed strides) in pieces of up to 512 bytes (PB), never splitting a piece across a row.
  - 512 bytes per clock is DRAM's whole bandwidth at 1 GHz (512 GB/s), so one engine can use all of it. The first version moved 64 bytes per clock and made the tiny decode 6× slower than the model (292,501 clocks vs 48,045).
  - **DRAM reads** are a request plus an in-order response, with up to 256 pieces in flight. Each piece remembers its destination, so any DRAM latency works.
  - The stride products are signed 58-bit, so a negative stride walks backwards correctly.
  - **W** requests only while the FIFO has room for everything already in flight.
  - **Done** fires at the ISA's signal points: R, L and W when the last byte has landed, X when DRAM has accepted the last write.
  - S2H goes out a host port. H2S faults: there is no host-input model yet, and FSim doesn't implement it either.
- **`chip_wfifo.sv`, the 192 KB weight FIFO.** The matrix unit sees its first 1,024 bytes and pops 32, 576 or 1,024.
- **`chip_mxu.sv`, the matrix unit's command controller** around the verified datapath (`rtl/mxu`: lane products, the 16-input adder tree, the 4-segment beat reducer).
  - **TAB, MM and MMS:**
    - lut4 codes looked up in the TAB table, with E4M3 group scales
    - raw int8 weights (scale 1)
    - seg 4, which broadcasts a 256-input activation segment to the four quarters
  - **Accumulation:** `acc = fp32(acc + v)` per output in input order, then `y = fp32(acc × tscale)`. The f32 epilogue writes y; acc writes `fp32(dst + y)`; argmax keeps a running best per row, ties to the lowest index.
  - **The order of work is the order of the weight stream:**
    - lut4: tile of 64 outputs, then 1,024-input chunk, then output
    - int8: segment, then beat of 1 or 4 outputs
  - **Loading, beats and the epilogue overlap (D-029, part 5):** double activation buffers of 128 rows, double accumulator halves, and a 256-output-per-clock drain.
  - **RP rows are computed per clock.** More rows hold each beat ceil(rows / RP) clocks. The chip has RP = 64; simulation uses RP = 1, which gives identical results, since rows are independent.
- **`chip_vunit.sv`, the V queue's unit.** It decodes the Vec descriptor plus the registers into `vec_unit`'s ports.
  - It reproduces FSim's register meanings exactly. r0–r3 offset the bases, except: CAUSAL's r2/r3 are position and limit; DN's a and b are absolute and r1 is the first head. r4 overrides n, m or p.
- **`chip_sram.sv`, the SRAM's port model.** It is behavioural, with one port per client: V (3 × 512 gathers, 512 scatters), M (since D-029: two 1,024-byte reads, one 64-byte read, 16 line reads and 16 line writes), R, X, L (512 bytes each) and S (words).
  - It also counts, every clock, the distinct 64-byte lines each of D-028's 64 banks would have to serve (part 3).
  - Within a clock every port reads memory as it was before that clock's writes: reads are sampled at the falling edge, writes land at the rising edge.
  - This is the contract the banked SRAM (part 3) has to keep.
- **`chip_top.sv`** links them all. DRAM sits outside as three ports (W reads, R reads, X writes).
- **`tb_chip.sv` and `run_chip.py`, the whole-chip test.**
  - The testbench's DRAM is timed the way `isa/sim_v1.py` times it: one shared server of 471 bytes per clock (512 GB/s × 0.92), plain reads before weight reads before writes, writes costing 1.15× their bytes, data arriving 150 clocks after a read is served.
  - It writes a trace of every command's start and done clock (`trace.txt`); `compare_timing.py` lines it up against the timing simulator, unit by unit.
  - `SIM=verilator` builds with Verilator (`build_verilator.sh`), about 60× faster than Icarus. The smoke program takes 6 s instead of 380 s.
  - `run_chip.py` builds the program and its images, runs FSim, runs the RTL, and compares every byte of SRAM and DRAM.

## Part 2. Verified

- **The smoke program, `python3 rtl/chip/run_chip.py smoke`: all 1,048,576 SRAM bytes and all 1,048,576 DRAM bytes match FSim.**
  - It covers:
    - a hardware loop and scalar arithmetic, SW, then LW after a WAITS, with a negative immediate
    - DMA R, a strided 3-D L, X, and W three times
    - TAB, then a lut4 MM over 3 rows with the f32 epilogue
    - a second TAB and a lut4 MM with argmax plus an id base
    - MMS in int8 seg 4 over two segments, with tscale 0.5
    - an int8 seg-1 MM from the FIFO, with tscale 2
    - V ADD and RSUMSQ
    - semaphores fanning out (one V signal releasing X and L) and joining (three WAITS)
  - 29 bundles issued (25 in the program, plus the loop's repeats). It took 2,566 clocks at first, 1,178 with 512-byte DMA, and 1,108 with D-029's matrix unit.
  - Logs: `rtl/chip/results/smoke.log`, `smoke_vl.log`, `smoke_d029.log`.
- **The real decode program, `SIM=verilator python3 rtl/chip/run_chip.py tiny [tokens]`.**
  - It runs a 4-layer Qwen3.5-shaped model: hidden 1,024, FFN 1,024, vocab 1,024, 3 DeltaNet layers and 1 attention layer, random weights quantized as the 9B's are. The whole token loop runs, through the LM head's argmax.
  - **1 token: all 589,824 SRAM bytes and 20,915,904 DRAM bytes match FSim.** 371 bundles, 57,443 clocks.
  - **3 tokens: all bytes match.** 1,069 bundles, 172,283 clocks, 399 s of simulation.
- **The real prefill program, `tinyprefill`: one 128-token chunk through the same model. All 3,801,088 SRAM bytes and 20,303,424 DRAM bytes match.**
  - 14,015 bundles; 2,232,728 clocks at 8 rows per clock with the serial matrix unit; 75 minutes of simulation.
  - Running it needed 128 rows in the matrix unit, which is how the accumulator size came up (part 5).
- **After D-029's matrix unit (part 5), everything was rerun:**
  - **smoke:** passes, 1,108 clocks
  - **tiny 1 token:** passes, 57,314 clocks
  - **`mmfuzz`, seeds 1–10: every byte matches.**
    - Each seed is 14 random matrix commands: lut4 MM with f32, acc and argmax; int8 MM from the FIFO; MMS in seg 4 (packed and not) and seg 1.
    - Rows are drawn from 1, 3, 17, 64, 65, 100 and 128, at 8 rows per clock, so the row groups are partial.
    - Shapes include 1–3 chunks, 1–3 tiles, and output counts that aren't a multiple of 16.
- **The first run differed in 59 bytes, and the bug was in the test.** The program's X wrote DRAM 0x10000–0x10800, inside the lut4 matrix that W was still streaming into the FIFO.
  - FSim reads all of W's bytes at its start. The RTL streams them, so 5 beats (outputs 106–110) saw the new bytes.
  - This is the cross-queue version of D-027's rule: DRAM accesses from different queues aren't ordered, so a program needs a semaphore between them. Moving X elsewhere made every byte match.
- **D-027 lint, `python3 isa/alias_lint.py`:** the 9B decode, speculative and prefill programs have no partial overlaps between a vector command's outputs and inputs. A self-test with a shifted in-place ADD is caught.

## Part 3. The SRAM: banking, measured before choosing

- **What the programs ask of SRAM** (`python3 isa/sram_traffic.py`, 9B, 1 GHz):
  - **Decode at 32K:** 50 bytes per clock on average. The vector unit moves 270 MB per token, the matrix unit 161 MB, DMA 116 MB.
  - **A prefill chunk at 16K:** 1,766 bytes per clock on average (matrix unit 962, vector unit 786).
    - The vector unit spends 1.2M of the chunk's 22.4M clocks in commands that want 4–6 KB per clock: ROPE, MUL, FMA, QUANT, DEQ.
    - That's two 2 KB operand reads and a 2 KB write per 512-lane beat.
    - The matrix unit wants 1–2 KB per clock for 15.6M clocks, mostly the activation reload at every 64-output tile.
- **Capping the vector unit's SRAM bandwidth in the 9B timing run** (RTL rates as built):
  - no cap, 8 KB, 4 KB per clock: 32K prefill 11.4 s, 44.46 ms per chunk
  - 3 KB per clock (leaving 1 KB for the matrix unit): 44.48 ms per chunk
  - Decode is unchanged in every case.
  - The heavy vector commands are already held back by their own arithmetic (the 128-wide row path, the tile), not by SRAM.
- **Decision (D-028): keep ISA v1's 64 banks × 64-byte lines, 4 KB per clock.**
  - Each bank is a single-ported 104 KB macro (1,664 lines), low-order interleaved: bank = (address ÷ 64) mod 64.
  - A contiguous 2 KB vector operand covers 32 banks in one clock.
  - 128 banks would buy nothing measurable at the RTL's rates and cost about twice the bank periphery.
  - Revisit if the vector unit reaches the model's architectural rates.
- **The banked design that replaces the port model:**
  - **Clients:**
    - the vector unit's three read streams and one write stream (up to 32 lines each per clock, or 33 when unaligned; a strided operand up to 512 lines)
    - the matrix unit (16 lines read, 1 written)
    - DMA R, X and L (1 line each)
    - the scalar core (1 word)
  - **Coalescing:** lanes that hit the same line share one access.
  - **Each bank grants one line per clock,** with rotating priority among the clients asking for it.
  - **A client's beat completes when all its lines are granted.** Lines granted early wait in that client's line buffer. Clients see this as ready = 0, and the vector unit already freezes on it (note 32).
  - **The DeltaNet state** keeps its own banks (D-023). Four 64 KiB head states read 2 KB per clock without touching the shared banks.
  - **Read-before-write within a clock** has to stay true: a bank never reorders a read behind a later write to the same line.

## Part 4. What is behavioural, and what that means

- **Behavioural, by design for now:**
  - the SRAM (part 3 is the plan)
  - the weight FIFO's storage
  - the descriptor and instruction memories (loaded by the testbench)
  - DRAM, in the testbench
- **Everything that decides what happens when is real RTL:**
  - issue, queues, semaphores, loops and branches
  - DMA sequencing and its latency handling
  - the matrix unit's command sequencing, accumulation and epilogues
  - the whole vector unit
- **Simplifications to revisit for timing:**
  - The matrix unit accumulates in one combinational step per beat. The chip needs a pipelined adder with the same order. That's safe: for lut4, an output's next update is 64 beats away, but int8 seg 4 with few outputs updates the same output every beat.
  - The 2.9 ns activation quantizer (note 32) still needs pipelining to 1 GHz.

## Part 5. Timing against the model, and the matrix unit's overlap (D-029)

- **How the RTL's clocks compare** (`compare_timing.py` lines up the RTL's per-command trace with the timing simulator):
  - **Tiny decode, 1 token: 57,443 RTL clocks vs 48,045 in the model (1.20×); 3 tokens: 172,283 vs 144,094 (1.20×).** The gap, by unit:
    - the matrix unit, about 2,900 clocks: activation loads and the epilogue, which the model doesn't charge
    - the vector unit, about 6,500: SFU commands and reductions take 110–145 clocks each beyond their beats (pipeline depth), where the model charges 8
    - TAB waiting for the weight FIFO, about 900
  - **Tiny prefill, one 128-token chunk, at 8 rows per clock: 2,232,728 clocks vs 238,399.** The model assumes 64 rows per clock, and the first RTL loaded and drained serially.
- **What those two costs do to the 9B** (`python3 isa/mxu_overhead.py`, added to every matrix command of the real programs):
  - decode at 32K: 11.005 ms in the model; 11.233 with the serial matrix unit; 11.005 with any overlapped design
  - **prefill chunk at 16K: 22.395 ms in the model, 115.97 ms with the serial matrix unit (5.2×)**
  - overlapped, by epilogue width: 16 outputs per clock 77.36 ms; 64: 36.33; 128: 29.49; 256: 26.44
  - plus packed seg-4 loads: 128 per clock 26.55 ms; **256 per clock 23.92 ms (+6.8%)**
  - why prefill is hit so hard: its q·K and P·V blocks are 128 rows × 256 outputs from one 256-input segment, 128 clocks of beats each. A 16-output epilogue needs 2,048 clocks to write their 32,768 results, and there are 16,640 of them per chunk.
- **The vector unit's pipeline latency, measured the same way** (RTL latencies per opcode added to the model): decode 11.035 ms (+0.3%), prefill chunk 24.21 ms (+8.1%). That counted only latency beyond the beats. The full vector RTL rates are in `bench_rtl_rates.py`, and they cost much more on prefill (see "The prefill at 64 rows per clock" below).
- **D-029, the matrix unit as rebuilt (`chip_mxu.sv`).** Three engines run at once, so in steady state a command costs only its beats:
  - **Loader.** Two activation buffers of 128 rows. The next (tile, chunk) loads while the beats read the other buffer.
    - one row per clock: 1,024 bytes on port A, its exponents on port E
    - four rows per clock for seg 4 when rows are 256 bytes apart and their exponents fit one 64-byte read
    - a one-chunk command loads once for all its tiles
  - **Beats.** The datapath is unchanged. Up to 128 rows, held 2 clocks per beat at 64 rows per clock.
  - **Drain.** Two accumulator halves of 8,192 words; a finished tile drains while the next tile fills the other half.
    - 16 rows × 16 outputs per clock, one 64-byte line per row, on 16 write ports (and 16 read ports for the acc epilogue's dst)
    - Tiles: lut4 has 64 outputs per tile; one-segment int8 has 16 (so q·K and P·V stream); multi-segment int8 has one tile of all outputs.
  - **Order is unchanged,** so results are bit-identical:
    - each output still accumulates in input order
    - argmax still sees each row's outputs in increasing order, ties to the lowest
    - the weight stream is consumed in the same order
  - **Cost:** about 1.2 mm² of ASAP7 cells: activation buffers 0.63, accumulators 0.15, 512 epilogue FMAs 0.45. The fallback is 8 rows per drain clock: half the epilogue, 26.55 ms per chunk.
- **ACC_WORDS was wrong in ISA v1.** It said 4,096 (64 outputs × 64 rows), but prefill runs 128-row matrix commands. Now it's 8,192 per half.
- **SRAM bank conflicts, measured** (the port model counts, every clock, the lines each of the 64 banks would serve):
  - tiny decode: 16,946 clocks with SRAM traffic, 236,643 lines; one line per bank per clock would take 18,017 clocks
  - so conflicts can add at most 1,071 clocks (1.9% of the run), and less once requests queue instead of stalling everything

## Part 6. The chip's infrastructure in RTL (2026-11-22, night)

Note 09 listed what we write besides the compute units. Before the first full profile, these replace the testbench shortcuts. Written and linted module by module; **not yet built or run** (the machine was overloaded, and runs are now one at a time).

- **The weight FIFO** (`chip_wfifo.sv`) is now a ring of 64-byte lines that maps onto 32 two-port banks, line l in bank l mod 32.
  - The 1,024-byte front window spans at most 17 lines and a 512-byte write at most 9, so neither ever needs a bank twice.
  - The window is a register: each clock reads the lines at the read pointer after this clock's pop. A line written in the same clock is bypassed from the write data.
  - It behaves the same as the behavioural ring: the window seen in a clock is the ring after the previous edge.
- **The host register bus, register map, reset sequencer and boot** (`chip_ctrl.sv`):
  - A 32-bit bus, one access per clock, reads answered the next clock.
  - Registers: ID, version, control (run, soft reset), status (halted, idle, fault, which unit faulted), the program counter, cycles, bundles.
  - A load window for the instruction memory (256-bit words, 8 pieces) and the descriptor memory (512-bit, 16 pieces). Scalar register writes.
  - The host input FIFO and 24 performance counters.
  - The reset sequencer holds the chip in reset for 16 clocks after the external reset or a soft reset. Boot: load the program and registers over the bus, then set CONTROL.run.
- **Loading ports on the core** (`chip_core.sv`): instruction memory, descriptor memory and register writes, accepted only while the core is stopped. The core also now reports its in-flight queues and why issue stalled.
- **Host input (H2S)** in DMA L (`chip_dma.sv`): the source is the host input FIFO. A piece moves when its bytes are there, and its SRAM write request waits for them. That isn't an SRAM ready, so it doesn't break the arbiter rule from note 34.
- **Performance counters** (`chip_perf.sv`, 24, 48-bit):
  - clocks running
  - per queue, clocks with a command in flight and commands finished
  - issue stalls on semaphores, full queues and the SRAM
  - SRAM waits for the vector read and write, the matrix unit, X and L
  - clocks the matrix unit is busy with less than one beat of weights
  - DRAM bytes read and written
- **The testbench** waits for the reset sequencer. With `+boot` (`BOOT=1` in `run_chip.py`), it loads everything over the bus as a host would. It prints the counters at the end, read over the bus.
- **Still behavioural:** the SRAM banks (a cycle model of the arbiter over a flat array; the synthesizable banks, crossbar and the DeltaNet state's own banks come with the tapeout phase), DRAM, and the IP blocks (PHYs, PLLs).
- **Not exercised by any program yet:** H2S. Prompts still come preloaded in SRAM.
