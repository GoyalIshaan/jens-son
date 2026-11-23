# Handoff to the implementation agent (2026-11-21)

*Claude plans and reviews; the implementation agent builds and tests. Read `design/32_vector_unit_rtl.md` (all 4 parts) and `design/31_functional_simulator_first_run.md` first.*

## Ground rules

- **Golden v2 is the contract.** Every RTL result must match `golden/bitexact.py`, `bitexact_vec.py` and `chip_sfu.py` bit for bit. Changing operation order inside a rule (R1–R8) is a contract change: stop and say so, don't do it silently.
- **"Done" means verified and measured.**
  - verified: the bench passes, and the decoded command matches the functional simulator byte for byte (`rtl/vec/test_vec_unit.py`)
  - measured: cycles from the RTL bench, then 9B tok/s from `isa/bench_rtl_rates.py`
  - Never feed an RTL cycle count into the performance model without asking whether the controller is the bottleneck (the part 3 lesson).
- **Environment:**
  - Python with torch: `export PATH=/usr/local/bin:$PATH`
  - Icarus only: Verilator can't link on this Mac (SDK mismatch)
  - Long runs in the background, logs in `rtl/vec/results/logs/` (not /tmp; it was wiped once)
- **Notes:** plain bullets, small concrete numbers, no tables. Record each bug and intricacy in the note and in `blog/notes_part2_intricacies.md`. Don't edit `design/DECISIONS.md` or the memory files; list amendments in your report.
- **Decisions:** don't record anything as "approved by the user" unless the user said it to you directly. The 4-head DeltaNet approval needs confirming.

## Step 0: finish verifying what's already written (do first)

1. **Run `tb_vector_exec`** (64-lane bench; the tile was dropped from it because R7 needs 512 lanes). It must PASS.
2. **Run the full decoded test** `python3 rtl/vec/test_vec_unit.py` (about 40–60 min) on the current RTL: new DeltaNet controller, streaming box ops, the 128-wide row path, the streaming softmax tile. All 33 descriptors must match.
   - If the tile descriptors fail: the vec_unit wiring is new (128-position beats, write column `wcol`, 4 exponent sideband lanes at `base[0] + e_offset + row*e_stride + group`, no EXP_WRITE cycle). Check those first.
3. **Run the whole regression** `sh rtl/vec/run_tests.sh`. Fix compile lists if any bench misses `fp32_fma_pipe8_ce.v`.
4. **Re-run `isa/bench_rtl_rates.py`** with the tile's measured cost (199 clocks per 1,024-position row, rows not overlapped) in place of the old 4,070 formula. Expect decode near 88 tok/s; prefill still slow.

## Step 1: stream across rows (the biggest remaining prefill gap)

- **Problem:** `vec_unit` launches each row of a row command, reduction and tile separately and waits out the full pipeline latency (64–136 clocks for the special functions, about 200 for a tile row, about 90 for a reduction) before the next row.
- **Target:** issue row r + 1 while row r drains. Rows are independent in every rule.
  - **Row ops:** treat the (m, p) rows of one descriptor as one stream; per-row sidebands (QKV scale, QUANT exponents) ride with each beat.
  - **Reductions (R1):** per lane, the chain within a row is inherent; overlap different rows (e.g. 8 rows in flight to cover the FMA's 8 cycles).
  - **Softmax tile:** pass 1 of row r + 1 during pass 2 / the tree of row r. The 128-unit exp pool, the bank quarters and the write port are the shared resources.
- **Measure:** `bench_rtl_rates.py` says row streaming alone takes a 32K prompt from about 31 s to about 9 s; the tile overlapped is the rest. Target: within 1.5× of the model's 22.4 ms per prefill chunk.

## Step 2: pack short rows into one box beat

- **Problem:** the box walker puts one row per 512-lane beat. The lite solve (128-wide rows × 63), the DeltaNet norms and similar waste 3/4 or more of the lanes. About +4 ms per prefill chunk for FMA, +1.6 MUL, +1 COPY.
- **Fix:** when n < 512, map lane j to (column j mod n, row m + j / n), with masks; the wave schedule for multi-tap ops stays as it is.

## Step 3: smaller rate gaps

- **QUANT, DEQ and QKV at 512 wide** (now 128). QKV's two passes could share one read.
- **CONV4 with a saved window** (decode) on the streaming path: the window write needs a second write slot, or a bubble per beat.
- **COPY, MAX and CAUSAL** on the streaming path (pass-through through the FMA is exact for chip values; MAX and CAUSAL need a bypass lane in the delay line).

## Step 4: area and timing of the whole unit

- **Synthesize the full vector unit** on ASAP7 (`rtl/vec/synth_vec.py`): 512 lanes, 4 DeltaNet contexts, the row path's unit pools (128 exp + 64 sigmoid + 2×3), the tile (its own 128 exp + 1).
  - Report area and the slowest stage per block.
  - **Known:** the int8 quantizer's 2.9 ns path needs pipelining.
- **Area questions to answer:**
  - the tile and the row path each have 128 exp units: share one pool (they never run at once) and measure the saving
  - the DeltaNet contexts' registers (about 4 × 70 Kbit)

## Step 5: next big blocks (after the vector unit meets its rates)

- **The matrix unit around the existing lane group and beat reducer:**
  - the TAB header (16-entry table + fp32 tscale)
  - E4M3 decode, the int8 path and segment-major weights
  - the epilogues: tscale, accumulate, argmax with ties going to the lowest index
  - Check against `golden/vectors/mm_beats.hex` and decoded MM/MMS commands in the functional simulator.
- **Then:** DMA engines, the weight FIFO, SRAM banks, then the scalar core (issue, 4 queues, semaphores, loops, branches).

## What Claude will do (thinking side)

- Review each step's design before it's built and each result after.
- Decide the cross-row streaming structure for the tile (shared resources, ordering).
- Re-derive performance after each step; keep the notes and the blog collection honest.
- Open contract questions:
  - a fixed-point special-function contract (smaller units; needs golden changes and a quality rerun)
  - the merged in-projection's quantization on the 9B (Beam)
