# 34. The SRAM: what to change and what to leave (a proposal)

Status: **proposal, not decided.** It revisits D-028 (64 banks × 64-byte lines), as D-028 asked, now that the vector unit runs near its model rates. Scripts: `isa/sram_cap.py` (bandwidth only) and `isa/bank_conflicts.py` (bank collisions from real addresses). Both use the as-built RTL rates (`isa/bench_rtl_rates.py`), 9B, a 16K prefill chunk, 1 GHz.

## Part 1. Where it stands

- **The SRAM:** 6.5 MB in 64 single-ported banks of 104 KB. One 64-byte line per bank per clock, so 4 KB per clock in all.
- **Where a line lives:** bank = (address ÷ 64) mod 64. So any 4 KB of contiguous bytes covers all 64 banks once.
- **Who uses it:** the vector unit (three 2 KB reads and one 2 KB write per 512-lane beat), the matrix unit (1–2 KB per clock), DMA, the scalar core.
- **In RTL it's still a behavioural model,** plus a model of the arbiter (`rtl/chip/chip_sram.sv`, ARB = 1) that decides which request each bank serves.

## Part 2. What changed since D-028

- D-028 kept 64 banks because the vector unit was slow for other reasons: its beats were 128 lanes wide, and short rows wasted most of a beat. A beat rarely needed more than 4 KB per clock.
- That's no longer true. Packing (note 32, part 8, and the QUANT/QKV/DEQ work) puts several short rows in one beat, one beat per clock.
- A full FMA beat moves 8 KB: three 2 KB reads and a 2 KB write. That's twice what 64 banks can serve in a clock.

## Part 3. A bug: the arbiter livelocks

- **What was seen:** with the arbiter on, prefill stopped at bundle 697 around clock 6,000 and never moved again (three runs, 2 hours each). Decode and the matrix fuzz were fine.
- **Why:** the vector unit's read and write are two separate requesters in the arbiter. Each forgets its progress the moment it completes. But the vector unit only moves when both are ready in the same clock.
- **A small example:**
  - The read needs 2 clocks of bank time and so does the write, and they started one clock apart.
  - Clock 1: the read completes. The write doesn't, so the unit waits, and the read forgets it was served.
  - Clock 2: the write completes, but the read has started over. Wait again; the write forgets.
  - Clock 3 is clock 1 again. It never ends.
- **The fix:** the vector unit's read and write become ready together. A half that's already been served keeps its progress until the other half is done. It's a few lines in `chip_sram.sv`, written and waiting for the current runs to finish.

## Part 4. How much the SRAM costs prefill

Projected 32K prefill with the vector unit as built: **6.6 s if the SRAM cost nothing.**

- **Bandwidth alone** (each vector command takes at least its bytes ÷ bytes per clock):
  - 4 KB per clock: 6.7 s. 4 KB at 80% usable: 6.9 s.
  - 8 KB per clock: 6.6 s.
  - If the matrix unit takes 1–2 KB per clock at the same time, leaving the vector unit 3 or 2 KB (at 80%): 7.2 or 7.9 s.
  - Decode stays at 90.3 tok/s in every case: it moves about 50 bytes per clock.
- **Bank collisions** (each beat's real addresses; a bank serves one line per clock; a beat's reads share the clock with the write of the beat its pipeline latency earlier). These four layouts are compared:

  - 64 banks, plain (today): 7.8 s (2.4 bank clocks per beat)
  - 64 banks, skewed: 7.4 s (2.0)
  - 64 banks, perfect placement: 6.85 s (1.4)
  - 128 banks, plain: 7.3 s (1.8)
  - 128 banks, skewed: 7.1 s (1.6)
  - 128 banks, perfect placement: 6.6 s (1.0)
  - "Skewed" is `chip_sram.sv`'s SKEW = 1: bank = (line + line ÷ 64 + line ÷ 4096) mod 64. Each 4 KB block is rotated by its block number.
  - "Perfect placement" is the best any layout could do: each beat's lines spread evenly, so lines ÷ banks clocks.
- **What it says:** today's SRAM would cost about 1.2 s of the 6.6. Most of that is where data sits, not how many banks there are. Even 128 banks leave 0.5–0.7 s on the table unless layout improves too.

## Part 5. Where the collisions come from

- **Buffers that line up.** The worst single command, 2.3M clocks per chunk, is an FMA over 128 rows of 256 floats. Its a, c and destination buffers all start at the same offset inside a 4 KB stripe. So line 0 of each lands on bank 0, line 1 of each on bank 1, and so on. A beat puts 3 accesses on each of 32 banks and none on the other 32. It needs 3 clocks where 2 would do. Skewing doesn't help, because the offsets are the same.
- **Power-of-two strides.** Some operands are read down a column. One FMA's c operand has 512 bytes between elements. 512 bytes is 8 lines, so lane k lands on bank 8k mod 64: only 8 different banks for 128 lanes, which is 16 clocks for that operand. Skewing helps this kind a lot, which is why it saves 0.4 s.
- **Short rows at power-of-two row strides.** Packed rows of 64 values (256 bytes) placed 16 KB apart all start on the same bank. Skewing spreads them.

## Part 6. The options

1. **Fix the arbiter livelock.**
   - Required whatever else is chosen.
   - Cost: nothing.
2. **Turn on the skew.**
   - Cost: two small adds on the bank number (6 bits each). Software sees the same addresses; any run of up to 63 contiguous lines still hits distinct banks.
   - Gain: 7.8 → 7.4 s.
3. **Lay data out for the banks (software, in the program builders).**
   - Start the buffers one command uses together at different offsets inside the 4 KB stripe. For example, a at 0, c at 2 KB, so they take different halves of the banks.
   - Pad power-of-two strides by one line, for example a 512-byte column stride becomes 576. Each lane then moves to a new bank.
   - Cost: allocator work, a little SRAM for padding (one line per padded row).
   - Gain: toward the 6.85 s bound. How close needs measuring.
4. **128 banks (8 KB per clock).**
   - Gain over options 2 and 3: about 0.25 s (6.85 → 6.6). More room if the matrix unit and the vector unit really do contend.
   - Cost: twice the bank count for the same 6.5 MB, so about twice the bank periphery, and a wider crossbar between the banks and the 512 lanes. D-028 estimated this.
   - Worth it only if the RTL arbiter shows real contention after options 1–3.
5. **Dual-ported banks (a read and a write to the same bank in one clock).** Leaning no.
   - Writes are about a quarter of vector traffic, so it relieves only that quarter.
   - Two-port macros are commonly quoted at roughly 1.3–1.5× the area of single-port ones. Not checked against a real ASAP7 macro.
6. **Wider matrix-unit output tiles** (fewer activation reloads, which are most of the matrix unit's 1–2 KB per clock).
   - Only if the arbiter shows the matrix unit and the vector unit fighting.
   - Cost: a bigger accumulator.

## Part 7. Recommendation

- **Now:** options 1 and 2. They're free and needed.
- **Next:** option 3, then measure with the real arbiter. Run the tiny prefill at 1, 2 and 4 chunks, at 64 and 128 banks, skew on.
- **Decide option 4 after that measurement.** If 64 skewed banks with good layout come within about 0.3 s of 128, keep 64 (D-028 stands, with skew on).
- **Leave the rest as is:** line size, single-ported macros, the DeltaNet state's own banks (D-023), and the read-before-write rule.

## Part 8. What these numbers don't know

- The bank model samples each descriptor's first 3 uses and the first 64 beats of each. It takes a beat's write lag from the command's pipeline latency.
- It charges vector beats only. The matrix unit's and DMA's lines aren't in the collision count; the bandwidth model brackets them (7.2–7.9 s).
- The real cost comes from the arbiter in RTL. The tiny model (hidden size 1024) has different strides from the 9B, so it checks the mechanism, not the exact seconds.

## Part 9. Results of running the fixes (2026-11-22, night)

The user asked for the planned fixes to be run. What each one measured (same bank model, 32K prefill, 6.6 s with a free SRAM):

- **Livelock fix:** works. Prefill with the arbiter on now finishes.
- **A second arbiter bug, found by that run:** the vector unit's pass path (COPY, MAX, CAUSAL) made its read request depend on its own write's ready. A read that appeared after the arbiter's decision met the default "ready" of a requester that wasn't asking, and the unit used stale data. Fixed in `vec_box_command.sv`. The SRAM model now stops with an error if any request changes after arbitration. Decode with the arbiter on passes every byte again (51,250 clocks).
- **Skew:** on by default in the chip runs (`SKEW=1` in `run_chip.py`). 7.8 → 7.4 s.
- **Staggering buffer starts in the allocator** (`BANK_STAGGER` in `qwen_v1.py`, off by default): almost nothing, 7.41 → 7.37–7.39 s for steps of 16, 29 and 37 lines. The colliding operands are mostly inside one buffer, or a command reading and writing in place. Moving buffer starts can't separate those. It costs 29–48 KB of padding.
- **A write buffer in front of the banks** (new option): the vector unit's write lines wait in a 64-line buffer (4 KB) and drain into banks the clock's reads leave free, so reads keep priority.
  - 64 banks, skew + write buffer: 7.19 s
  - 128 banks, skew + write buffer: 6.89 s
- **Where that leaves the choice:**
  - Keeping 64 banks with skew and the write buffer costs about 0.6 s of the 6.6.
  - Going to 128 banks buys about 0.3 s more, for twice the bank periphery.
  - The worst command is still the in-place FMA over 128 rows of 256 floats. Its reads hit the same 32 banks. Changing that command's layout in the program (for example a separate destination at a 2 KB offset) is the next software fix to try.
