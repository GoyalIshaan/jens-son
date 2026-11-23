# 24. Progression step 1: one matrix-unit group in RTL, tested and synthesized

*Done 2026-11-20. Code: `rtl/mxu/` (`mxu_group.v`, `ref_group.py`, `tb_group.v`, `syn_wrappers.v`, `synth.py`). Results: `rtl/mxu/results/`. All numbers below are from the **ASAP7 7 nm FinFET** library (RVT, typical corner, 0.7 V, 25 °C), unless marked otherwise. N6-class area is taken as ASAP7 ÷ 1.18, from TSMC's claim that N6 is 18% denser than N7.*

## Background for readers: cells, gates, paths and the clock

**Where "1 GHz" comes from**
- It isn't a law. It comes from matching the multipliers to the memory.
- Memory delivers 512 GB/s × 92% = 471 GB/s, and each weight takes 9/16 of a byte, so about **837 billion weights arrive per second.**
- The matrix unit must keep up: lanes × clock ≥ 837 billion per second.
  - 1,024 lanes at 1 GHz: 837 weights arrive per cycle, with 20% headroom.
  - 800 MHz: 1,047 per cycle, too many for 1,024 lanes.
  - 2 GHz: 512 lanes would do.
- So a slower clock means more lanes (area), and a faster clock means shorter logic between registers (harder timing). 1 GHz is a moderate target for a 6 nm-class process. Step 1 checks it can be met.

**Standard cells and gate equivalents**
- Chips are built from a factory's library of pre-made blocks called **standard cells:** NAND, XOR, full adder, flip-flop, in several strengths. All have the same height and different widths, so they line up in rows like bricks.
- **Synthesis** turns Verilog into a list of these cells (a netlist).
- **The unit of size is a 2-input NAND cell:**
  - SkyWater 130 nm: 3.75 µm²
  - ASAP7 7 nm: 0.0875 µm², about 43× smaller
- **"Gate equivalents"** = a block's area ÷ the NAND2 area: a rough, node-independent size.
- **Anchors at 7 nm:**
  - an int8 × int8 multiplier: about 300 gate equivalents (26.6 µm²)
  - a 16-input adder tree: about 1,370 (120 µm²)
  - a flip-flop: 0.29 µm²
  - 1 mm² holds about 11 million NAND2 cells, or about 37,000 int8 multipliers if nothing else were there.
- **Real layouts aren't solid cells.** Wiring needs room, so typically only about 60–70% of the area holds cells. That's why we divide cell area by 0.70.

**Paths and timing**
- Between two flip-flops, a signal passes through a chain of cells, called a **path.** Each cell adds delay, and so do wires.
- At 1 GHz every path has **1,000 ps** (minus a little for clock skew and setup time).
- **The critical path** is the slowest one, and it sets the fastest clock.
  - Example: the int8 multiplier's longest path is about 531–659 ps at 7 nm, roughly 20–30 cells at about 20–30 ps each. It fits.
- **Pipelining:** if a path is too long, add flip-flops in the middle so it takes two cycles. Each stage gets its own 1,000 ps. The cost is extra flip-flops and one more cycle of latency.
- **Smallest vs fastest:** synthesis can build a smaller, slower version or a bigger, faster one. The adder tree is 912 ps at its smallest, or 750 ps for about 60% more area.

**Why 7 nm (ASAP7) and not 130 nm**
- SkyWater 130 nm is a real factory process, but we'd never build on it. It was just the first library set up.
- ASAP7 is an academic model of a 7 nm FinFET process, close to the class we'd use, so its numbers need no scaling guess. All numbers in this note are ASAP7.
- What carries over between the two is the **ratios:** the fp32 scale step is about 37% of each row on both.

## What was built

- **One group of the matrix unit:** 16 weight lanes × R token rows, a 3-stage pipeline (D-015, D-017, D-005).
  1. **Lookup and multiply:** each 4-bit code picks an int8 value from the 16-entry table, then multiplies the int8 activation of every token row.
  2. **Adder tree:** 16 products → 8 → 4 → 2 → 1 per row, exact.
  3. **Scale step:** times the group's E4M3 weight scale and the row's activation scale.
- **The full unit is 64 of these groups** (1,024 lanes). Not built yet: the step that adds the 64 group results of a beat together, and the per-output accumulators.
- **Three versions of the scale step:**
  - **Mode A** (D-017 as written): two fp32 multiplies per row, rounding twice.
  - **Mode C:** an exact 19 × 4-bit integer multiply by the E4M3 mantissa, then one fp32 multiply. It rounds once, so it's also more accurate than A.
  - **Mode B:** activation scales are powers of two, so the step is shifts only and the output is exact. This would change the number format, so it needs a quality check.
- **Conventions fixed in the RTL** (now part of the bit-exact contract):
  - round to nearest, ties to even
  - fp32 results below 2⁻¹²⁶ flush to +0
  - a zero result is +0
  - the E4M3 NaN code is never used for a scale

## Correctness

- **Bit-exact against the Python reference**, every output bit compared:
  - 4 and 32 rows, all three modes: 1,600 vectors each
  - then 4 rows, all three modes: 24,000 vectors each
  - **0 mismatches**
- **The vectors include worst cases:** all +127 × +127 and all +127 × −127, which push the adder tree to its maximum of ±258,064.
- **The test catches bugs.** I put deliberate errors into copies of the RTL:
  - rounding "half up" instead of to even: caught, 192 mismatches in mode A and 1 in mode C
  - an adder level too narrow to hold its sum: caught, 640 mismatches
- **A real saving the test turned up:** values are always in −127..127, never −128. So each product is at most 16,129 and the tree levels need only 16, 17, 18 and 19 bits, one bit less per level than first written. The RTL now uses the narrower tree.

## Area (ASAP7)

**Pieces:**
- **one int8 × int8 multiplier:** 26.6 µm² (304 gate equivalents; one gate equivalent = one NAND2 = 0.0875 µm²)
- **one 16-input adder tree:** 119.5 µm²
- **the table lookup for one group** (16 lanes × a 16-to-1 mux of 8 bits), shared by all rows: 205.8 µm²
- **the scale step for one row:**
  - mode A: 377 µm²
  - **mode C: 320 µm²**
  - **mode B: 51 µm²**
  - mode D (added for experiment set 5: bf16 activation scale, a 24 × 8 mantissa multiply): 175 µm², and 26,152 µm² for a 32-row group. See note 25.

**Per token row in one group:**
- 16 multipliers 425 µm² + adder tree 120 µm² + scale step
- **With mode C, the scale step is 37% of the row. With mode B it's 8%.**

**A whole group, synthesized with flip-flops:**
- 4 rows, mode C: 4,132 µm². Flip-flops are 13.9% of the group: the stored activations and the pipeline registers.
- 32 rows:
  - mode A: 33,467 µm²
  - **mode C: 30,830 µm²**
  - **mode B: 22,618 µm² (27% smaller than C)**
- 64 rows, mode C: 62,871 µm². That's 1.98× the 32-row group, so the cost grows in proportion to the rows.

**The full unit (64 groups).** Placed area assumes 70% of the area holds cells (an assumption; real layout would measure it):
- **4 rows, mode C:** 0.26 mm² of cells, 0.38 mm² placed, **0.32 mm² at N6**
- **32 rows, mode C:** 1.97 mm² of cells, 2.82 mm² placed, **2.39 mm² at N6**
- **32 rows, mode B:** 1.45 mm² of cells, 2.07 mm² placed, **1.75 mm² at N6**
- **64 rows, mode C:** 4.02 mm² of cells, 5.75 mm² placed, **4.87 mm² at N6**
- **64 rows, mode B (estimated as 2 × 32 rows):** about **3.5 mm² at N6**
- **Pipelining mode C's scale step to 1 GHz** adds 2 register stages: about 3,800 flip-flops per 32-row group at 0.29 µm² each, about +3.6%.

**Against the estimate in note 22:** scaling from TPU v1 gave 5–10 mm² for 32 rows. The measurement is 2.4 mm² at N6 in mode C, **2–4× smaller.** Either way, the matrix unit is a small part of an 80 mm² die: **64 rows costs about 4.5 mm² more than 4 rows, about $0.70 of silicon.**

## Timing at 1 GHz (1,000 ps per stage)

- **The pieces alone** (smallest version / fastest version):
  - multiplier: 659 / 531 ps
  - adder tree: 912 / 750 ps
  - table lookup: 121 / 152 ps
  - scale step:
    - **mode B: 670 ps** (sized)
    - **mode C: 2,301 ps**
    - mode A: 2,441 ps
- **Whole groups, with buffers and gate sizing:**
  - **mode B:** worst stage **743 ps** at 4 rows and **782 ps at 32 rows.** Fits 1 GHz with about 20% margin.
  - **mode C:** worst stage 2,376 ps at 4 rows and 2,569 ps at 32 rows. That's the fp32 scale step, so it needs 3 pipeline stages.
- **A trap found on the way:** without buffering, a 32-row group reported a 27 ns worst stage, and a 64-row group 70 ns. Each looked-up weight drives 32–64 multipliers, and plain gate mapping adds no buffers. With ABC's buffering and gate sizing switched on, it drops to 782 ps. So **fan-out has to be designed for:** a buffer tree, or a register per block of rows.
- **Limits:**
  - This is synthesis, not place and route, so wire delay isn't counted. Real paths will be somewhat slower.
  - Power wasn't measured, because that needs switching-activity simulation and a power tool (OpenROAD/OpenSTA, not installed).

## What this decides, and what it opens

- **1 GHz holds** for the lookup, the multipliers and the adder tree at 7 nm. The fp32 scale step needs 3 pipeline stages, or mode B.
- **The row count is cheap in area.** 64 rows costs about 4.9 mm² at N6 in mode C, or about 3.5 mm² in mode B. With prefill at 32K tokens in 6.8 s instead of 11.3 s, **64 rows is now the better choice for the long-context target** (D-015, still proposed until power is measured).
- **The scale step is the expensive part, not the multipliers.** Options, in order of how much they change:
  1. **Use mode C instead of A.** It's smaller, more accurate, and needs no quality test. Adopted in D-017.
  2. **Share one scale step between the two weight groups that share an activation scale (per 32).** Line up their exact products, add them, then do one fp32 multiply. Half the fp32 multipliers, same numbers.
  3. **A shorter activation-scale mantissa** (8 bits, bf16-like): the multiplier is about 3× smaller. Needs a quality test.
  4. **Power-of-two activation scales** (mode B, like OCP's MXINT8): 27% smaller groups, one pipeline stage, and the 64 group results of a beat can be added exactly in fixed point, then converted to fp32 once per row instead of 64 times. Needs a quality test.
- **Recommended next:** a GPU test of options 3 and 4 against today's fp32 activation scales, with KL on code and prose (experiment sets 2–3 style). About $1–2.
- **The next RTL piece:** the beat reducer (64 group results → 1 per row) and the per-output accumulators. Its size depends on the test above:
  - with fp32 scales it's 63 fp32 adders per row
  - with power-of-two scales it's wide integer adders and one conversion per row

## Tools and files

- **Yosys 0.63** with ABC (`synth -booth`, `map`, then `buffer -p; upsize; dnsize` for timing), Icarus Verilog 13.0 and Verilator 5.048.
- **Libraries** (in `rtl/third_party/`):
  - **SkyWater 130 nm HD** (`sky130_fd_sc_hd__tt_025C_1v80.lib`), from OpenROAD-flow-scripts. Used only for the first checks; its absolute numbers aren't used above.
  - **ASAP7** RVT typical-corner NLDM libraries, from OpenROAD-flow-scripts. NAND2xp33 is 0.08748 µm².
- **Rerun:**
  - `cd rtl/mxu; python3 ref_group.py 32 1 200 vec`
  - `iverilog -g2005 -P tb_group.R=32 -P tb_group.MODE=1 -o sim tb_group.v mxu_group.v; vvp -n sim`
  - `SYN_LIB=asap7 python3 synth.py 4 8 32 64`
- **N6 scaling sources:**
  - N7 dense cells about 91.2 MTr/mm² (WikiChip, https://en.wikichip.org/wiki/N7)
  - N6 has 18% higher logic density than N7 (TSMC, reported by AnandTech, https://www.anandtech.com/show/14228)
