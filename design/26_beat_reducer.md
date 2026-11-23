# 26. The beat reducer: adding 64 group results per row

*Done 2026-11-20. Code: `rtl/mxu/beat_reducer.v`, reference `rtl/mxu/ref_reducer.py`, testbench `rtl/mxu/tb_reducer.v`, results `rtl/mxu/results/synth_asap7_reducer*.json`. ASAP7 7 nm. Follows note 24 (the group) and note 25 (activation scales).*

## What it does

- One weight beat covers 1,024 lanes, which is 64 groups of 16. Each group gives one result per token row (note 24).
- The **beat reducer** adds those 64 results into one number per row.
- It then adds that number into the output's running total. An output with 4,096 inputs takes 4 beats; the FFN down matrix, with 12,288, takes 12.
- **Small example with 4 groups instead of 64:** with power-of-two scales, each group result is an exact integer times a power of two, say 300 × 2⁻⁸, −45 × 2⁻⁶, 7 × 2⁻⁸ and 120 × 2⁻⁷.
  - The largest exponent is −6, so line everything up to 2⁻⁶:
    - 300 × 2⁻⁸ = 75 × 2⁻⁶
    - −45 × 2⁻⁶
    - 7 × 2⁻⁸ = 1.75 × 2⁻⁶
    - 120 × 2⁻⁷ = 60 × 2⁻⁶
  - With fraction bits kept, nothing is lost. The exact sum is 91.75 × 2⁻⁶.
  - Convert to fp32 once at the end.

## Two versions

- **For power-of-two activation scales (mode B):**
  1. find the largest exponent among the 64 groups
  2. shift each group's integer to line up, keeping **F = 16 extra fraction bits**
  3. add all 64 exactly in a 46-bit integer adder tree (24 + 16 + 6 bits)
  4. convert to fp32 once (round to nearest even)
  5. one fp32 add into the running total
- **For fp32 activation scales (modes A, C, D):** each group result is already an fp32 number, so it takes a tree of **63 fp32 adders** per row (rounding at every add), then the same fp32 add into the total.

## A numerics choice this adds to the contract

- When a group's exponent is more than 16 below the largest, its lowest bits fall below what is kept. They are dropped, **rounding down (toward −∞)**, which is the cheapest rule: an arithmetic shift.
- **When it matters:**
  - only if the 64 groups of one beat differ in scale by more than 2¹⁶
  - even then, the dropped part is far below fp32 precision
  - In the tests, switching to "toward zero" changed 1 of 2,000 final results.
- Golden v2 (progression step 4) must copy this rule exactly, and its quality check covers it.

## Correctness

- **fp32 adder:** 30,000 vectors, 0 mismatches. They include exact cancellation, near-cancellation, exponent gaps over 27 bits, the ties region and adding zero.
- **Power-of-two reducer plus the accumulator:** 12,000 beats (3,000 outputs × 4 beats), 0 mismatches. They include exponent spreads of 2, 3, 8, 20 and 40 inside a beat, all-zero beats and cancelling groups.
- **The tests catch bugs:**
  - ignoring the adder's sticky bit: caught, 187 mismatches
  - rounding half up: caught, 36
  - truncating toward zero instead of rounding down: caught, 1 (rare by nature, see above)
- **A lesson on writing RTL for synthesis:** the first version described the max search and the sum as loops, which synthesized into a 64-step chain (15–25 ns). Rewritten as explicit balanced trees (6 levels), it gives the same bits at 4.25 ns.

## Area and timing (ASAP7, per token row)

- **One fp32 adder:** 113.5 µm² (5.0 ns) at its smallest, 198 µm² (1.8 ns) sized for speed.
- **fp32 version, 63-adder tree:** 7,034 µm² (31.4 ns) at its smallest, **11,735 µm² (13.7 ns)** sized for speed.
- **Power-of-two version (tree):** 3,506 µm² (8.2 ns) at its smallest, **3,621 µm² (4.25 ns)** with buffering and sizing, 5,909 µm² (3.9 ns) at its fastest.
- **So the power-of-two reducer is 3.2× smaller and 3.2× faster** than the fp32 tree, before pipelining.
- **Pipelining to 1 GHz:**
  - **power-of-two version:** about 4 stages. The exponents are known before the group sums arrive (the weight-scale stream and the stationary activation scales), so the max search runs a cycle early. The other stage boundaries go where data is narrow: after the third adder level, 8 × 46 bits per row. That's about 130 µm² extra per row (about 4%).
  - **fp32 version:** about 14 stages, with wide registers between fp32 adders.

## For the whole matrix unit (1,024 lanes)

- **32 rows:**
  - power-of-two reducers about 0.12 mm² of cells, against about 0.38 mm² for the fp32 trees (before the fp32 trees' heavier pipelining)
- **64 rows:** about 0.23 mm² against about 0.75 mm².
- **Plus one fp32 accumulator per row** in both versions: 113–198 µm² each. That's 32 or 64 per unit.
- **With note 24's groups, the matrix unit at 32 rows** (cells, before the 0.70 utilization and the N6 factor):
  - **power of two:** groups 1.45 mm² + reducers 0.12 mm² ≈ **1.57 mm²**
  - **fp32 (mode C):** groups 1.97 mm² + reducers about 0.38 mm² ≈ **2.35 mm²**, plus more pipeline registers
  - So power-of-two scales make the whole unit **about one third smaller** and much easier to time at 1 GHz.

## Not done yet

- The pipelined versions (registers placed) and their exact area.
- How results leave the unit: writing outputs to SRAM, then the activation step and the next matrix's input rounding (progression step 2/3).
- Power, which needs place and route (open question).
