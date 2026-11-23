# 29. Golden v2, part 1: the exact arithmetic contract

*Progression step 4, started overnight 2026-11-21. Code: `golden/bitexact.py` (the matrix unit, bit-exact), `golden/chip_sfu.py` (the special-function unit as real algorithms), the `CHIP` switches in `golden/qwen35_golden.py`, experiment set 8 (`golden/beam_exp8.py`).*

## What golden v2 is for

- Until now the golden model was "the model with the chip's formats": weights, activations and KV rounded the chip's way, but the arithmetic done the GPU's way.
- **Golden v2 says exactly what the chip computes,** down to the order of every fp32 add.
- Two uses:
  - **It's the contract:** the RTL and the instruction-level simulator (step 6) must match it bit for bit, and its test vectors check every block.
  - **It's the quality check:** run on real text, it must give the same quality as the chip-format model (experiment set 3), within noise.

## 1. The matrix unit, bit-exact (`golden/bitexact.py`)

- **Activations (D-019), with integers only:**
  - The scale exponent is the smallest e with amax ≤ 127 · 2^e.
  - Write amax = m · 2^x with m in [0.5, 1). Then e = x − 7 if m ≤ 127/128, else x − 6. No division and no log, so the hardware needs only a comparator.
  - **Checked:** it gives the same values as the golden format's `2^ceil(log2(amax/127))` on every value tested.
  - **Added 2026-11-21 (found by experiment set 9):** a group whose exponent would fall below −126 (amax under about 2⁻¹¹⁹) is stored as all zeros with e = −126. Its values would flush to zero anyway, and the old helper computed 2⁺¹³³, which overflows fp32 and turned 0 × ∞ into NaN. Strongly decaying DeltaNet heads produce such groups in the chunked form (note 30).
- **Weights (D-005):** weight = (k / 127) × E4M3 × gmax.
  - k: the table entry, an int8
  - E4M3: the scale per 16 weights
  - gmax: an fp32 scale per tensor
- **One output, one beat, as the RTL does it (notes 24, 26):**
  - S_g = Σ k·q over 16 lanes (exact)
  - P_g = S_g × M (the E4M3 mantissa)
  - emax over all 64 groups
  - keep 16 fraction bits, rounding down
  - one fp32 rounding per beat
  - one fp32 add per beat, in input order
- **Found here, a gap in the RTL path:** the per-tensor gmax / 127 isn't a power of two, so it can't ride in the exponents.
  - The matrix unit's output stage needs **one fp32 multiply per output** by a per-matrix constant: y = fp32(acc × fp32(gmax/127)).
  - About 64 multipliers, 0.01 mm². It goes in the MM descriptor as a field.
- **Checks:**
  - **Against the RTL's own reducer reference** (`rtl/mxu/ref_reducer.py`): 0 mismatches in 3,000 random beats, including exponent gaps over 45 bits.
  - **Against a float64 product** of the same rounded values, on gaussian weights in the chip format and activations with outliers:
    - the chip's exact path is off by at most 4.5 × 10⁻⁷ of the typical output
    - a plain fp32 matrix multiply is off by 6.3 × 10⁻⁶
  - **The chip's arithmetic is about 14 times more accurate than the GPU's fp32.** Its group sums are exact integers, and it rounds once per 1,024 inputs instead of once per add.
  - **So the quality check for the matrix part is already done:** experiment set 3 used ordinary fp32 accumulation, which is less accurate than the chip.

## 2. The special-function unit as real algorithms (`golden/chip_sfu.py`)

Experiment set 6 imitated the unit by rounding exact results to b bits. These are the actual algorithms the hardware will run (D-020's widths):

- **eˣ:**
  - y = x · log₂e, split into an integer n and a fraction f
  - 2^f from a 64-entry table of 2^(i/64), times 1 + r + r²/2 for the remainder r
  - shift by n, round to 16 bits
  - Max error 2⁻¹⁶·¹.
- **1/x:** a 256-entry table on the mantissa, one Newton step, 12 bits. Max error 2⁻¹³·⁰.
- **1/√x:** a 256-entry table on [1, 4), one Newton step, 16 bits. Max error 2⁻¹⁵·⁹.
- **sigmoid:** 1/(1 + e^−|x|) from the two units above, mirrored for x < 0, 12 bits. Max error 2⁻¹²·⁴. **SiLU** is x × sigmoid(x).
- **softplus:** max(x, 0) + ln(1 + e^−|x|), with ln from a 64-entry log₂ table plus a quadratic, 12 bits. Max error 2⁻¹²·⁹.
- **A bug the self-test caught:**
  - For very negative x, 1 + e^−|x| rounds to exactly 1 in fp32, so the log gave 0: a 100% relative error.
  - The hardware needs a short series, ln(1 + z) ≈ z − z²/2 + z³/3, when z < 2⁻⁶.
- These are written as single fp32 operations in a fixed order, so the GPU and the chip produce the same bits.

## 3. The other chip-exact pieces (switches in `qwen35_golden.CHIP`)

- **rope:**
  - RoPE angles come from two 1,024-row tables, angle(p >> 10 × 1,024) and angle(p & 1,023), made in float64 offline and combined by angle addition in fp32.
  - This is actually *more* accurate than the reference. The reference computes position × frequency in fp32, which at position 32K has an angle error of up to about 0.002 rad.
- **softmax:**
  - online softmax over 1,024-position tiles (note 28's order)
  - the V scale is folded into P before P is rounded to int8
  - 1/l comes from the recip unit
  - **Checked on random data:** with the int8 rounding off, it matches plain softmax attention to 1.4 × 10⁻⁶ (3 × 10⁻⁵ with the real special functions).
- **dnq:** DeltaNet's q × fp32(1/√128) instead of a division.

## 3b. The vector unit's operation order (`golden/bitexact_vec.py`)

- **Why it matters:** the same sum done in a different order gives different bits. Bit-exactness needs the order written down. Proposed rules, for 512 lanes with fused multiply-add:
- **Reductions** (sum, sum of squares):
  - lane j adds elements j, j + 512, j + 1,024, ... in sequence
  - then a pairwise tree over the 512 lanes
  - Max is order-free.
- **RMSNorm:** s = sum of squares; r = 1/√(fma(s, 1/n, ε)); y = x · r; then int8(y · w). Two roundings, in that order.
- **DeltaNet head step (one-pass order):**
  - lanes = 4 rows × 128 columns
  - S·k and S·q accumulate with fma over rows i ≡ r (mod 4), then (p₀ + p₁) + (p₂ + p₃)
  - δ = (v − d·Sk)·β
  - o = fma(k·q, δ, d·Sq)
  - S ← fma(k, δ, d·S)
  - **Replay** uses the same update without the q side.
- **Conv:** nested fmas, oldest input first.
- **Fused softmax tile:** × K scale; mask; max; exp(s − m); the sum by the reduction rule; α = exp(m_old − m_new); l = fma(l, α, sum); int8(p × V scale).
- **Checks:**
  - **The reference fma is exactly once-rounded:** TwoSum plus a midpoint fix; 0 mismatches against exact rational arithmetic.
  - **Agreement with float64:** reductions to 2.5 × 10⁻⁹; the DeltaNet step to 3 × 10⁻⁶, which comes from the 16-bit eˣ of the decay.
  - **Deterministic.**
- **Test vectors** (`golden/vectors/`, `python3 golden/make_vectors.py`):
  - the 5 special functions (4,096 inputs each, plus edge cases)
  - activation quantization
  - full matrix outputs through the real weight format (8 rows × 64 outputs × 4,096 inputs)
  - DeltaNet token steps
  - fused softmax rows

## 4. Experiment set 8: quality of all of this on real text

*2026-11-21, RTX 4090, 47 minutes, `golden/beam_exp8.py`, `golden/results/exp8_chip_exact.json`. Paired KL against exact attention and functions, 8,192 tokens of prose and of Python code, plus 32,768 tokens of prose. The first launch ran out of GPU memory: the table-based eˣ makes several full-size temporaries of the 8K × 8K score matrix, so the special functions now work in 4M-value pieces.*

- **Baseline (bf16 weights, q and P int8, D-021, untiled):** prose 0.000386, code 0.000068, 32K 0.000669. This reproduces experiment set 7 exactly.
- **One piece at a time, paired against that baseline:**
  - **real special-function algorithms:** prose −0.00007 ± 0.00011, code +0.000002. No change.
  - **RoPE from tables:** prose −0.00009 ± 0.00006, code 0.000000. No change, slightly closer if anything.
  - **DeltaNet q × 1/√128:** exactly 0. No change.
  - **tiled online softmax with the V scale folded into P:** prose +0.00029 ± 0.00016, code **+0.000052 ± 0.000004**. A real but tiny increase.
- **All together:**
  - **8K:** prose 0.00177 (paired +0.0014 ± 0.0011, a noisy mean set by a few positions), code 0.000123
  - **32K:** 0.00111 ± 0.00041 (vs 0.00067), top-1 agreement 99.37%, perplexity +0.64%
- **On top of the chip's full numerics** (GPTQ weights, D-019 activations, int8 KV, D-021):
  - prose 0.0527 → 0.0560 (paired +0.0033 ± 0.0030, within noise)
  - code 0.0562 → 0.0564 (+0.0002 ± 0.0005)
  - 32K 0.0479 → 0.0448 (within noise)
  - **Not measurable next to the weights' error.**
- **Conclusion:** golden v2's non-matrix pieces cost nothing measurable on top of the chip's numerics, so the chip-exact model keeps experiment set 3's quality.
  - The one piece that shows up on its own, with bf16 weights, is folding the V scale into P. It's still about 50 times below what the 4-bit weights cost.
  - **The likely reason:** each position's P is scaled by its own V scale before the shared rounding of each 32-position group, which widens the range inside a group.
  - **The cheap alternative, if it ever matters:** one V scale per 32 positions per head, so the fold is a single factor per group. That changes the KV format and needs an fp32 staging copy of the newest 32 V rows per layer (about 1 MB). Not worth it now.
- **Step 4's quality check, the plan's "golden v2 matches experiment set 3 within noise," is therefore met:**
  - for the matrix part by construction: it's more accurate than the fp32 the experiments used (§1)
  - for the rest by experiment set 8
