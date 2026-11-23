# 25. Can the activation scale be cheaper? (experiment set 5)

*Started 2026-11-20, quality run finished 2026-11-20 (RTX 4090, 24 minutes). Status: **measured, including GSM8K and HumanEval (2026-11-20).** Recommendation below, waiting on the user's sign-off. Hardware: `rtl/mxu/` (modes B, C, D in `mxu_group.v`), ASAP7 7 nm. Quality: `golden/beam_exp5.py`, results in `golden/results/exp5_act_scale.json`.*

## The problem, in one paragraph

The matrix unit does cheap int8 multiplies, but then every result has to be **rescaled.** Each group of 16 products gets an integer sum, and that sum is multiplied by two scales: the weight's scale (E4M3, per 16 weights) and the activation's scale (per 32 activations). With 32 token rows and 64 groups, that's 2,048 rescales every clock cycle. Building one group showed (note 24) that **rescaling with a full fp32 activation scale is 37% of each row's logic** (320 µm² out of about 865 µm² per row at 7 nm). It's also too slow for one 1 GHz clock cycle (about 2.3 ns, against a 1 ns budget). The multipliers themselves are not the expensive part.

## A small example of what "the activation scale" is

- Take 32 activations, the largest of which is 3.81.
- Today's format: scale = 3.81 / 127 = 0.0300, stored as a full fp32 number. Each activation is stored as round(value / 0.0300), an int8 between −127 and 127. So 3.81 → 127, and 1.00 → 33.
- The hardware later computes (integer sum) × 0.0300 × (weight scale).
- **The two ideas change only how precisely 0.0300 is stored:**
  - **Idea 1 (bf16-style):** keep an 8-bit mantissa (1 + 7 bits), so 0.0300 = 1.92 × 2⁻⁶ becomes 1.921875 × 2⁻⁶ = 0.03003 (off by 0.1%; the worst case for any value is 2⁻⁸ ≈ 0.4%).
  - **Idea 2 (power of two):** use the next power of two, 2⁻⁵ = 0.03125. 3.81 → round(3.81 / 0.03125) = 122 instead of 127, so a little of the int8 range goes unused. In exchange, "multiply by the scale" becomes "shift the binary point by 5", which costs almost nothing in hardware.

## Why the multiplier shrinks

- A multiplier's size grows roughly with **the product of its input widths.**
  - fp32 × fp32 needs a 24 × 24-bit mantissa multiplier (576 bit-products).
  - fp32 × bf16 needs 24 × 8 (192 bit-products, a third).
  - fp32 × power of two needs no multiplier, just adding the exponents.
- That's the whole hardware argument. The question is only whether the model's output changes.

## Hardware (measured, ASAP7 7 nm; bit-exact tests pass for every mode)

- **The scale step for one row:**
  - two fp32 multiplies (mode A, D-017 as first written): 377 µm²
  - **today, one fp32 multiply (mode C):** 320 µm², 2.30 ns
  - **idea 1, bf16 scale (mode D):** **175 µm² (−45%)**, 2.07 ns when sized for speed
  - **idea 2, power of two (mode B):** **51 µm² (−84%)**, 0.67 ns
- **A whole 32-row group:**
  - mode C: 30,830 µm²
  - **mode D: 26,152 µm² (−15%)**
  - **mode B: 22,618 µm² (−27%)**
- **The full 1,024-lane unit at 32 rows, at N6-class** (÷ 0.70 for wiring room, ÷ 1.18 for N6 vs 7 nm):
  - mode C: 2.39 mm²
  - mode D: about 2.03 mm²
  - mode B: 1.75 mm²
- **Timing at 1 GHz:**
  - modes C and D still need about 3 pipeline stages for the scale step
  - **mode B fits in one** (the whole 32-row group's worst stage is 782 ps with buffering)
- **Mode D's correctness check:** for bf16-valued scales it must give exactly mode C's answer. 0 mismatches at 4 and 32 rows, and over 24,000 vectors.

## A bonus only idea 2 gives

- With power-of-two scales, every group's result is an exact integer times a power of two.
- So the 64 group results of one 1,024-lane beat can be **added exactly in fixed point,** and converted to floating point **once per row** instead of once per group.
- That shrinks the next block too (the "beat reducer", not built yet): 63 fp32 adders per row become wide integer adders plus one conversion.

## Quality: what's measured so far

*KL = KL divergence of the next-token distribution from the bf16 reference model, averaged over positions. Lower is better; 0 means identical.*

**Smoke test** (512 tokens of Python code and 512 of English prose; GPTQ calibrated on only 8 short sequences):
- **bf16 weights** (only the activation format changes), KL on prose / code:
  - today (fp32 scale): 0.00074 / 0.00064
  - idea 1, bf16 nearest: 0.00068 / 0.00074
  - idea 1, bf16 rounded up: 0.00095 / 0.00080
  - idea 2, power of two rounded up: 0.00142 / 0.00248
  - idea 2, OCP MX style: 0.00148 / 0.00254
- **Chip weights + int8 KV:**
  - today: 0.0507 / 0.0870
  - idea 1, nearest: 0.0499 / 0.0882; rounded up: 0.0501 / 0.0855
  - idea 2, rounded up: 0.0521 / 0.0835; MX style: 0.0510 / 0.0820

**Full run** (8,192 tokens of Python code and 8,192 of English prose; GPTQ calibrated on 64 × 512 tokens; RTX 4090):

*Two numbers per line: KL from the bf16 reference, then the **paired** change against today's format on the same positions (± standard error). Code is the precise measurement: its positions are uniform, stderr about 0.0001. Prose is dominated by a few positions with very large KL (stderr about 0.003), so prose differences under about 0.005 can't be told apart.*

- **bf16 weights** (only the activation format changes):
  - **today, fp32 scale:** code 0.00102, prose 0.00917
  - **idea 1, bf16 nearest:** code 0.00105, change **+0.00004 ±0.00005**; prose 0.00713, change −0.0020 ±0.0011
  - **idea 1, bf16 rounded up:** code 0.00108, change +0.00006 ±0.00004; prose 0.00696, change −0.0022 ±0.0013
  - **idea 2, power of two rounded up:** code 0.00216, change **+0.00114 ±0.00010**; prose 0.01088, change +0.0017 ±0.0026
  - **idea 2, OCP MX style:** code 0.00217, change +0.00115 ±0.00010; prose 0.00691, change −0.0023 ±0.0023
- **Chip weights + int8 KV** (the real chip):
  - **today:** code 0.05572, prose 0.05667 (perplexity +5.9% and +0.5% against bf16)
  - **idea 1, bf16 nearest:** code change **−0.0008 ±0.0003**, prose −0.0031 ±0.0016
  - **idea 1, bf16 rounded up:** code −0.0005 ±0.0004, prose −0.0043 ±0.0036
  - **idea 2, power of two rounded up:** code change **+0.0002 ±0.0006**, prose −0.0044 ±0.0038
  - **idea 2, OCP MX style:** code +0.0006 ±0.0006, prose −0.0028 ±0.0036
  - Next-token loss (NLL) changes against today: all within about 1.5 standard errors of zero (largest −0.005 ±0.002, MX on prose with bf16 weights).

**What this shows:**
- **Idea 1 (bf16 scale) is free.**
  - On code with bf16 weights, the change is +0.00004 ±0.00005, zero within noise.
  - With the chip's weights it's, if anything, slightly lower than today (−0.0008 ±0.0003). Treat that as no change: the scale's 0.4% worst-case rounding is invisible next to everything else.
- **Idea 2 (power-of-two scale) has a real but small cost.**
  - Measured on its own (bf16 weights, code), it **doubles the activation error:** KL 0.00102 → 0.00216, +0.00114 ±0.00010. That's 11 standard errors, so it's real. It's what wasting up to 1 bit of the int8 range costs.
  - **On the real chip it disappears into the weight error:** +0.0002 ±0.0006 (rounded up) and +0.0006 ±0.0006 (MX), against a total of 0.0557. That's at most 1% of the total, and far below the 0.004 we treat as run-to-run noise (note 15 §8).
  - Rounded-up and MX-style behave the same. Rounded-up never clips, so it's the simpler rule.
- **Why the chip hides it:** errors add up roughly by size, and the 4.5-bit weights' error is about 50× larger than the extra activation error. Halving the activation precision barely moves the total.

## Real tasks: GSM8K and HumanEval (`exp5_act_tasks.json`)

*RTX 4090, 213 minutes. Chip weights (one GPTQ run shared by all three formats), every linear layer's input rounded with the activation format. To keep the comparison fair, each input reaches the matmul at about fp32 precision: it's split into a bf16 high part and a bf16 low part, and both are multiplied. Greedy decoding, the same prompts as experiment set 3. Paired against today's format on the same problems, with an exact sign test on the problems that changed.*

- **Today, fp32 scale:** GSM8K 1,143/1,319 = **86.66%**, HumanEval 116/164 = **70.73%**
- **Idea 2, power of two:**
  - GSM8K 1,146/1,319 = **86.88%:** **+0.23 ±0.51 points,** lost 21, gained 24, p = 0.77
  - HumanEval 114/164 = **69.51%:** −1.22 ±1.73 points, lost 5, gained 3, p = 0.73
- **Idea 1, bf16 scale:**
  - GSM8K 1,150/1,319 = **87.19%:** +0.53 ±0.44 points, lost 13, gained 20, p = 0.30
  - HumanEval 115/164 = **70.12%:** −0.61 ±1.06 points, lost 2, gained 1, p = 1.0
- **Identical generated text to today's format:** power of two 60% (GSM8K) and 76% (HumanEval); bf16 69% and 83%. Small changes in rounding change some answers' wording, and the right/wrong outcomes balance out.
- **For context, the cost of rounding activations at all:**
  - against experiment set 3's chip run with unrounded activations (87.41%), today's int8 activations score 86.66%: −0.76 ±0.49 points, lost 26, gained 16, p = 0.16. Not significant.
  - Caveat: experiment set 3 used a separate GPTQ run, so this comparison also includes run-to-run weight differences.
- **What it shows:** none of the differences is significant. On GSM8K, power-of-two scales are within ±1 point of today's (95% interval about −0.8 to +1.2 points). HumanEval is too small (164 problems) to rule out a couple of points either way. It matches the KL result: on the real chip, the scale's precision is lost in the weights' error.

## Recommendation

- **Adopt power-of-two activation scales** (idea 2, rounded up: scale = 2^⌈log₂(amax/127)⌉ per 32 activations).
  - **Hardware:** 27% smaller matrix-unit groups (2.39 → 1.75 mm² at N6 for 32 rows; about 4.9 → 3.5 mm² for 64), and the scale step fits one 1 GHz cycle instead of three.
  - **Bonus:** the 64 group results of a beat can be added exactly as integers, so the beat reducer gets cheaper too.
  - **Cost:** no measurable quality change with the chip's weights. KL +0.0002 ±0.0006, against a total of 0.056.
- **Confirmed on real tasks** (above): no significant change on GSM8K (+0.23 points) or HumanEval (−1.2 points, 2 problems).
- **Fallback, if ever needed:** idea 1 (bf16 scale). It's free in quality and saves 45% of the scale step, but it still needs 3 pipeline stages.

## How we decide

- **If idea 2 holds** (paired KL change with the chip's weights within noise, and small with bf16 weights): use power-of-two activation scales (mode B) and an exact fixed-point beat reducer. A follow-up GSM8K run would confirm it on a real task, as we did for the weights (note 15 §13).
- **If only idea 1 holds:** use bf16 activation scales (mode D), and share one scale step between the two weight groups that share an activation scale (halves it again).
- **Either way:** this changes the activation format in D-005, so it gets logged as a decision with this evidence.
