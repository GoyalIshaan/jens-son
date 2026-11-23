# 10. Number formats: weights, activations, accumulators and state

*Started 2026-11-10. Status: under consideration. The leaning is recorded as D-005 (proposed) in `DECISIONS.md`. The numbers come from `number_formats()` in `calc/qwen35_budget.py`.*

> **Update 2026-11-11: measured.** `15_measurements.md` tests every option below on the real model. In short:
> - plain int4 costs +20% code perplexity; NVFP4-style scales and a learned 4-bit table do clearly better at the same 4.5 bits
> - 3-bit is ruled out
> - int8 activations and int8 KV cost nothing measurable
> - a bf16 DeltaNet state does measurable harm that grows with length, so the state stays fp32
>
> D-005 is updated to match. The rest of this note is the reasoning as it stood before the measurements.

## Why this matters more than usual

At batch 1, almost every byte we move is a weight byte. So **the weight format sets our speed:**
- Each bit per weight costs about 1 GB per token, because there are 7.94 billion weights read per token and 7.94 B × 1 bit is about 1 GB.
- At 512 GB/s, that's about 22 tok/s per bit at our operating point.

The formats of everything else (activations, accumulators, state) matter less for speed. They matter for **accuracy and multiplier area**. So the question splits in two:
- weights: as few bytes as possible without losing quality
- everything else: as accurate as needed, as cheap as the area allows

## What the reference model uses

From the Qwen3.5-9B `config.json`:
- **Weights are bf16** (`dtype: bfloat16`). Every lower-bit format is our quantization, and we have to check its quality ourselves.
- **The DeltaNet state is fp32** (`mamba_ssm_dtype: float32`). The authors keep the recurrent state at full precision, which is a strong hint not to cut it.

## Weights

Options, with total bits per weight including the shared scales, and the speed each gives on our card (512 GB/s, 92% utilization, 4K context):
- **int4 with one fp16 scale per 32 weights:** 4.50 bits, 4.46 GB per token, about **102 tok/s**. This is our current default and the same size as llama.cpp's Q4_0.
- **int4 with one fp16 scale per 128 weights:** 4.12 bits, 4.09 GB, about 111 tok/s. 9% faster, but coarser scales usually cost some quality.
- **MXFP4** (an OCP standard: 4-bit floats called E2M1, with one 8-bit power-of-two scale per 32): 4.25 bits, 4.22 GB, about 107 tok/s.
- **NVFP4** (NVIDIA's format: E2M1 with an FP8 scale per 16): 4.50 bits, same size as our default.
- **Mixed precision: FFN at 3 bits, everything else at 4 bits:** 3.89 bits, 3.86 GB, about **117 tok/s**. The FFN is 61% of the weight bytes, so it's the place where saving bits pays most.
- **int8:** 8.12 bits, about 57 tok/s. Too slow to beat the RTX 5060.
- **bf16:** 16 bits, about 29 tok/s. That's under our 30 tok/s floor. It's the quality reference only.

A small hardware detail in MXFP4's favor:
- An E2M1 value times 2 is always a whole number between −12 and 12: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12.
- Its group scale is a power of two, so applying it is a shift, not a multiply.
- So an MXFP4 multiplier is about as small as an int4 one, and the scale step is nearly free.

The catch: a power-of-two scale is coarse. It can only be ×1, ×2, ×4 and so on, so values in a group get rounded more. Commonly reported results put plain MXFP4 below int4 with fp16 scales at the same group size. **We'll measure this on Qwen3.5-9B rather than trust that.**

**Leaning:** make the hardware support **int4 with a 16-bit scale per 32 weights as the main path, plus a 3-bit mode.** Then pick the exact recipe (group size, which layers get 3 bits) in the golden model by measuring quality. Each layer's format is set by the program, so we can change it without changing the hardware.

## Activations (the inputs to each matrix multiply)

At batch 1, the input to a matrix multiply is a single vector of 4,096 or 12,288 values. It's tiny, so its format affects accuracy and multiplier size, not bandwidth.

- **bf16 activations (W4A16):** the safest choice for quality, and what llama.cpp and most GPU kernels effectively do. Each multiplier handles int4 × bf16, which costs more area: it needs exponent handling.
- **int8 activations with a dynamic scale (W4A8):** the multiplier becomes int4 × int8, which is tiny. The risk is that LLM activations have a few huge outliers, and one scale per vector rounds the small values badly. A scale per 32 activations fixes most of that.
- **FP8 activations (E4M3):** a middle ground with built-in range, at a slightly bigger multiplier than int8.

How much does area matter here? Decode needs about 840 MACs per cycle, and prefill or verification needs about 4,096 (see `13_matrix_shapes.md`). At 4,096 MACs, a multiplier several times bigger is still a modest share of an 80 mm² die, and the die is only about 8% of the card's cost (`08_post_design_levers.md`).

**Leaning:** build **W4A8 with a scale per 32 activations** as the main path, and keep **bf16 activations** as the quality fallback. The golden model decides which one ships.

## Accumulators (where products are added up)

Worst-case widths for int4 × int8, from the script:
- **One product** is at most 8 × 127 = 1,016, which needs 11 bits.
- **A sum of 32 products** (one weight group) is at most 32,512, which needs 16 bits.
- **A full row of 4,096** in raw integers needs 23 bits, and **a row of 12,288** (FFN down) needs 25 bits.

But each group of 32 has its own scale, so we can't simply add raw integers across a whole row. The natural design is:
1. Add the 32 products of a group in a **narrow integer adder** (16 to 20 bits).
2. Multiply that group sum by the weight scale times the activation scale.
3. Add the result into an **fp32 accumulator** for the row.

That's one fp32 multiply and one fp32 add per 32 MACs, which is cheap.

**Leaning:** integer accumulation within a group, and **fp32 for everything across groups.** Every output of a matrix multiply is fp32 before it's rounded for the next step.

## Everything else

- **The vector work** (RMSNorm, SiLU, softmax, the gates, the DeltaNet decay `exp`) happens on vectors of a few thousand values, so it's tiny. Do it in **fp32**. Precision here costs almost nothing.
- **The DeltaNet state:** fp32 or bf16. The state is read and written every token, for each of 24 layers:
  - bf16: 50.3 MB per token, 1.1% of the weight bytes
  - fp32: 100.7 MB per token, 2.3% of the weight bytes

  **Leaning: fp32, matching the reference.** It costs 1.1% of speed (about 1 tok/s). An error in the state lasts for every token after it, and at 1M context that's a lot of tokens.
- **The KV cache** (full-attention layers only): int8 with a scale per 64 values by default, and an int4 mode for very long context. At 1M context this choice dominates everything (see `11_memory_capacity_and_1m_context.md`).
- **The logits:** fp32, then argmax or top-k sampling on chip.

## Follow-up (2026-11-10)

`15_golden_model_and_quantization.md` adds three things:
- **Activation quantization costs about 0.03% of the work** and doesn't slow decode, but int8 activations don't speed it up either. The choice is quality vs multiplier area.
- **No official 4-bit Qwen3.5-9B exists.** Community versions are all post-training quantizations.
- **Our plan:** round-to-nearest as the baseline, then GPTQ and AWQ aimed at our format, with QAT as a later option.

## What the golden model must measure

- **Perplexity and a few task scores** for each candidate format against bf16:
  - int4 with a scale per 32 and per 128
  - MXFP4
  - FFN at 3 bits
  - W4A8 vs W4A16
- **Long-context quality** with int8 vs int4 KV, and bf16 vs fp32 state, at 128K and above. Errors that are invisible at 4K may add up over a million tokens.
- **Whether Qwen publishes official quantized checkpoints** (FP8, GPTQ int4) that we could start from. Not checked yet.
