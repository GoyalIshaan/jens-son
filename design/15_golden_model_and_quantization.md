# 15. The golden model, and how quantization actually works

*Written 2026-11-10 from a design discussion. Status: explainer plus leanings. It adds to `06_build_and_verify_plan.md` (the layers of models) and `10_number_formats.md` (the format options).*

## What a golden model is

A golden model is software that does **exactly the arithmetic our chip will do**: the same number formats, the same rounding, the same order of operations. Its outputs are the answer key. When the chip runs in simulation, every number it produces must equal the golden model's number, bit for bit. "Golden" is the usual chip-verification name for the trusted reference.

### Why the normal Qwen code isn't good enough as the answer key

The normal code (Hugging Face, PyTorch) computes in bf16 and fp32 floats. Our chip computes with 4-bit weights, int8 activations, integer sums inside each group, then fp32. So the two disagree slightly, even when the chip has no bugs.

One dot product of 4 values shows it:
- **Real weights:** 0.031, −0.018, 0.052, 0.009
- **Real activations:** 0.51, 1.02, −0.26, 0.19
- **The exact float answer** (what the normal code gets): 0.0158 − 0.0184 − 0.0135 + 0.0017 = **−0.0144**

What the chip does:
- Weight scale 0.01, so the weights round to the integers 3, −2, 5, 1.
- Activation scale 0.05, so the activations round to 10, 20, −5, 4.
- Integer sum: 30 − 40 − 25 + 4 = −31.
- Apply the scales: −31 × 0.01 × 0.05 = **−0.0155**

A correct chip says −0.0155 and the normal code says −0.0144. If we compared the chip against the normal code, we couldn't tell a bug from rounding. The golden model does the chip's rounding, so it says −0.0155 too, and any difference at all means a bug.

### Three versions of the model, two separate checks

The three versions:
- **The normal Qwen code:** the truth about the model.
- **The golden model:** the truth about our chip's arithmetic.
- **The RTL:** the hardware design.

The two checks:
- **Golden model vs normal code** asks: *how much quality do our formats cost?* Small differences are expected. We measure whether they're small enough, for example how often both pick the same next token, and the distance between their token probabilities. This is where we choose weight, activation and KV formats, by changing a few lines of Python before any hardware exists.
- **RTL vs golden model** asks: *is the hardware correct?* These must match exactly. No tolerance.

Exact matching needs one more rule. In fp32, the order of additions changes the last bits: (a + b) + c can differ slightly from a + (b + c). So the golden model adds things **in the order the hardware will,** for example group sums in the order the lanes produce them.

### What it looks like

A Python file of small functions, one per chip operation, each doing the chip's exact arithmetic:
- `matvec_w4a8(weight_codes, code_table, weight_scales, x_int8, x_scales)`: the matrix unit
- `rmsnorm(x, gain)`: the vector unit
- `deltanet_step(state, q, k, v, beta, decay)`: one DeltaNet update
- `attention(q, k_cache, v_cache)`: one attention read
- `run_layer`, `run_token`: those pieces chained in the model's order

That structure gives us two more things:
- **Test data for every hardware block.** The RTL for the matrix unit gets the same inputs as `matvec_w4a8`, and its outputs must match. That works long before the whole chip runs.
- **A first draft of the instruction set.** Each function is something the chip must be able to do. If the instruction set can't express one, we find out in Python, not in silicon.

### Three jobs, one model

- **Settles the quality decisions:** weight format, activation format, KV format, MTP acceptance.
- **Is the reference the RTL is checked against.**
- **Becomes the operation list the ISA must cover.**

## Does quantizing activations slow the chip down?

No. The work is tiny, and it's done once per vector, while each vector is reused against thousands of weight rows.

The numbers per token:
- **What gets quantized:** the input to each matrix multiply. Each layer has four: the mixer input (4,096 values), the mixer's output-projection input (4,096), the FFN input (4,096) and the FFN-down input (12,288). That's 24,576 values per layer.
- **Total:** 32 layers × 24,576 + 4,096 for the LM head, about **790,000 values per token.**
- **Work per value:** find the group's largest value, multiply by 1/scale, round. About 3 simple operations, so roughly **2.4 million per token.**
- **Compare with the matrix work,** 7.94 billion multiply-adds per token: about **0.03%**.

Why it's so cheap: the FFN's 4,096-value input is quantized once, then multiplied against 24,576 weight rows (gate plus up). Each rounding is paid once and used 24,576 times.

The time:
- At 64 values per cycle, 790,000 values take about 12,400 cycles, **about 12 µs per token** at 1 GHz.
- A token takes about 10 ms at 100 tok/s, so that's about 0.1%.
- It can hide almost entirely: quantizing a 4,096-value vector takes about 64 cycles, shorter than the wait for the next matrix's first weights to arrive from DRAM.

But int8 activations **don't make decode faster either.** At batch 1 the chip waits on memory and the multipliers are mostly idle. bf16 activations would run at the same speed.

The reason to consider int8 at all is **multiplier size and power**. An int4 × int8 multiplier is several times smaller than an int4 × bf16 one, and we need about 4,096 of them for prefill and verification. So **the trade-off is quality vs area, not speed.** If the golden model shows int8 hurts quality, or synthesis shows bf16 multipliers fit comfortably, bf16 activations is a fine choice.

## Where 4-bit weights come from

### No official 4-bit Qwen3.5-9B exists

Checked on Hugging Face, 2026-11-10:
- **From Qwen:** only the bf16 model (`Qwen/Qwen3.5-9B` and `-Base`), plus sparse-autoencoder research models. No official int4 or FP8.
- **From the community:**
  - llama.cpp files (unsloth, bartowski, lmstudio-community)
  - AWQ 4-bit (cyankiwi, QuantTrio)
  - NVFP4 (AxionML)
  - FP8 (RedHatAI)
  - MLX 4-bit (lmstudio-community, mlx-community)
- **None of these were trained in 4 bits.** They all start from the same bf16 weights and round them afterwards. As far as we know, they all use one of the after-training methods below. We haven't checked each one.

### The methods, simplest to best

- **Round to nearest:** split the weights into groups, pick a scale per group, round each weight to the nearest level. No data, no training. llama.cpp's basic Q4_0 does this, and it's what the example above did (0.031 → 3 × 0.01).
- **GPTQ (smarter rounding):** round the weights one at a time, and after each one, nudge the not-yet-rounded weights to cancel the error just introduced. If rounding 0.031 down to 0.03 makes the row's output slightly too small on typical inputs, the next weights get pushed slightly up. "Typical inputs" come from a few hundred sample texts (calibration data). No training.
- **AWQ (protect the important weights):** a few input channels carry much bigger activations than the rest, so errors in their weights matter more. AWQ multiplies those weights by a factor before rounding, giving them finer relative precision, and divides the matching activations by the same factor, so the maths is unchanged. Calibration data, no training.
- **Quantization-aware training (QAT):** fine-tune the model with rounding simulated during training, so it learns weights that survive rounding. The best quality, but it needs real training compute and data. Qwen hasn't released a QAT version of this model.

### What we plan to do

- **Run the methods ourselves, aimed at our format.** The community checkpoints probably don't match it. AWQ files usually use groups of 128 with a zero point, and ours is a scale per 32.
- **Order:** round-to-nearest as the baseline, then GPTQ and AWQ, measuring how much each closes the gap to bf16.
- **Cost:** GPTQ or AWQ on a 9B model is a calibration run on one GPU, not training. We'd guess around an hour, which we haven't checked. We have Beam GPU credits for this.
- **QAT stays a later option,** if 4-bit (or 3-bit FFN) quality isn't good enough.

### Weights and activations are rounded differently on purpose

- **Weights** are rounded once, offline, so slow, careful methods like GPTQ are affordable.
- **Activations** are rounded on the chip, live, every token, so that rounding must be simple: one scale per 32 values, round to nearest.

The design gives each one the method that fits when it happens.
