# 15. Measurements: golden model, number formats, long context, GPU speed

*Measured 2026-11-11 (UTC) on Beam GPUs. Status: evidence. Each number here comes from a JSON file in `golden/results/` written by the code in `golden/`. Section 5 says how to rerun everything.*

This note records what we measured, how, and what each result means for the design. Notes 10 to 12 contained guesses; this note replaces them with data where we have it.

## 0. Setup

- **Hardware:** NVIDIA GeForce RTX 5090 (33.7 GB) on Beam, 16 CPU cores and 68.7 GB host RAM for validation. It was the largest GPU Beam had available serverless. H100s needed a manual reservation.
- **Software:** PyTorch 2.8.0 with CUDA 12.8, transformers 5.22.2, Python 3.11.11.
- **Model:** `Qwen/Qwen3.5-9B` from Hugging Face, the published bf16 weights, text model only.
- **The golden model** (`golden/qwen35_golden.py`) is our own from-scratch implementation:
  - every operation is written out, in the same order as the layer map
  - all arithmetic is in fp32, with TF32 turned off
  - weights stay in their stored form and are expanded to fp32 just before each matrix multiply
  - number formats are pluggable (`golden/formats.py`): weights, matmul inputs, KV cache and DeltaNet state can each be rounded the way the chip would round them

## 1. Is the golden model correct? (`results/validate.json`)

**Method:**
- Run our golden model on the GPU and Hugging Face's Qwen3.5 implementation on the CPU, both in fp32, with eager attention, on the same weights.
- Two prompts: an English one (40 tokens) and a Python one (15 tokens).
- Compare the logits (the score for each of the 248,320 tokens), the hidden state after every layer, and 32 tokens of greedy generation.

**Results (English prompt, Python prompt):**
- **Largest logit difference:** 1.96e-5, 1.05e-5. Typical logits are about 2 in size, and the largest about 23.
- **Mean logit difference:** 1.9e-6, 1.1e-6.
- **KL divergence** between the two predicted distributions: at most 2.6e-7.
- **Top-1 and top-5 predictions:** identical at every position (100%).
- **32 greedy tokens:** identical on both prompts. For example, both continue the English prompt with "that it enables efficient long-context modeling without the quadratic memory and compute overhead of standard self-attention."
- **Hidden state after each layer:** the relative error grows smoothly from 1.6e-7 to 8.7e-7 across layers 1 to 31, which is fp32 rounding noise. No layer stands out, so no single operation is wrong. The 32nd entry differs because Hugging Face records it *after* the final norm and we record it before. The logit agreement shows that last step matches too.
- **Prefill vs decode paths:** running a prompt all at once and running it one token at a time agree to 2.1e-5. These are the two ways the chip will run the model (prefill and decode), so both paths are checked.
- **Cost:** golden model GPU memory 19.2 GB. One token in decode takes 49 ms in our unoptimized Python loop. Hugging Face on 16 CPU cores takes about 2 s per generated token.

**Conclusion:** the golden model computes exactly what Qwen3.5-9B computes. Every later result can blame its differences on the number format being tested, not on our code.

## 2. Which weight format? (`results/quality.json`, `results/quality2.json`)

**Method:**
- **Reference:** the golden model with the published bf16 weights, everything else fp32.
- **Text:** 2,048 tokens of WikiText-2 test (English prose), and 2,048 tokens of Python standard-library source (code).
- **Metrics, each compared with the reference on the same tokens:**
  - **KL divergence (mean):** how different the predicted distributions are. 0 means identical. This is the main metric, because perplexity can hide changes that cancel out.
  - **±:** standard error of the mean over tokens.
  - **Top-1 agreement:** how often the most likely next token is the same.
  - **Top-5 overlap:** how many of the reference's top 5 tokens are also in the candidate's top 5.
  - **Perplexity change:** reference perplexity is 6.918 for prose and 1.473 for code.
  - **Weight error:** relative RMS error of the stored weights, ‖W − Ŵ‖ ÷ ‖W‖.
- **Rounding:** no calibration data is used anywhere. The quantizers only look at the weights.

### Round 1: plain round-to-nearest

Each entry gives bits per weight, then prose KL / top-1 / perplexity change, then code KL / top-1 / perplexity change.
- **int4, fp16 scale per 32 (our default), 4.5 bits:** prose 0.127 ± 0.018 / 89.7% / −0.1%. Code 0.153 ± 0.010 / 91.1% / **+19.9%**.
- **int4, scale per 128, 4.12 bits:** prose 0.170 / 87.3% / +9.7%. Code 0.246 / 88.6% / +32.8%.
- **MXFP4, 4.25 bits:** prose 0.150 / 86.5% / +6.4%. Code 0.174 / 90.0% / +23.9%.
- **NVFP4, 4.5 bits:** prose **0.086** / 89.7% / +4.4%. Code **0.110** / 92.2% / +13.4%.
- **FFN 3-bit, rest 4-bit, 3.89 bits:** prose 0.252 / 82.5% / +15.7%. Code 0.374 / 83.6% / +56.5%.
- **3-bit everywhere, 3.5 bits:** prose 0.520 / 72.9% / +52.8%. Code 0.707 / 74.5% / +125%.
- **int4 with the LM head left in bf16, 6.16 GB per token:** prose 0.120. Code 0.149 / +19.1%. A tiny gain, and speed on our card drops from 100 to 76 tok/s.

### Other rounding, applied on top of int4 weights with a scale per 32

Compare each against int4 alone: prose 0.127, code 0.153.
- **int8 activations, scale per 32:** prose 0.126, code 0.154.
- **int8 KV cache, scale per 64:** prose 0.127, code 0.153.
- **int4 KV cache, scale per 32:** prose 0.137, code 0.159.
- **bf16 DeltaNet state:** prose 0.126, code 0.154.
- **The chip candidate (int4 weights, int8 activations, int8 KV, fp32 state):** prose 0.129, perplexity +0.15%. Code 0.154, +20.0%.

At 2K tokens, all of these are within one standard error of int4 weights alone. **The weight format is what matters.**

### Round 2: better rounding at the same 4.5 bits

- **int4 with the scale searched to minimize each group's error:** weight error 8.8% (was 9.9%). Prose 0.099, code 0.145.
- **int4 with NVFP4's scale scheme** (groups of 16, FP8 scale): weight error 8.8%. Prose 0.084, code 0.117.
- **NVFP4 with scale search:** weight error 8.8%. Prose **0.076**, code 0.105.
- **A 16-level table learned per weight matrix (k-means), groups of 32, fp16 scale:** weight error 8.3%. Prose 0.094, code 0.105.
- **Learned table plus scale search:** weight error **8.1%**. Prose 0.093, code **0.101**, top-1 92.7%.
- **The NVFP4 anchor reproduced round 1 exactly,** so the setup is deterministic.

### What the format results show

1. **Code is much more sensitive than prose.** The model is very sure of code (perplexity 1.47), so the same weight error costs 13 to 20% perplexity on code but only a few percent on prose. We have to judge formats on code too, not just prose.
2. **At 4.5 bits, how you round matters about 1.5× in KL.** Code KL goes from 0.153 (plain int4) to 0.101 (learned table plus scale search). Small groups with FP8 scales (NVFP4-style) and learned levels both help.
3. **Weight-only tricks plateau.** The best four formats all land at code KL 0.100 to 0.105 and about +13 to +14% code perplexity. Lowering weight error further (8.8% → 8.1%) barely moves the output. Doing better needs rounding that uses the activations, as GPTQ and AWQ do, or quantization-aware training. Those are software changes that produce the same codes and scales, so **the hardware doesn't need to wait for them.**
4. **3-bit is ruled out** with this kind of rounding. The "fewer bits" lever in note 07 is closed unless smarter methods rescue it.
5. **int8 activations cost nothing measurable.** That confirms the int8 × int8 multiply in the PE (note 10).

**What this means for the hardware:**
- **Keep the 16-entry lookup table in the datapath.** int4, E2M1 (FP4) and learned tables all run on it, and the best result used a learned table.
- **Support both scale schemes,** since they tie at 4.5 bits: groups of 32 with an fp16 scale, and groups of 16 with an FP8 (E4M3) scale. The scale unit then handles up to 64 groups per cycle instead of 32. That's cheap.

## 3. KV cache and DeltaNet state over a longer context (`results/long.json`)

**Method:**
- 8,192 tokens of WikiText-2 test. Reference perplexity 8.556.
- KL is also reported by position: tokens 0–1K, 1–2K, 2–4K and 4–8K.
- Two setups change **only** the KV cache or the state, with bf16 weights, to see each effect without weight rounding mixed in.

**Results (KL overall, then by position):**
- **int4 weights alone:** 0.128. By position: 0.126, 0.128, 0.096, 0.144. Perplexity +3.0%.
- **+ int8 KV:** 0.128, identical to the fourth decimal at every position. **No measurable cost.**
- **+ int4 KV:** 0.135. By position: 0.131, 0.142, 0.101, 0.151.
- **int4 KV only, bf16 weights:** 0.015, top-1 96.2%. By position: **0.012, 0.016, 0.008, 0.020**. Perplexity +0.09% (log-loss change 0.0009 ± 0.0033, not significant).
- **bf16 state only, bf16 weights:** 0.013, top-1 97.6%. By position: **0.0028, 0.0020, 0.0025, 0.0237**. Perplexity **+1.40%** (log-loss change 0.0139 ± 0.0024, about 6 standard errors).

**What this shows:**
- **A bf16 DeltaNet state is harmful, and the harm builds up with length:** about 9× worse by the 4–8K range. The state is updated every token, so each rounding error carries forward. At 1M context this would be much worse. **Decision: fp32 state (D-005), now with evidence.** It costs 1.1% speed.
- **int8 KV is free at 8K. Default.**
- **int4 KV costs a little, and grows slowly with position.** It stays the opt-in long-context mode (note 11). Its error trend has to be measured at much longer context before we promise 1M quality with it.

## 4. Real GPU speed for Qwen3.5-9B (`results/bench.json`)

**Method:**
- **Software:** llama.cpp at commit `aa75ff32`, built with CUDA 12.8. The published bf16 weights were converted with llama.cpp's own `convert_hf_to_gguf.py`, then quantized to Q4_0 with `llama-quantize`.
- **Test:** `llama-bench` with all layers on the GPU and flash attention on. Each test runs 3 times.
  - **Prefill:** 512 prompt tokens.
  - **Decode:** 128 generated tokens at batch 1.
  - Both are run at context depths of 0, 4,096 and 32,768 tokens.
- **What the GPU reads per generated token,** taken from the file's real tensor table:
  - all weights except the embedding table (only one row of it is read): **4.867 GB**
  - DeltaNet state in fp32, read and written: 0.10 GB
  - f16 KV cache: 32 KB per token of context
- **The file is 5.45 GB, not the 5.03 GB we assumed.** llama.cpp stores the LM head as Q6_K (6.56 bits, 0.834 GB) instead of Q4_0. So llama.cpp reads about 5.10 GB per token at 4K context, more than the 4.70 GB we used in note 07.
- **Utilization** = tokens per second × bytes per token ÷ peak memory bandwidth from the spec sheet.

**Results** (decode tok/s ± standard deviation, then utilization, at depth 0 / 4K / 32K):
- **RTX 4090** (1,008 GB/s):
  - 154.5 ± 0.02 / 151.4 ± 0.3 / 130.6 ± 0.2 tok/s
  - utilization **76.2% / 76.7% / 78.3%**
  - prefill: about 11,100 / 10,800 / 8,500 tok/s
- **RTX 5090** (1,792 GB/s):
  - 233.3 ± 8.5 / 224.0 ± 0.5 / 196.5 ± 0.7 tok/s
  - utilization **64.7% / 63.8% / 66.3%**
  - prefill: about 11,500 / 12,800 / 8,800 tok/s
- **The A10G run did not happen.** Beam placed the job labelled "A10G" on an RTX 4090 (A10G capacity was low), as `nvidia-smi` shows in `results/bench.json`. That run gives 154.8 tok/s at depth 0, within 0.2% of the RTX 4090 run, so it serves as a repeat measurement. Its "utilization" field is wrong (it used the A10G's 600 GB/s) and should be ignored.

**What this shows:**
1. **Qwen3.5-9B uses GPU bandwidth less well than Llama 2 does.** We assumed 86%, from Llama 2 7B on an RTX 5060. We measured 76–78% on the RTX 4090 and 64–66% on the RTX 5090. The faster GPU uses its bandwidth less fully, probably because fixed per-token overheads (kernel launches, the 24 small DeltaNet steps, sampling) matter more when each token takes only about 4 ms.
2. **Re-estimating the RTX 5060** (448 GB/s, 5.10 GB per token at 4K):
   - at the 4090's 76.7%: **67 tok/s, 0.224 tok/s per $** at $300
   - at the 5090's 63.8%: 56 tok/s, 0.187 tok/s per $
   - note 07 estimated 82 tok/s and 0.273
   - So the bar is probably **lower** than D-010 assumed. We still keep 0.273 as the minimum goal until we measure a real RTX 5060. A conservative bar can't be accused of being too easy.
3. **Our 92% utilization target would beat every GPU we measured** by 14 to 28 percentage points. Utilization is a real lever, not a rounding error.
4. **Context costs GPUs less than the byte count predicts.** From depth 0 to 32K the bytes per token grow 22%, but tok/s drops only 15% (RTX 4090), so utilization goes *up* with depth. Our guess (not measured): reading the KV cache streams more smoothly than reading many small weight matrices.
5. **Prefill on GPUs is huge** (about 11,000 tok/s) next to our floor of 2,048 tokens in 4 s (512 tok/s). That's expected: GPUs carry many times our compute. We decided prefill is a floor for us, not a race (D-007).

## 5. How to rerun

From `golden/`, with the Beam SDK's Python:
```
BEAM=~/.local/share/uv/tools/beam-client/bin/python
$BEAM beam_app.py validate     # section 1, about 6 min
$BEAM beam_app.py quality      # section 2 round 1, about 5 min
$BEAM beam_app.py quality2     # section 2 round 2
$BEAM beam_app.py long         # section 3
$BEAM beam_bench.py gguf       # build llama.cpp, convert to Q4_0
$BEAM beam_bench.py bench      # section 4
$BEAM beam_exp2.py sweep       # section 7
$BEAM beam_exp2.py gptq_a      # section 8 (also gptq_b ... gptq_e; one GPU each, about 10 min per config)
$BEAM beam_exp2.py long32k     # section 9
$BEAM beam_exp3.py smoke bf16  # 8 problems per task, batched-vs-single check
$BEAM beam_exp3.py tables_a    # section 12 (int8 entries + control), about 20 min
$BEAM beam_exp3.py tables_b    # section 12 (E4M3, int6, int5), about 30 min
$BEAM beam_exp3.py tasks bf16  # section 13 (also: tasks rtn, tasks winner), 45-55 min each
python3 analyze_exp3.py        # prints every number in sections 12-13 (no torch needed)
```
Beam notes: jobs that wait in Beam's queue for more than about 10 to 20 minutes lose their inputs and fail, which is why GPTQ configs run in batches and accept any of RTX 5090, RTX 4090 or A10G. The account allows about 5 GPUs and 24 CPUs at once.
The weights are cached on the Beam volume `qwen35-weights`.

## 6. Limits of these measurements

- **2K and 8K tokens are short** next to the 262K native and 1M extended context. The state and KV trends are clear, but how they continue past 8K is extrapolation.
- **Two text domains.** Chat, maths and multilingual text aren't tested. No task benchmarks yet (MMLU-style, HumanEval-style).
- **No calibration-based quantizers yet** (GPTQ, AWQ). These are the obvious next test, and they would only improve the numbers above.
- **One run per setup.** The quantizers are deterministic. The k-means table uses a random sample, so it could vary slightly between runs. The ± figures are per-token standard errors, not run-to-run spread.

---

# Experiment set 2 (2026-11-11): the best setting for every number format

*Code: `golden/gptq.py`, `golden/formats.py`, `golden/beam_exp2.py`. Raw data: `golden/results/exp2_*.json`. GPUs: RTX 5090 and RTX 4090 on Beam (each file records which). Same evaluation text and metrics as sections 2 and 3: 2,048 tokens of English prose and 2,048 of Python code, compared with full precision.*

## 7. Where does rounding error come from? (`exp2_sweep.json`)

**Method:** round one part of the model at a time and keep everything else at full precision. Plain rounding, no calibration.

**4-bit (NVFP4) on one part only.** Code KL, perplexity change on code, and the part's share of bytes read per token:
- **FFN:** 0.054, +6.1%, 61% of bytes
- **DeltaNet projections:** 0.037, +4.1%, 20% of bytes. **About 2× more sensitive per byte than the FFN.**
- **Attention projections:** 0.014, +1.6%, 6% of bytes
- **LM head:** **0.003**, +0.1%, 13% of bytes. Very robust.
- **Embedding:** 0.007, +1.7%. Only one row is read per token.

**3-bit (learned 8-level table) on one part only:**
- **FFN:** 0.213, +29%
- **DeltaNet:** **0.442, +66%.** DeltaNet cannot go to 3 bits.
- **Attention:** 0.120, +14%
- **LM head:** 0.011, **+1.4%.** The LM head tolerates 3 bits.
- **Embedding:** 0.050, +7.6%

**3-bit FFN by depth, one quarter of the layers at a time:**
- layers 0–7: code KL 0.066, +7.3%
- layers 8–15: 0.063, +8.2%
- layers 16–23: 0.050, +7.2%
- layers 24–31: **0.035, +3.6%**

Later layers tolerate fewer bits better.

**Activation formats alone,** with full-precision weights (code KL, then code perplexity change):
- **int8, scale per 32:** 0.0010, +0.1%
- **int8, scale per 128:** 0.0035, +0.3%
- **int8, one scale per whole vector:** **0.105, +13.9%.** Bad.
- **8-bit float, scale per 32:** 0.0040, +0.6%

**3-bit everywhere, plain rounding:**
- learned 8-level table, scale per 32: code KL 1.43, +359%
- 3-bit integers, 8-bit-float scale per 16: 0.53, +86%

At 3 bits, small groups with 8-bit-float scales matter much more than learned levels.

## 8. GPTQ: calibrated rounding (`exp2_gptq_*.json`)

**Method:**
- GPTQ (Frantar et al. 2022), standard algorithm: blocks of 128 columns, 1% damping, each group's scale set when its first column is reached.
- **Calibration:** 64 passages of 512 tokens. Half come from WikiText-2 *train*, half from Python standard-library files *not* used in the evaluation.
- The model is processed one layer at a time, and each layer is fed the already-quantized previous layer's outputs.
- The LM head also gets GPTQ. The embedding uses plain rounding, because it's a lookup with no inputs to calibrate on.
- Each result is scored twice: weights only, and with int8 activations (scale per 32) and int8 KV (scale per 64) added. The second is "chip".

**Results, all at 4.5 bits per weight** (code KL, code perplexity change, code top-1; then prose KL):
- **GPTQ int4, scale per 32:** 0.108, +12.2%, 92.6%; prose 0.048. Plain rounding was 0.153 and 0.127.
- **GPTQ NVFP4:** 0.086, +11.3%, 93.4%; prose 0.058. Plain was 0.110 and 0.086.
- **GPTQ learned table, scale per 32:** 0.067, +8.8%, 93.9%; prose 0.065. Plain was 0.105 and 0.094.
- **The same, with the embedding at 8 bits:** 0.060, +7.9%, 94.6%; prose 0.057.
- **The same, plus a 3-bit LM head (4.37 bits on average, 103.0 instead of 100.2 tok/s):** 0.065, +8.2%; prose 0.065.
- **GPTQ learned table with an 8-bit-float (E4M3) scale per 16, embedding at 8 bits:** **0.050, +6.1%, 95.2%; prose 0.052.** ← the best result
- **"Chip" setting on the best result:** code 0.051, +6.4%; prose 0.049.

**3-bit with GPTQ** (code KL, code perplexity change):
- **FFN 3-bit learned + NVFP4 elsewhere (3.89 bits):** 0.177, +24.6%
- **FFN 3-bit integers + NVFP4 elsewhere:** 0.236, +30.9%
- **3-bit learned everywhere (3.5 bits):** 0.320, +46.8%. GPTQ rescued it from +359%, but not far enough.

**Robustness checks:**
- **The winner (table, scale per 32, 8-bit embedding) rerun with a re-sampled table:** code 0.0567 vs 0.0597, prose 0.053 vs 0.057. **So differences below about 0.004 KL are run-to-run noise.** This rerun also used a different GPU type, which is included in that noise.
- **128 calibration passages instead of 64:** code 0.0564 vs 0.0567. **No gain, so 64 passages are enough.**
- **The best result's lead over the next one** (0.050 vs 0.057 on code) is larger than that noise.

**What this shows:**
1. **GPTQ cuts the output error by 20 to 60%** at the same bits, for every format. As predicted, it *raises* weight error. With the learned table, the FFN weight error went from 8.3% to 10.6%, while the output error fell.
2. **GPTQ, a learned table, small groups with 8-bit-float scales, and an 8-bit embedding each help, and the gains add up:**
   - plain int4: code KL 0.153, perplexity +20%
   - all four combined: **0.050, +6.1%**

   That's **3× less error with the same bytes and the same speed.**
3. **3-bit is still not usable,** even with GPTQ. The cheapest 3-bit option costs +25% on code. The one exception is the LM head, which can go to 3 bits for 2.8% more speed.
4. **int8 activations and int8 KV add almost nothing** on top of GPTQ weights. On code the change is 0.001 or less. On prose it's within noise.

## 9. KV cache and state at 32K context (`exp2_long32k.json`)

**Method:**
- 32,768 tokens of WikiText-2 test, scored at every 8th position (4,096 positions).
- Weights at full precision, so only the KV cache or state is being tested.
- Prefill runs in 4K-token pieces, carrying the cache and state between pieces. That's the same maths as one big pass, and the same way the chip will do it.

**Results** (KL overall, top-1, perplexity change; then KL at 0–2K / 2–8K / 8–16K / 16–32K):
- **KV int8, scale per 64:** 0.0002, 99.6%, +0.04%; 0.0000 / 0.0001 / 0.0007 / 0.0001
- **KV int8, one scale per 256-value head:** 0.0003, 99.6%, −0.15%; 0.0001 / 0.0002 / 0.0007 / 0.0003
- **KV 8-bit float, one scale per head:** 0.0019, 98.8%; 0.0006 / 0.0032 / 0.0032 / 0.0009
- **KV int4, scale per 32:** 0.020, 96.3%; 0.020 / 0.012 / 0.034 / 0.015
- **KV int4, scale per 64:** 0.032, 95.1%; 0.010 / 0.016 / 0.047 / 0.033
- **DeltaNet state bf16:** 0.034, 96.4%, **+3.36%** (log-loss change 0.033 ± 0.005, about 7 standard errors); 0.0006 / 0.0031 / **0.067** / 0.032

**What this shows:**
- **int8 KV costs nothing, even with one scale per head.** That's 4× fewer KV scales than one per 64.
- **int4 KV costs a small, roughly constant amount** out to 32K. It doesn't grow steadily. A scale per 32 beats one per 64.
- **The bf16 state's damage keeps growing with length:** perplexity +1.4% at 8K, +3.4% at 32K. At 262K or 1M tokens it would be much worse. **fp32 state is required.**

## 10. Final recommendation for every number format

Each item gives the setting, then the evidence for it.

- **Weights (all matrices that are read every token):**
  - **Format:** a 4-bit code, a 16-entry lookup table learned per matrix, an 8-bit-float (E4M3) scale per 16 weights, and one fp32 scale per matrix. **4.5 bits.**
  - **Rounding:** GPTQ with 64 × 512 calibration tokens.
  - **Evidence:** section 8. Code KL 0.050, perplexity +6.1%, top-1 95.2%, at the same speed as plain int4 (100 tok/s on our card).
- **Embedding table:** int8 with a 16-bit scale per 32 values. It costs no speed and 0.57 GB more memory. **Evidence:** section 8 (0.067 → 0.060 on code).
- **LM head:** the same 4-bit format by default. **Optional 3-bit mode,** which is 2.8% faster for about +0.005 KL. **Evidence:** sections 7 and 8.
- **No 3-bit anywhere else.** **Evidence:** section 8, where the best 3-bit option costs +25% on code.
- **Activations (matmul inputs):** int8 with a scale per 32. The scale per 128 is also fine; the cost difference is tiny either way, since activations are on chip. One scale per vector is not acceptable. **Evidence:** section 7.
- **Accumulation:** an integer sum within each group, fp32 across groups.
- **KV cache:** int8 with **one scale per head** (256 values). int4 with a scale per 32 is an opt-in mode for very long context. **Evidence:** section 9.
- **DeltaNet state:** **fp32.** **Evidence:** sections 3 and 9.

**What this means for the hardware:**
- **The weight path:** a 4-bit code goes through a 16-entry table to a value that the PE multiplies. Groups of 16 mean the scale unit handles 1,024 ÷ 16 = **64 scales per cycle** per token row.
- **The 8-bit-float scale needs a small E4M3-to-fp32 converter,** plus one fp32 multiply per matrix for its overall scale.
- **The table is loaded per matrix,** so the instruction set needs a "load table" step before each matrix.
- **Table entries are int8** (each level stored as k/127). This keeps the PE an 8-bit × 8-bit integer multiplier. Measured free in section 12.

## 11. Limits that still apply

- **Two text domains, 2K tokens each, for weight formats.** Task benchmarks were added in section 13.
- **KV and state measured to 32K,** not the 262K native or 1M extended context.
- **Table entries:** resolved in section 12 (int8 is free).
- **GPTQ results vary by about ±0.004 KL between runs,** including a change of GPU type.

---

# Experiment set 3 (2026-11-12): the last two checks before freezing the PE

*Code: `golden/beam_exp3.py`, summary script `golden/analyze_exp3.py`. Raw data: `golden/results/exp3_*.json`.*

Two questions were left open by section 10:
- **Table entries.** Each matrix has 16 learned levels. The software kept them in fp32, but the chip has to store them in a few bits. How few is safe?
- **Real tasks.** KL and perplexity are proxies. Does the quantized model still solve maths problems and write code that passes tests?

## 12. How many bits do the 16 table levels need? (`exp3_gptq_*.json`)

**Setup:**
- The winning setup from section 8: GPTQ, learned 4-bit table, E4M3 scale per 16, int8 embedding.
- The only change is how the 16 levels of each matrix's table are stored. The table spans −1 to +1, so "int8" means each level is rounded to the nearest k/127.
- Same evaluation as section 8: 2,048 tokens of Python and 2,048 of WikiText-2.
- A **control rerun** with fp32 levels ran in the same batch, to show the run-to-run spread on another GPU.

**Results on Python code** (the precise metric; ± is the per-token standard error):
- **fp32 levels (section 8, RTX 4090):** KL 0.0500 ±0.0037, top-1 95.2%, perplexity +6.1%
- **fp32 levels, control rerun (RTX 5090):** KL 0.0516 ±0.0036, top-1 94.3%, +6.8%
- **int8 levels:** KL **0.0500** ±0.0038, top-1 95.6%, +7.1%
- **int6 levels:** KL 0.0504 ±0.0037, top-1 95.4%, +7.0%
- **E4M3 levels:** KL 0.0518 ±0.0037, top-1 94.5%, +5.8%
- **int5 levels:** KL **0.0638** ±0.0045, top-1 94.3%, +7.3%. That's +0.014 KL, about 3.5× the run-to-run spread, so this is where it breaks.

**Results on WikiText-2:**
- KL ranges 0.038 to 0.065 across all six runs, including the two fp32 controls (0.052 and 0.051).
- WikiText's error is ±0.013–0.015 per run, so the prose set can't separate these options.
- Perplexity change ranges +1.8% to +4.1%, again inside the noise.

**With the full chip pipeline** (int8 activations with a scale per 32, int8 KV) on top:
- code KL is 0.051 for int8 levels
- that adds 0.001, as before

**Why int8 is enough** (measured over all 249 learned tables):
- The smallest gap between two neighbouring levels in any fp32 table is **0.077**. The median gap is 0.121.
- int8 places each level to within half a step: 1/254 ≈ 0.004. That's about 5% of even the tightest gap, and no two levels ever merged (all 249 tables kept 16 distinct levels).
- int5's step is 1/15 = 0.067. That's about the size of the tightest gap, so every table is forced close to a uniform grid. A uniform grid is exactly plain int4, which explains why int5 loses quality.
- Table stats weren't recorded for the E4M3 and int5 runs, because the code that finds learned tables skips tables whose first entry is a whole number such as −1. The quality numbers above are unaffected.

**Decision: int8 table entries.**
- Quality is the same as fp32, with 3 bits of margin, since int6 also passes.
- The PE stays an **int8 × int8 integer multiplier.** E4M3 entries pass too, but they'd need an fp8 × int8 multiplier for no gain.
- Table storage per matrix is 16 bytes. Loading a table per matrix is 16 B × about 250 matrices ≈ 4 KB per token, which is negligible.

## 13. Task benchmarks: maths and code that actually runs (`exp3_tasks_*.json`)

**The three models:**
- **bf16:** the published weights.
- **Plain int4:** int4 with an fp16 scale per 32 and round-to-nearest. This is the same scheme as llama.cpp's Q4_0, our GPU comparison point.
- **Chip format:** the section 12 decision, i.e. GPTQ + learned table with **int8 entries** + E4M3 scale per 16 + int8 embedding. These are exactly the weight values the chip would multiply.

**The tasks** (all problems in each set, greedy decoding):
- **GSM8K:** all 1,319 test problems. 8-shot chain of thought; the examples are the first 8 training problems with the calculator annotations removed. The answer is taken from "The answer is N" (or the last number) and compared exactly.
- **HumanEval:** all 164 problems. The model completes the function, stopping at the usual markers (`\ndef`, `\nclass`, `\nif __name__`, `\nprint(`, `\n#`, `\nassert`). **The code is run** against the official tests, with a 15 s timeout.
- **MBPP:** test problems 11–510 (500). The standard 3-shot prompt uses problems 2–4 as examples and stops at `[DONE]`. **The code is run** against the three assert tests.

**How they run:**
- Hugging Face `generate` in bf16. Prompts go in batches of at most about 8,192 tokens, with left padding.
- Only the weights differ between the three models. Quantized weights are expanded back to bf16, which adds about 0.2% relative rounding on top of a ~10% quantization error.
- Activations and KV are bf16 here; int8 for both was shown in section 12 to add only about 0.001 KL.
- **Batching check:** on 4 problems per task, batched and one-at-a-time generation gave byte-identical text (`smoke3_tasks_bf16.json`).
- The smoke test's failures were all read by hand. Each was a real model mistake (for example, a triangular-prism volume missing its ÷2), not a scoring bug.

**Accuracy** (± is the binomial standard error):
- **bf16:** GSM8K 1,178/1,319 = **89.31%** ±0.85; HumanEval 114/164 = **69.51%** ±3.59; MBPP 279/500 = **55.80%** ±2.22
- **Plain int4:** GSM8K 1,079 = **81.80%** ±1.06; HumanEval 108 = **65.85%** ±3.70; MBPP 271 = **54.20%** ±2.23
- **Chip format:** GSM8K 1,153 = **87.41%** ±0.91; HumanEval 116 = **70.73%** ±3.55; MBPP 277 = **55.40%** ±2.22

**Paired comparisons:**
- All three models answered the same problems, so each pair is compared problem by problem.
- "Lost" means the first model solved it and the second didn't; "gained" is the reverse.
- The p-value is an exact two-sided sign test on the lost and gained counts (McNemar's test).

**Plain int4 vs bf16:**
- **GSM8K: −7.51 points ±0.96.** Lost 133, gained 34, p = 5×10⁻¹⁵. **A clear, real loss.**
- HumanEval: −3.66 ±3.45. Lost 19, gained 13, p = 0.38.
- MBPP: −1.60 ±1.86. Lost 47, gained 39, p = 0.45.

**Chip format vs bf16:**
- **GSM8K: −1.90 points ±0.76.** Lost 63, gained 38, p = 0.017. **A small but real loss.**
- HumanEval: +1.22 ±2.73. Lost 9, gained 11, p = 0.82. No detectable change.
- MBPP: −0.40 ±1.44. Lost 27, gained 25, p = 0.89. No detectable change.

**Chip format vs plain int4:**
- **GSM8K: +5.61 points ±1.00.** Gained 126, lost 52, p < 10⁻⁷.
- HumanEval: +4.88 ±3.33. p = 0.20.
- MBPP: +1.20 ±1.86. p = 0.59.

**Other measurements:**
- **Byte-identical output to bf16:** chip format 23% (GSM8K), 40% (HumanEval) and 35% (MBPP); plain int4 9%, 22% and 20%. Long answers drift apart even when both reach the same result.
- **Weight error barely differs** (relative RMSE): the chip format is 0.097 for the FFN, 0.100 for DeltaNet, 0.105 for attention and 0.092 for the LM head. Plain int4 is 0.099. But GPTQ aims its error where the inputs don't look, so the outputs and tasks come out far better.
- **Answers without a final-answer line on GSM8K:** 3 for bf16, 3 for plain int4, 2 for the chip format. The scorer isn't biased by one model rambling.
- **Runtime:** about 45 to 55 minutes per model on one GPU. The bf16 run was on an RTX 5090, the other two on RTX 4090s. GPU type can change bf16 rounding inside the matmuls slightly, which affects the byte-identical rates but not accuracy in any material way.

**What the tasks show:**
- **Maths is the most sensitive task.** Multi-step arithmetic compounds small errors.
  - Plain int4 (what a GPU running Q4_0 does) loses **7.5 points**.
  - The chip format loses **1.9 points**, recovering three-quarters of that loss at the same 4.5 bits and 100 tok/s.
- **Code** (HumanEval and MBPP, 664 problems with real execution) shows **no detectable loss** for the chip format.
- **This matches the KL ranking** from sections 2 and 8. KL was a reliable proxy, and the order of the formats doesn't change.

## 14. Final status of the number formats

- **D-005 is complete:** every setting now has a quality measurement, a speed figure and a hardware cost.
- **Changed by experiment set 3:**
  - learned table entries are **int8**
  - the expected task cost is **−1.9 points on GSM8K maths and no measurable change on code**
- **If the remaining maths gap matters later**, these options don't change the hardware:
  - a quantization-aware finetune (QAT) into the same format
  - 5- or 6-bit codes for the most sensitive matrices (DeltaNet was 2× more sensitive per byte in section 7)

  The first costs no bytes; the second costs a few percent of speed. Both stay in the open questions.
- **Still not measured:**
  - KV and state beyond 32K
  - chat and multilingual quality
  - a real RTX 5060
  - the MTP acceptance rate

# Experiment set 4: speculative decoding, 128K context, the one-pass DeltaNet order

*Measured 2026-11-18 (UTC) on Beam RTX 4090 and RTX 5090. Code: `golden/beam_exp4.py` and `golden/mtp.py`. Summary script: `python3 golden/analyze_exp4.py`. Rerun from `golden/` with the Beam Python: `beam_exp4.py mtp bf16`, `beam_exp4.py mtp chip`, `beam_exp4.py long128k`, `beam_exp4.py onepass32k`.*

## 15. MTP draft acceptance (`exp4_mtp_bf16.json`, `exp4_mtp_chip.json`)

**What was measured:**
- How often Qwen's own draft head (the MTP layer) guesses the next tokens right, for draft 1, 2 and 3 in a chain.
- **Text:** what the model itself writes, greedily:
  - chat: 100 Alpaca instructions, 256 tokens each
  - maths: 100 GSM8K problems
  - code: all 164 HumanEval problems, plus 100 MBPP problems
  - The task completions are the bf16 outputs from experiment set 3.
- **Scoring:** a position counts only if the next 3 tokens are exactly what the main model would emit. Then teacher-forced scoring equals real speculative decoding. The rest are skipped and counted.
- **Draft vocabularies:** the full 248,320 tokens, or the top 64K, 32K or 16K by frequency. Frequencies come from 4.76M tokens of WikiText, code, GSM8K, MBPP and Alpaca training text, none of it the test text.
- **Two conditions:**
  - the published bf16 weights
  - the exact chip format: main model GPTQ 4.5-bit, draft head 4.5-bit (round to nearest), int8 activations and int8 KV

**Wiring check:** the checkpoint doesn't say how the draft head's two inputs are joined, so it was measured, on 4,645 positions.
- Embedding first, main hidden state after the final norm: draft 1 hits **0.873**.
- Embedding first, before the final norm: 0.858.
- Hidden state first: 0.001, so that wiring is wrong.

**Measures:**
- **p_cond[j]:** the chance draft j + 1 is right, given drafts 1..j were right.
- **Tokens per step:** 1 + p₁ + p₁p₂ + … for k drafts. It counts the bonus token the main model always adds.

**Results, bf16 weights (tokens per step at k = 3 ± stderr; p_cond for drafts 1, 2, 3):**
- **chat** (20,554 of 21,063 positions valid)
  - full: 0.833, 0.747, 0.710 → 2.897 ±0.008
  - 64K: 2.808. 32K: 0.783, 0.707, 0.672 → 2.709. 16K: 2.542.
- **GSM8K** (8,144 of 8,290)
  - full: 0.957, 0.907, 0.849 → 3.561 ±0.009
  - 64K: 3.559. 32K: 3.547. 16K: 3.505.
- **HumanEval** (9,377 of 9,617)
  - full: 0.959, 0.924, 0.882 → 3.626 ±0.008
  - 64K: 3.572. 32K: 0.932, 0.900, 0.862 → 3.493. 16K: 3.311.
- **MBPP** (4,043 of 4,105)
  - full: 0.912, 0.851, 0.811 → 3.316 ±0.016
  - 64K: 3.263. 32K: 3.193. 16K: 3.041.

**Results, chip format (same layout):**
- **chat** (16,531 positions)
  - full: 0.865, 0.771, 0.711 → 3.007 ±0.009
  - 64K: 2.910. 32K: 0.813, 0.729, 0.671 → 2.804. 16K: 2.630.
- **GSM8K** (7,814)
  - full: 0.961, 0.907, 0.842 → 3.566
  - 64K: 3.564. 32K: 3.551. 16K: 3.509.
- **HumanEval** (9,022)
  - full: 0.965, 0.924, 0.873 → 3.634
  - 64K: 3.583. 32K: 3.503. 16K: 3.322.
- **MBPP** (3,755)
  - full: 0.919, 0.846, 0.790 → 3.312
  - 64K: 3.254. 32K: 3.191. 16K: 3.041.
- The draft head's 4.5-bit weights have a relative RMSE of 0.080 on average, at most 0.105.

**Mean of the four sets, chip format:**
- full vocabulary: p_cond 0.928, 0.862, 0.804 → 3.380 tokens per step
- 64K: 3.328
- **32K: 0.900, 0.841, 0.782 → 3.262**
- 16K: 3.126

**What this shows:**
- **Acceptance is much higher than the 0.6–0.7 that D-013 assumed.** Maths and code reach about 3.5 tokens per step at k = 3, and chat about 2.8.
- **The chip format doesn't hurt drafting.** Chip minus bf16 is within ±0.01 tokens per step on GSM8K, HumanEval and MBPP.
  - Chat shows +0.11, but that's not a real gain. The chip run counts only positions where *its own* main model agrees with the bf16-written text, 16,531 instead of 20,554. That keeps the easier positions.
- **The trimmed vocabulary costs little on maths, more on chat.** At 32K, the right next token is in the list 99.6% of the time for maths and 93.6% for chat. 16K drops chat to 87%.
- **Caution, these are upper bounds for real use.** The text is the model's own greedy output. Sampling with temperature and text written by people would both accept less. We haven't measured either yet.

## 16. KV cache at 128K context (`exp4_long128k.json`)

- **Setup:** one 131,072-token document, bf16 weights and fp32 state, so only the KV format changes. 4,096 positions scored (every 32nd). RTX 5090. The reference pass took 559 s.
- **int8 KV, one scale per 256-value head:** KL 0.0002 ±0.0001, top-1 99.68%, perplexity −0.06%.
  - KL by position: 0–8K 0.0004, 8K–32K 0.0005, 32K–64K 0.00006, 64K–96K 0.00007, 96K–128K 0.00008.
  - **No growth with distance. int8 KV stays free at 128K.**
- **int4 KV, one scale per 32:** KL 0.0160 ±0.0041, top-1 96.19%, perplexity −0.16% (within noise).
  - KL by position: 0–8K 0.020, 8K–32K 0.044, 32K–64K 0.012, 64K–96K 0.0071, 96K–128K 0.0070.
  - **It gets smaller deeper into the context, not larger.** The long-context int4 mode holds at 128K.
- Still not measured: 262K and 1M.

## 17. The one-pass DeltaNet order at 32K (`exp4_onepass32k.json`)

- **Setup:** a 32,768-token sequence, 4,096 positions scored, RTX 4090. The chip's one-pass order (note 18) is compared with the reference three-pass order, both in fp32.
- **Result:** KL 2.9 × 10⁻¹⁰, top-1 identical at **every** position, largest log-probability difference 0.0032, perplexity change −0.00005%.
- **So the one-pass order is exact up to fp32 rounding. The DeltaNet unit can read each head's state once per token.**

## 19. A draft head that only looks at recent text (`exp4_mtp_window.json`)

*Measured 2026-11-18 (UTC), RTX 4090, 6.6 minutes. Rerun: `beam_exp4.py mtp_window`.*

- **The question:** the draft head is a full-attention layer, so at 1M context each draft reads its whole 2 GB KV cache (note 21). Does it still guess well if its attention sees only the last W positions?
- **Setup:**
  - Three 32,768-token sequences, used nowhere else: two Wikipedia prose slices (WikiText-2 test, tokens 0–32K and 64K–96K) and 12 stdlib Python files.
  - bf16 weights (§15 showed the chip format changes acceptance by under 0.01). Top-32K draft vocabulary.
  - Positions from 8,192 on whose next 3 tokens are what the main model emits; every 4th one. That's 6,169 positions in all.
  - Every window is scored on the **same** positions, so the comparison is paired.
- **Checks, both 1.0000:** the new chunked drafting code gives the same drafts as the original, and a window longer than the sequence changes nothing.
- **Results, all three sequences (p_cond for drafts 1–3 → tokens per step at k = 3, change vs full ± paired stderr, share of positions with identical drafts):**
  - full: 0.866, 0.808, 0.700 → **3.054**
  - 16K: 0.860, 0.806, 0.697 → 3.036, −0.018 ±0.005, 95.7% identical
  - **4K: 0.855, 0.796, 0.682 → 3.000, −0.054 ±0.007 (−1.8%), 89.6% identical**
  - 1K: 0.844, 0.776, 0.660 → 2.930, −0.124 ±0.009, 84.3% identical
  - 256: 0.813, 0.742, 0.638 → 2.801, −0.253 ±0.011, 77.7% identical
- **By sequence, tokens per step at k = 3, full → 4K window:**
  - prose A: 3.102 → 3.013 (−0.089 ±0.017). 1,411 positions; the main model matches this text 57.6% of the time.
  - prose B: 2.984 → 2.935 (−0.050 ±0.015). 1,225 positions, 55.7%.
  - code: 3.059 → 3.017 (−0.042 ±0.009). 3,533 positions, 80.3%.
- **What this shows:**
  - The draft head mostly uses recent text. Cutting it to the last 4K positions loses under 2% of tokens per step, and the loss shrinks steadily as the window grows.
  - So the draft head can keep a **4K-position KV ring buffer: 8 MB instead of 2 GB at 1M.**
- **Limit:** measured at 32K context. At 1M the window drops a larger share of the history. The trend here (−0.018 at 16K, −0.054 at 4K) suggests the loss comes from the nearest few thousand positions, not the far past, but 1M itself is not measured.

## 18. Status after experiment set 4

- **The model-side evidence for spec v1 is complete:** the weight, activation, KV and state formats (D-005), MTP acceptance (D-013), KV at 128K, and the one-pass DeltaNet order.
- **Still not measured:**
  - acceptance under sampling with temperature, and on text written by people
  - KV at 262K and 1M
  - acceptance for drafts 4 and beyond

# Experiment set 5: cheaper activation scales

*Measured 2026-11-20 (UTC), RTX 4090, 24 minutes. Code: `golden/beam_exp5.py`; formats in `golden/formats.py` (`int8_g32_bf16s`, `int8_g32_bf16up`, `int8_g32_p2`, `int8_g32_mx`). Full write-up with hardware context: note 25. Rerun: `beam_exp5.py run`.*

## 20. Activation scale precision (`exp5_act_scale.json`)

- **Question:** the activation format is int8 with one scale per 32 values. Does the scale need fp32 precision?
- **Setup:** 8,192 tokens each of Python code and English prose. Each format is compared with the bf16 reference model, and paired with today's format on the same positions.
- **Results** (paired KL change against today on code, the precise set):
  - **bf16-precision scale:** +0.00004 ±0.00005 with bf16 weights, −0.0008 ±0.0003 with chip weights. **No loss.**
  - **power-of-two scale:** +0.00114 ±0.00010 with bf16 weights (activation error 0.00102 → 0.00216), +0.0002 ±0.0006 with chip weights (total 0.0557). **A real cost on its own, invisible on the chip.**
  - OCP MX style: the same as power of two rounded up (+0.00115 / +0.0006).
- **Prose** is too noisy at this length for differences under about 0.005 (stderr about 0.003, dominated by a few positions).

## 21. Activation scale precision on real tasks (`exp5_act_tasks.json`)

- **Setup:** chip weights (GPTQ), greedy, GSM8K (1,319) and HumanEval (164). Every linear layer's input is rounded with the activation format, and each input reaches the matmul at about fp32 precision (bf16 high and low parts), so only the scale precision differs. RTX 4090, 213 minutes.
- **Results** (paired against fp32 activation scales):
  - **fp32 scale:** GSM8K 86.66%, HumanEval 70.73%
  - **power-of-two scale:** GSM8K 86.88% (+0.23 ±0.51, lost 21, gained 24, p = 0.77), HumanEval 69.51% (−1.22 ±1.73, lost 5, gained 3, p = 0.73)
  - **bf16 scale:** GSM8K 87.19% (+0.53 ±0.44, lost 13, gained 20, p = 0.30), HumanEval 70.12% (−0.61 ±1.06, lost 2, gained 1, p = 1.0)
- **Int8 activations at all,** against experiment set 3's unrounded-activation chip run (87.41%): −0.76 ±0.49 on GSM8K (p = 0.16, not significant; includes GPTQ run-to-run differences).
- **Conclusion:** no significant change from cheaper activation scales on either task.

# Experiment set 6: special-function precision

*Measured 2026-11-20, RTX 4090, 48 minutes. Code: `golden/beam_exp6.py`; emulation in `golden/qwen35_golden.py` (`SF`, each result rounded to b mantissa bits). Write-up: note 27 §2.*

## 22. How precise must exp, sigmoid/SiLU, softplus, 1/sqrt and 1/x be? (`exp6_sfu.json`)

- **bf16 weights, all functions at b bits** (KL code / prose, 8,192 tokens each): 16 bits 0 / 0; 12 bits 0.000008 / 0.0025 ±0.0018; 10 bits 0.00011 / 0.0077 ±0.0021; 8 bits 0.00065 / 0.0174; 6 bits 0.0030 / 0.0416; 4 bits 0.0169 / 0.0725.
- **One function at 6 bits:** exp 0.0026 / 0.0420; 1/sqrt 0.00035 / 0.0023; sigmoid/SiLU 0.00008 / 0.0010; 1/x 0.000008 / 0.00005; softplus 0.000001 / 0.000005.
- **Chip numerics, paired against exact functions:** 10 bits +0.0003 / +0.0021 (within noise); 8 bits +0.0009 / +0.0028; 6 bits +0.0037 / +0.0234 (a loss).
- **Conclusion:** exp dominates, and its sensitivity grows with context length. Chosen: exp and 1/sqrt at 16 bits, the rest at 12.

# Experiment set 7: attention on the int8 lanes

*2026-11-20, RTX 4090. Code: `golden/beam_exp7.py`; options `attn_q`, `attn_p` in `qwen35_golden.Numerics`. The first run was cancelled by Beam after 21 minutes (out of credits) after the 8K bf16 rows and the 32K power-of-two row; the remaining rows ran the same day (`beam_exp7.py rest`, 27 min). Results: `golden/results/exp7_attn_int8.json`. Write-up: note 27 §4.*

## 23. int8 queries and int8 softmax probabilities

- **8,192 tokens, bf16 weights, KL against exact attention** (code / prose):
  - q int8 (power-of-two scale per 32): 0.00002 / 0.00011
  - P int8 (power-of-two scale per 32 positions): 0.00005 / 0.00029
  - both: 0.00007 / 0.00039
  - **P on a fixed 1/127 grid: 0.070 / 0.515**
- **Smoke test:** 2,048-token prose, both int8 (power of two) 0.000065, fixed grid 0.20. With the chip's numerics at 512 tokens: 0.0506 vs 0.0521 exact (prose).
- **32,768-token prose, bf16 weights** (4,096 scored positions):
  - both int8, power of two: **KL 0.00067 ± 0.00017**, top-1 99.4%, perplexity +0.43%; by position 0.00066 / 0.00009 / 0.00094 / 0.00068 (0–4K / 4–8K / 8–16K / 16–32K), flat
  - q int8 + P fixed grid: **KL 1.02**, top-1 69%, perplexity +198%; by position 0.31 / 0.70 / 1.17 / 1.20, growing with distance
- **Chip's full numerics, 8,192 tokens** (prose / code):
  - exact attention 0.0523 / 0.0559
  - q and P int8, power of two: 0.0527 / 0.0562; paired +0.0004 ± 0.0020 / +0.0002 ± 0.0005 (not significant)
  - q int8 + P fixed: 0.473 / 0.130
- **Conclusion:** int8 attention with power-of-two scales is free at 32K and on top of the chip's numerics. A fixed grid is not an option.

# Experiment set 8: chip-exact non-matrix numerics (golden v2)

*2026-11-21, RTX 4090, 47 min. Code: `golden/beam_exp8.py`; switches `qwen35_golden.CHIP`. Results: `golden/results/exp8_chip_exact.json`. Write-up: note 29 §4.*

## 24. Real special functions, table RoPE, tiled softmax, DeltaNet q multiply

- **bf16 weights, q and P int8 (D-021), KL against exact** (prose / code; 32K prose):
  - baseline: 0.000386 / 0.000068; 32K 0.000669
  - + special functions: paired −0.00007 ± 0.00011 / +0.000002
  - + RoPE tables: −0.00009 ± 0.00006 / 0.000000
  - + DeltaNet q multiply: 0 / 0
  - + tiled softmax with V scale folded: +0.00029 ± 0.00016 / +0.000052 ± 0.000004
  - all: 0.00177 / 0.000123; 32K 0.00111 ± 0.00041, top-1 99.37%, perplexity +0.64%
- **The chip's full numerics:** 0.0527 → 0.0560 prose (+0.0033 ± 0.0030), 0.0562 → 0.0564 code; 32K 0.0479 → 0.0448. Within noise.

## 25. The chunked DeltaNet for prefill (`exp9_chunked_dn.json`)

*2026-11-21, RTX 4090, 38 min. Note 30.*

- **The question:** can prefill compute the DeltaNet in 64-token chunks, with matrix products on the int8 matrix unit, without losing quality?
- **bf16 weights, KL vs the sequential DeltaNet:**
  - all fp32: below 10⁻⁶
  - int8 one pass: prose 0.0057, code 0.0075, 32K 0.0037
  - int8 3-pass split: at most 0.000004
- **On the chip's full numerics:**
  - **one pass:** code +0.0045 ± 0.0010, a real increase
  - **3-pass split:** prose +0.0019 ± 0.0024, code −0.0003 ± 0.0005, 32K 0.0475 vs 0.0448. Within noise.
- **Answer:** yes, with the 3-pass split (D-025). A 32K prompt takes 5.37 s instead of 6.85 s.

## 26. What the chunked DeltaNet changes inside the model, layer by layer (`exp10_dn_layers.json`)

*2026-11-21, RTX 4090, 29 min. Note 30 part 2.*

- **The method:**
  - 2,048 tokens each of prose and code, on the chip's full numerics.
  - At every layer, the relative change of the DeltaNet/attention output, the FFN output and the residual, plus the share of int8 values entering the next matrices that change.
- **The yardstick (a harmless fp32 reordering):** about 5% relative, and 54–63% of the int8 values change, nearly all by one step. Any change at all decorrelates the rounding, so this is the floor.
- **Results:**
  - **3-pass split and the lite version:** at the floor
  - **one int8 pass:** about 1.6× the floor
- **Quality of lite:** within noise (prose +0.0020 ± 0.0019, code +0.0001 ± 0.0005, 32K 0.0461 vs 0.0448).
- **Lesson:** counting changed int8 values or relative errors alone overstates harm. Always compare against a known-harmless change of the same kind.

## 27. The ISA program on a real model: functional simulator, Qwen3.5-0.8B (`isa/results/fsim_08b.json`)

*2026-11-21, laptop CPU, 93 s. Note 31.*

- **The run:** the real decode program, weights packed in the chip's formats, a 12-token prompt fed one token per step, then greedy.
- **Tokens:** 11 of 12 argmaxes match the golden model with the same weights. The chip completes "The capital of Japan is" with "Tokyo". The mismatch is a near tie: top-2 margin 0.028.
- **Each layer alone:** 69 of 72 sublayer checks at fp32 rounding (1–3 × 10⁻⁷); 3 at 3–11 × 10⁻⁴ (single int8 rounding flips).
- **End to end:** last-layer residual 2.4–3.5% apart, against a golden-vs-golden yardstick of 2.1–3.3% (two valid DeltaNet orders). At the rounding floor.
- **Speed (simulated):** 945 tok/s for the 0.8B at short context, 89% of its memory bound.
- **Found:** 6 program or contract bugs, all fixed except the fused softmax command's missing operands (note 31).


## 28. The prefill program on a real model: functional simulator, Qwen3.5-0.8B (`isa/results/fsim_prefill_08b.json`)

*2026-11-21, laptop CPU, 234 s. Note 31 part 2.*

- **The run:**
  - the real prefill program: lite chunked DeltaNet, fused softmax
  - weights packed in the chip's formats
  - a 256-token prompt in 2 chunks of 128
  - then the decode program, from the chip's own KV cache and state, for 6 steps
- **Each sublayer alone (chip input, chip state):**
  - DeltaNet mixer 0.4–1.2 × 10⁻³
  - attention mixer 0.5–4.0 × 10⁻³
  - FFN 0.8–4.3 × 10⁻⁴
  - These are int8 rounding flips. The probes inside a head agree to 1–6 × 10⁻⁵, the 3-pass product's precision, and the solve is bit-identical to R8.
- **End to end:** last-layer residual 3.2–3.4% apart, against a golden-vs-golden yardstick of 3.2% (attention tile 1,024 vs 256, running-sum vs scan decay). At the rounding floor from layer 5 on.
- **Tokens:** the next token after the prompt matches (" its"), and decode from the chip's state matches 6 of 6.
- **Speed (simulated):** 3.14 ms per 128-token chunk, about 41K tokens/s for the 0.8B at short context.
- **Found:** 14 groups of program and contract bugs, all fixed. One of them, a semaphore off by one at the chunk boundary, had made the 9B prefill look 0.21 ms per chunk faster. Corrected 9B, lite at 16K: 22.40 ms per chunk (was 22.11).

## 29. Vector arithmetic mapping and the decoded SRAM interface

2026-11-21. The eight-stage split-product FMA passed 20,000 streamed inputs. ASAP7 delay-oriented mapping reports 975.9 µm² and 0.919 ns. Area-oriented mapping reports 843.5 µm² and 1.091 ns. The 32-value activation quantizer now maps to 547.9 µm² and 2.871 ns in the area-oriented run. These are pre-layout cell estimates. [Note 32, part 2](32_vector_unit_rtl.md#part-2-eight-stage-arithmetic-and-decoded-vector-commands)

The 512-lane decoded controller passed 31 SRAM differential cases. Dedicated benches passed eight full-size DeltaNet vectors and 64 softmax tiles of 1,024 scores. This run covered the single-head controller. The subsequent four-head implementation is measured in note 32, part 3.

## 30. Instruction capacity and speculative output on Qwen3.5-0.8B

2026-11-21. Fused 9B lite prefill assembles to 5,516 of 8,192 instruction bundles after the KV-head loop. The 0.8B lite program takes 4,132 bundles. Its two-chunk prefill takes 6.282 ms in the timing model and agrees with the golden on the first token and six follow-on decode steps. [Note 31, part 3](31_functional_simulator_first_run.md#part-3-instruction-memory-speculative-output-and-schedule-sensitivity)

Ten speculative steps emitted the same 18 tokens as ordinary simulated decode after correcting the host DMA source stride. The independent golden greedy run agrees on 13 of 18 tokens. For those 18 tokens, simulated speculative and ordinary decode times were 13.433 ms and 19.056 ms, respectively. [Speculative results](../isa/results/fsim_spec_08b.json)
