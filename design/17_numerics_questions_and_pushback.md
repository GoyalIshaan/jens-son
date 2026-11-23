# 17. Numerics: the questions we asked, and the pushback

*Written 2026-11-11. Status: design history. This records questions that came up while reading the measurements in `15_measurements.md`, and the answers that kept the design where it is. The numbers come from `15_measurements.md` and `calc/qwen35_budget.py`.*

Each section starts with the question as it was asked.

## "What is the point of all this numerics work?"

- The model is about 9 billion numbers (weights). Producing one token means reading about 8 billion of them from memory.
- At batch 1, **speed is set by bytes read per token, not by arithmetic.** On our planned card (512 GB/s):
  - bf16 (2 bytes per weight): about 16 GB per token, so about 29 tok/s
  - 4 bits plus scales (about 0.56 bytes per weight): about 4.5 GB per token, so about 100 tok/s
- Fewer bits means rounding, and rounding changes the model's answers. Numerics is choosing how to round each kind of number (weights, activations, KV cache, DeltaNet state) so that we stay fast without the model getting noticeably worse.
- We first built our own copy of the model (the golden model) and checked it matches Hugging Face exactly. So any change we measure is caused by the rounding, not by bugs.

## "What is plain rounding, and what is smarter rounding?"

**Plain rounding (round-to-nearest int4 with a scale per 32 weights):**
- Find the largest weight in the group, say 0.70. Scale = 0.70 ÷ 7 = 0.10.
- The allowed values are the 16 evenly spaced steps −0.8 … 0.7.
- Snap each weight to the nearest step: 0.13 → 0.10, −0.04 → 0 (erased), 0.26 → 0.30.
- **Two weaknesses:**
  - one big weight sets a coarse step size for the whole group
  - evenly spaced levels waste slots at the edges, where few weights live

**Smarter rounding (all at the same 4.5 bits):**
- **Scale search:** try 16 scales from 70% to 100% of max ÷ 7, and keep whichever gives the smallest total error for the group.
- **Smaller groups:** a scale per 16 weights instead of 32, with the scale stored as an 8-bit float (FP8) to keep the size the same.
- **FP4 levels (NVFP4):** the levels 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6. They're closer together near zero, where most weights are.
- **A learned table:** k-means clustering places 16 levels where each matrix's weights actually bunch up. The chip's 16-entry lookup table holds this table.

**Results on Python code (KL, lower is better):**
- plain int4: 0.153, code perplexity +20%
- plain int4 + scale search: 0.145
- int4 levels, groups of 16, FP8 scales: 0.117
- NVFP4: 0.110
- NVFP4 + scale search: 0.105
- learned table: 0.105
- learned table + scale search: **0.101**, code perplexity about +13%

## "Are we finding something that works for us, or just testing popular methods?"

The honest answer: **so far, mostly testing known methods.**
- MXFP4 and NVFP4 are industry formats.
- Scale (clipping) search is a standard trick.
- A learned table per matrix is close to published codebook methods (SqueezeLLM, NF4).

What is ours:
- **The yardstick:** everything at the same 4.5 bits, on our exact model, scored on code as well as prose, and limited to formats our chip can decode cheaply.
- **The hardware conclusion:** build a 16-entry lookup table and support scale groups of 16 and 32. Every method above, and better ones later, then runs on the same silicon. **The hardware can be settled now while the rounding recipe keeps improving in software.**

Something that may be new: combining calibration-based rounding (GPTQ) with a learned table. That hasn't been tested yet.

## "What do you mean by the 13% ceiling?"

- **Perplexity** measures how surprised the model is by the real next token, so lower is better. On our code text, the exact model scores 1.47 and the best 4-bit versions score about 1.67. That's **13% worse**.
- Four quite different methods all landed at 13 to 14%, with each new trick helping less than the one before. "Plateau" is the better word than "ceiling".
- **Why we think it's a limit of the methods, not of 4 bits:** all of these methods keep each *weight* close to its original value. What matters is each layer's *output*, meaning weights times inputs.
  - **GPTQ:** round weights one at a time, nudging the rest to cancel the error on real inputs.
  - **AWQ:** protect the weights that meet large inputs.
  - **QAT:** briefly retrain the model with the rounding in place.

  These are known to do better, and they all produce codes and scales our chip can use.
- **How sure we are:** it's a pattern across 6 methods on 2,048 tokens. GPTQ is the test that will tell us.

## "Why not build a chip that runs bf16 weights and activations? If we design the hardware and software around this one model, won't performance be easy?"

**Pushback: specializing doesn't remove the bottleneck.** The bottleneck is moving bytes from memory, not doing maths.
- bf16 means about 16 GB read per token. At 512 GB/s that's about 30 tok/s however clever the chip is. The multipliers would sit idle, waiting for weights.
- bf16 at 100 tok/s would need about 1,600 GB/s of bandwidth (3× our card) and at least 18 GB of memory. That's RTX 5090 or datacenter-HBM class hardware.
- **Keeping the weights on the chip** (as Groq and Cerebras do) would need 18 GB of SRAM: hundreds of chips or a whole-wafer chip. Fine for a datacenter, impossible for a cheap personal card.
- **What specializing does buy:** using more of the memory bandwidth (92% vs a GPU's 80 to 86%), a small cheap die, and speculative decoding almost for free. All of these make each byte go further. None of them make 16 GB arrive faster.

## "Why are we obsessed with 100 tok/s? If competitors can't do it, isn't it fine? We just need the best tokens per dollar."

**Agreed on the goal:** the target is tokens per dollar (D-008). 100 tok/s is a consequence of 4-bit on our card, not a target. A choice that gives 80 tok/s on a much cheaper card would be better.

**Pushback: bf16 loses on tokens per dollar too, and badly.**
- Tokens per dollar ≈ (bandwidth per dollar) ÷ (bytes per token).
- bf16 makes bytes per token 3.5× larger. It also needs 24 GB of memory instead of 8, so the card costs more too. **Every bit per weight costs twice.**
- Rough numbers, at about $12/GB of memory and retail at 2× unit cost:
  - **our card at 4-bit:** 8 GB, about $158 to make, about 100 tok/s, about **0.32 tok/s per $**
  - **our card at bf16:** 24 GB, about $350 to make, about 29 tok/s, about **0.04 tok/s per $** (8× worse)
  - **RTX 5060 running 4-bit:** about **0.27 tok/s per $**
- **"Compare bf16 to bf16 instead?"** It's true that cheap GPUs can't even hold this model at bf16. But almost nobody runs a 9B model at bf16 at home. They run 4-bit, because it's about 3.5× faster and the quality loss is small. A bf16 tokens-per-dollar champion would win a contest nobody is entering.

**What we take from the question:**
- The 30 tok/s floor means "comfortably fast to read". It is not a number to maximize.
- 4-bit isn't sacred. If 5 or 6 bits gave much better quality and still beat the GPUs on tokens per dollar, it could be worth it. The bit width is chosen on quality and tokens per dollar together.

## The one-line takeaway for the blog

> For one user at a time, an inference chip is a memory-bandwidth machine with a little maths attached. Each bit removed from each weight is speed and money saved. How you round those bits matters almost as much as how many you keep.
