# 03. The model: Qwen3.5-9B

*Status: proposed 2026-11-10, waiting for a yes. See `DECISIONS.md`, D-002.*

## What we were looking for

- **About 8 to 10B parameters:** big enough to be genuinely useful, small enough to run on a laptop (about 5 to 6 GB at 4-bit), so we can compare against a Mac.
- **Recent:** released in 2026, so the project reads as current.
- **Open weights under a permissive license,** so we can publish everything and fine-tune it later for the RL step.
- **Interesting architecture:** a plain transformer would make a less interesting chip story.

## Candidates we looked at

*Searched 2026-11-09. Sources are at the bottom.*

- **Qwen3.5-9B** (Alibaba, February/March 2026, Apache 2.0). Hybrid design: 3 linear-attention layers for every 1 full-attention layer. Widely ranked the strongest open model under 10B. Runs in LM Studio, MLX and llama.cpp.
- **Granite 4.1 8B** (IBM, April 2026, Apache 2.0). Dense and strong at code, but less widely used and a more conventional design.
- **Qwen3-8B** (2025, Apache 2.0). A plain transformer, now a generation old.
- **Ministral 8B** (Mistral). Older, with a more restrictive license history.
- **ZAYA1-8B.** A mixture-of-experts model, 8.4B total with about 0.76B active per token. Interesting for MoE hardware, but a niche model.
- **Gemma 4** (Google). Its family has no dense model in the 8 to 10B range.
- **TwIL-8B** (webAI, 2026-11-04). Very new, but built on Granite 4.2 8B and licensed for non-commercial use only.
- **MiMo V2.6 9B** (Xiaomi). A fine-tune of Qwen3.5-9B. That's evidence Qwen3.5-9B is still the base other people build on.

## Why Qwen3.5-9B

- **Quality:** it is the best-regarded model in the size range, which makes the chip worth having.
- **Recency and ecosystem:** many 2026 models are fine-tunes of it, so a chip for it also runs those (same shapes, different weights).
- **Its architecture makes a good hardware story.** The DeltaNet layers keep a fixed-size state instead of a growing KV cache. That changes the memory picture in ways worth explaining.
- **License:** Apache 2.0 allows publishing and fine-tuning.

## The model's shape, from its config.json

- **Width:** hidden size 4,096.
- **Depth:** 32 layers in a repeating pattern of three Gated DeltaNet (linear attention) layers, then one gated full-attention layer. That's 24 linear and 8 full.
- **Full-attention layers:** 16 query heads and 4 key/value heads (grouped-query attention), head size 256. The output is multiplied by a learned gate (`attn_output_gate`). Rotary position embedding applies to only a quarter of each head (`partial_rotary_factor` 0.25), with `rope_theta` 10,000,000.
- **Linear-attention layers:** 16 query/key heads and 32 value heads, head size 128, a short causal convolution (kernel 4) before attention, and a per-head decay gate and write strength (beta).
- **FFN:** SwiGLU with inner size 12,288 (three matrices: gate, up and down).
- **Vocabulary:** 248,320 tokens, with separate (untied) input embedding and output head matrices.
- **Extras we will skip at first:** the vision encoder (27 layers, width 1,152) and the multi-token prediction head (one extra layer). We may come back to MTP, see `02_workload.md`.

## What this shape means for the chip

All numbers from `calc/qwen35_budget.py`.

- **Weights read per token: 7.94B.** The embedding table is 1.02B but only one row is read per token, so it doesn't count. The output head is 1.02B and is read in full every token.
- **The FFNs are 61% of all weight reads.** They're the main thing to make fast.
- **At int4 with one fp16 scale per 32 weights,** that's 4.46 GB per token. This sets the decode speed: speed = bandwidth ÷ 4.46 GB.
- **The output head is 13% of all weight reads** and produces 248,320 logits per token. The chip must find the top token (or sample) over a quarter-million values. That's a real design question for the vector unit.
- **The KV cache is small: 16 KB per token at int8,** because only 8 layers keep one, with 4 KV heads each. At 32K context that's 537 MB per user.
- **The DeltaNet state is fixed: 12.58M values** (24 layers × 32 heads × 128 × 128). That's 25 MB in bf16 or 50 MB in fp32, and it doesn't grow with context. It must be read and updated every token. The config's `mamba_ssm_dtype` is float32, so we must check whether bf16 or lower loses accuracy.
- **Compute is small: about 16 GFLOP per token** at short context, rising to 20 GFLOP at 32K context because of attention.

### The new operations DeltaNet brings

A plain transformer needs matrix multiplies, softmax, norms and an activation. The DeltaNet layers add:
- **A short causal convolution** over the last 4 tokens: a small sliding window per channel.
- **The state update** for each head. Roughly: decay the 128×128 state, read it with the key, compute an error against the value, write a correction back as an outer product, then read it out with the query. That's several passes over a 128×128 matrix per head per token. It's small in total (about 0.1 GFLOP per token) but a different data pattern from a weight matrix. The state is read and written, not just read.
- **Gates** using sigmoid, and SiLU in the FFN.

We need to write the exact DeltaNet update equations from the reference code before designing the vector unit. That is an open item in `OPEN_QUESTIONS.md`.

## The smaller model for simulation

A full 9B token costs about 33M cycles. Even at 10,000 simulated cycles per second, that's nearly an hour per token in RTL simulation. So we develop and verify on a model with the same architecture but far smaller:
- **First choice: Qwen3.5-0.8B,** if its config uses the same layer types (not yet checked).
- **Fallback:** our own scaled-down config (for example 4 layers, width 256), using the same code with random or distilled weights. It's good enough to check the chip computes correctly, though its output text won't mean anything.

The RTL will be parameterized, so the 9B version differs only in parameter values and DRAM size.

## Risks with this choice

- **Quantization:** we haven't yet measured Qwen3.5-9B's quality at int4 weights and int8 activations with our exact rounding. If it degrades badly, we may need int8 weights for some layers, which costs bandwidth. This is the first job of the golden model.
- **The huge vocabulary:** 248K logits per token is a lot of vector work and a big final matrix.
- **The architecture is newer than most tools.** Some quantization tools may not support DeltaNet layers well yet.

## Sources

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B) and [config.json](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)
- [Artificial Analysis: Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [LM Studio: Qwen3.5-9B](https://lmstudio.ai/models/qwen/qwen3.5-9b)
- [Labellerr: best small language models under 10B](https://www.labellerr.com/blog/best-small-language-models-under-10b-parameters/)
- [Google: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [MarkTechPost: webAI TwIL-8B](https://www.marktechpost.com/2026/11/04/webai-twil-8b-reasoning-model/)
- [Morph: best open-source LLMs](https://www.morphllm.com/best-open-source-llm)
- [Convly: Ollama models list 2026](https://convly.ai/ollama-models-list-2026/)
