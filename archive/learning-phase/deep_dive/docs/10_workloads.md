# Workloads

An AI chip is not one kind of chip. Training, prefill and decode want different hardware, and a large part of this field comes down to one number: arithmetic intensity.

::viz roofline

## The roofline

For a given operation, arithmetic intensity is operations performed per byte moved from memory. A machine has a ridge point equal to peak FLOP/s ÷ memory bytes/s.

- If intensity is below the ridge, you are **memory-bound**. Speed = intensity × bandwidth.
- If intensity is above the ridge, you are **compute-bound**. Speed = peak.

Adding compute to a memory-bound workload does nothing. Adding bandwidth to a compute-bound workload does nothing. The first job in any analysis is working out which side you are on. For Ironwood, the ridge is 316 FLOP/byte in BF16. [D]

## The three workloads

| | Training | Prefill | Decode |
| --- | --- | --- | --- |
| What it does | Updates weights | Reads the prompt | Writes output |
| Tokens per step per sequence | Thousands | Thousands | 1 to 4 |
| Limited by | Compute, if done right | Compute | Memory, unless heavily optimized |
| Extra needs | Transpose, optimizer vector work, 16-bit math, flexibility | Fast arrays | Bandwidth, capacity for KV |

Inference prices show this directly. Output tokens cost more than input tokens, even though the math per token is almost the same. [R]

Roune's advice to a startup is not to start with a training chip. Training needs transposition hardware, more vector compute, 16-bit numerics, more ops and more flexibility, for researchers who try new things daily. [R] Google also started with inference: TPU v1.

## Decode: where it gets hard

Generating one token per conversation means the matmul has one row. That gives under 1% utilization on a large array, and it reads every weight and all the keys to do very little math.

### Feed-forward in decode

Batching helps because users share weights. Intensity for batch B and b bytes per weight:

`intensity = 2B ÷ b`

So the batch needed to reach the ridge is `ridge × b ÷ 2`.

| Chip | Ridge in F/B | Batch at ridge [D] |
| --- | --- | --- |
| TPU v2, BF16 | 66 | 66 |
| TPU v4, BF16 | 229 | 229 |
| Ironwood, BF16 | 316 | 316 |
| Ironwood, FP8 | 632 | 316 |
| GB300, NVFP4 | 1875 | 469 |

Each generation asks for more batch. Lower precision does not change this, since it raises peak and cuts bytes per weight by the same factor. [D]

### Attention in decode

Batching does not help attention, since each user has different keys. [R] What helps is sharing keys across queries:

- **Grouped-query attention.** 128 heads in 8 groups gives 16 queries per key set.
- **Speculative decoding.** 4 tokens at a time gives 4× more.
- **Parallel agents.** Several chains of thought over the same context. [R]

Combined: 4 × 128 ÷ 8 = 64× reuse. [R] Attention intensity is `2q ÷ b` for q queries per key set. With q = 64 and 1 byte per element, that is 128 FLOP/byte: much better, and still under Ironwood's ridge. [D]

| Technique | Helps FF? | Helps attention? | Cost |
| --- | --- | --- | --- |
| Batch | Yes | No | Latency, KV capacity |
| GQA | No | Yes | Some model quality |
| Speculative decode | Yes | Yes | Wasted work on bad guesses |
| KV quantization | No | Yes | Accuracy risk |
| Sparse attention | No | Yes, greatly | Model work |
| Weight quantization | Yes | No | Accuracy risk |

### Why you cannot speculate forever

Guesses come from a cheap model, and the chance of 16 correct tokens in a row falls quickly. [R] Why not one KV group for everybody? Model quality suffers. [R] Every multiplier has its limit.

## Prefill

All tokens are known, so you process them at once. Each key gets reused thousands of times, and memory bandwidth is rarely the issue. The problem becomes the K dimension in attention: GQA models usually widen key vectors, which helps fill the array. [R]

RAG means models read more than they write, so Roune expects most future tokens to be prefill. That would keep arrays important even as decode gets the attention. [R]

## Mixing prefill and decode

| Strategy | Idea | Tradeoff |
| --- | --- | --- |
| Default aggregation | Same chip, unmanaged mix | Unpredictable balance |
| Disaggregation | Separate chips, moving KV between them | Specialized hardware, transfer cost |
| Managed aggregation (chunked prefill) | Mix in a fixed ratio on one chip | Needs a scheduler, but uses both resources fully |

The third is elegant. A memory-bound decode batch and a compute-bound prefill chunk use different resources, so running them together fills both. You could even provision less bandwidth, which is a risky bet on customer software. [R] Overlapping attention with feed-forward works the same way.

Google's 8t and 8i split is disaggregation at the product level. Nvidia pairing LPUs for decode FF with GPUs for attention and prefill is disaggregation within one model. [F, R]

## What software changes

Almost every provisioning answer depends on software, and answers can differ by 10× or 100×. [R] A chip for perfect software and a chip for arbitrary software are different products. This is the single largest source of disagreement between experts on this topic. Each side has a point: the empirical studies measure what is deployed today, and the hardware architects reason about what could be deployed.

## Other model families

Your factbook covers cases that do not fit this picture: state-space models with fixed-size recurrent state, diffusion language models, and hybrid attention. [F] Depthwise convolutions in Mamba models have K as small as 9 and run badly on big arrays, so they belong on vector units. [R] Architectures that replace attention would change KV math entirely.

Skeptics warned that ASICs would go out of date with AI's pace. Transformers went from a paper in December 2017 to 21% of Google's workload just 15 months later, and 74% by 2026. [J] TPU v2's architecture handled that shift because it was built on general matmul, vector units and programmable control. Designing for a stable abstraction is the defense against this risk.
