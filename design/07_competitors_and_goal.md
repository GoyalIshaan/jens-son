# 07. What we're up against, and our minimum goal

*Researched 2026-11-10. Status: the minimum goal is proposed, see `DECISIONS.md`, D-010. Every number here comes from the `competitors()` and `our_card()` sections of `calc/qwen35_budget.py`.*

## The question

Our score is decode tokens per second per dollar for Qwen3.5-9B at batch 1 (D-008). To claim "better than any accelerator", we need to know the best score anyone can buy today. That means looking at the cheapest hardware that runs a 9B model well at batch 1, not at datacenter chips.

## The 2026 price situation

DRAM is in a shortage, driven by AI datacenter demand, and it has pushed up the price of everything that contains memory. This matters for us because memory turns out to be most of our cost too.
- **The RTX 5060 Ti 16GB** went from a $429 launch price to a median of about $805 at Newegg by August 2026 ([Digital Citizen](https://www.digitalcitizen.life/nvidia-rtx-5060-ti-16gb-median-price-reaches-805-as-rtx-50-series-costs-keep-rising/), [TweakTown](https://www.tweaktown.com/news/113088/geforce-rtx-5060-ti-16gb-pricing-hits-800-in-the-us-as-memory-crisis-takes-hold/index.html)). Further RTX 50 price rises were reported for November ([Windows Report](https://windowsreport.com/?p=1514409)).
- **Apple raised the Mac mini's starting price** to $799 with 16 GB in May 2026 ([Bloomberg](https://bloomberg.com/news/articles/2026-05-01/apple-raises-mac-mini-s-starting-price-to-799-after-ai-frenzy-drains-supply)), and the M4 Pro Mac mini to $1,599 in June ([MacRumors](https://macrumors.com/2026/06/25/apple-hikes-m4-pro-mac-mini-starting-price)).
- **NVIDIA raised the Jetson Orin Nano Super** from $249 to $399 ([Hackster](https://www.hackster.io/news/nvidia-hikes-jetson-pricing-by-up-to-101-as-the-company-gets-caught-out-by-its-own-ai-bubble-91701d7c5783)).
- **LPDDR5X contract prices:** a 12 GB package rose from $77.10 to $145.90 in Q2 2026, per TrendForce as reported by TweakTown. That's about $12/GB ([memory-report.com summary](https://memory-report.com/compare)). We found no 2026 GDDR7 contract price.

Prices move weekly right now, and most of the figures below are from August 2026. Before publishing, we should recheck every price and date it.

> **Update 2026-11-11: measured on real GPUs** (`15_measurements.md`, section 4). With llama.cpp Q4_0, Qwen3.5-9B reaches 76–78% bandwidth utilization on an RTX 4090 and 64–66% on an RTX 5090, not the 86% we borrowed from Llama 2. llama.cpp also reads 5.10 GB per token at 4K context, not 4.70, because it keeps the LM head at Q6_K. Re-estimated RTX 5060: 56–67 tok/s and 0.19–0.22 tok/s per $ (vs 82 and 0.273 below). We keep 0.273 as the bar until we measure a real RTX 5060. The estimates below are the original reasoning.

## How we estimated each competitor's speed on Qwen3.5-9B

Nobody has published clean, comparable batch-1 numbers for Qwen3.5-9B across these devices. The guides we found disagree with each other by 2 to 3× and look like estimates rather than measurements. So we did it in two steps:

1. **Measure how well each device uses its own memory bandwidth,** using the most comparable public benchmark: llama.cpp's `llama-bench` text-generation test (tg128) on Llama 2 7B at Q4_0, a 3.56 GiB file. We call the result *utilization*.
   - Example, RTX 5060: 100.61 tok/s × 3.82 GB per token = 384 GB/s actually used, out of 448 GB/s peak. That's 86%.
   - We treat the whole file as read every token. That slightly overstates the bytes (by about 2%), so the utilization figures are slightly generous to the competitors.
2. **Apply that utilization to Qwen3.5-9B.** Run the way llama.cpp runs it, that's about 4.70 GB per token: Q4_0-like weights, fp32 DeltaNet state read and written, and an f16 KV cache at 4K context.
   - RTX 5060: 0.86 × 448 ÷ 4.70 = about 82 tok/s.

This assumes llama.cpp runs Qwen3.5's DeltaNet layers as efficiently as Llama 2's plain layers. That's probably optimistic: one user reports 40 to 50 tok/s for Qwen3.5-9B on an RTX 5060 Ti at long context with a larger quantization ([njannasch.dev](https://njannasch.dev/blog/running-qwen-3-5-35b-a3b-on-5060-ti/)). So treat our competitor numbers as **upper bounds**, which makes our target harder rather than easier.

## The results

Each entry gives price, peak bandwidth, measured utilization, and estimated Qwen3.5-9B speed and tok/s per dollar.

- **RTX 5060 8GB:** about $300 (launch price $299; €323 in Europe in August 2026), 448 GB/s, 86% measured. About **82 tok/s, 0.273 tok/s per $. The best score we found.**
- **RTX 3060 12GB, used:** about $300, 360 GB/s, 82%. About 63 tok/s, 0.209 per $.
- **Intel Arc B580 12GB:** $310 to $387 in August 2026, 456 GB/s, 60% (Vulkan backend, from a secondary source). About 59 tok/s, 0.189 per $. Intel's software leaves a lot of bandwidth unused.
- **RTX 5060 Ti 16GB:** about $805, 448 GB/s, 80%. About 76 tok/s, 0.094 per $. Same memory speed as the 5060, at nearly three times the price.
- **Jetson Orin Nano Super 8GB:** $399, 102 GB/s, 70% assumed (no measurement found). About 15 tok/s, 0.038 per $. A complete computer.
- **Mac mini M4 16GB:** $799, 120 GB/s, 77%. About 20 tok/s, 0.025 per $. A complete computer.
- **Mac mini M4 Pro 24GB:** $1,599, 273 GB/s, 70%. About 40 tok/s, 0.025 per $. A complete computer.

Two caveats:
- **The graphics cards need a host PC.** The Macs and the Jetson don't. Our chip will also be a card in a host, so the fair comparison is card against card. We list the complete computers for context.
- **The RTX 5060 has only 8 GB.** Qwen3.5-9B at 4-bit fits, but long context is tight. It still qualifies under our 32K-context constraint by our arithmetic (about 5 GB needed), but we should check it on real hardware.

## What the numbers say

- **The cheapest modern GPU is the strongest competitor.** The RTX 5060 is basically our design already: 4 GDDR7 chips (8 GB, 128-bit bus) and good software reaching 86% of peak.
- **Memory speed sets everything, and utilization separates the rest.** The B580 has more bandwidth than the RTX 5060 but loses badly at 60% utilization.
- **The complete computers score about 10× worse per dollar,** because you're paying for a whole computer, and their memory buses are narrow.

## Can we beat the RTX 5060? First pass

Our design point:
- **Memory:** 4 GDDR7 chips (2 GB each, 32 Gb/s per pin), 8 GB, 512 GB/s.
- **Die:** about 80 mm² on TSMC N6. That area is an assumption until synthesis.
- **Package:** about $15. **Board, power and connector:** about $35.
- **Retail price** assumed to be 2× unit cost, to compare against their street prices.

From `our_card()`:
- **Memory at $12/GB, 92% utilization:** unit cost about $158, about 103 tok/s, retail-equivalent about $317. **0.324 tok/s per $, about 19% better than the RTX 5060.**
- **Memory at $8/GB:** 0.407 per $, about 49% better.
- **Memory at $15/GB, 86% utilization:** 0.263 per $. **We'd lose.**

Where the advantage comes from:
- **Faster memory:** 32 vs 28 Gb/s per pin, +14%.
- **Higher utilization:** 92 vs 86%, +7%.
- **A much smaller die:** our ~80 mm² vs the RTX 5060's roughly 180 mm² on a pricier node (approximate, to verify). But the die is only about $12 of our $158, so this saves less than it sounds.

**The honest summary:** memory is 60 to 75% of our unit cost, and NVIDIA buys memory at least as cheaply as we could. With the same memory, the most we can win by is our utilization edge plus faster memory parts, roughly 20%. To win by a lot, we need levers that change **bytes per token**, not dollars per byte:
- **Multi-token prediction (MTP):** check 2 to 3 drafted tokens per weight read. If 1.5 to 2 tokens get accepted per step on average, tok/s per $ goes up by that factor. Nothing else on this list is as large. A GPU can do this too, but at batch 1 our idle multipliers make it free.
- **Fewer bits per weight:** for example, mixed precision averaging 3.5 bits instead of 4.5 cuts bytes per token by about 22%. Needs quality checks in the golden model.

## Proposed minimum goal (D-010)

> **Beat the RTX 5060 8GB on Qwen3.5-9B batch-1 decode tokens per second per dollar: at least 0.273 tok/s per $ (retail-equivalent = 2 × unit cost).** For example, at least 82 tok/s at a unit cost of $150 or less.

**Stretch goal:** 1.5× the RTX 5060, about 0.41 tok/s per $, which will most likely need MTP or lower-bit weights.

All of this stays tied to the real constraints: at least 30 tok/s, 32K context, and the prefill floor.

## To check before the blog

- **Measure Qwen3.5-9B on real hardware.** On your Mac at minimum (llama.cpp and MLX), and ideally on an RTX 5060. That replaces our scaled estimates.
- **Recheck every price** in the month we publish.
- **Find a GDDR7 price source,** and an estimate of the RTX 5060's bill of materials, so we can compare on unit cost rather than with a 2× markup assumption.

## Sources

- [llama.cpp CUDA scoreboard (discussion 15013)](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [llama.cpp Apple Silicon table (discussion 4167)](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [Intel Arc B580 for local AI (secondary source for the Vulkan figure)](https://dev.to/jovan_chan_9500711396d4e6/intel-arc-b580-12gb-for-local-ai-in-2026-real-benchmarks-and-the-cuda-free-reality-31b5)
- [TechSpot: Arc B580 price history](https://www.techspot.com/products/graphics-cards/intel-arc-b580.302088/)
- [GPUPrix: Arc B580](https://gpuprix.com/us/gpus/arc-b580)
- [Tom's Hardware: where to buy the RTX 5060](https://www.tomshardware.com/pc-components/gpus/where-to-buy-nvidias-rtx-5060-8gb-gpu)
- [GPUPrix: RTX 5060 (EU)](https://gpuprix.com/eu/gpus/geforce-rtx-5060)
- [Digital Citizen: RTX 5060 Ti 16GB median $805](https://www.digitalcitizen.life/nvidia-rtx-5060-ti-16gb-median-price-reaches-805-as-rtx-50-series-costs-keep-rising/)
- [Windows Report: November RTX 50 increases](https://windowsreport.com/?p=1514409)
- [Bloomberg: Mac mini to $799](https://bloomberg.com/news/articles/2026-05-01/apple-raises-mac-mini-s-starting-price-to-799-after-ai-frenzy-drains-supply)
- [MacRumors: M4 Pro Mac mini price](https://macrumors.com/2026/06/25/apple-hikes-m4-pro-mac-mini-starting-price)
- [Hackster: Jetson price increase](https://www.hackster.io/news/nvidia-hikes-jetson-pricing-by-up-to-101-as-the-company-gets-caught-out-by-its-own-ai-bubble-91701d7c5783)
- [Memory Report: memory prices](https://memory-report.com/compare)
- [njannasch.dev: Qwen3.5 on an RTX 5060 Ti](https://njannasch.dev/blog/running-qwen-3-5-35b-a3b-on-5060-ti/)
