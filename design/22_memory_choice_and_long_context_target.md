# 22. Memory choice and the long-context target

*Written 2026-11-18. Closes open items 1 (memory type) and 3 (prefill floor) of the spec v1 plan (note 19). Numbers from `calc/perf_model.py` with the measured acceptance (note 15 §15 and §19). Prices are cited below. GDDR7 has no published contract price, so its figures come from supply-chain reports and are marked as such.*

## The target changed: long context is the real workload

- **Your direction (2026-11-18):** the chip should be good at long context. Prompts of 20–30K tokens, then working inside that context. Almost nothing useful happens at 4K.
- **What that means in numbers:**
  - **Score context:** the headline tok/s per $ is now measured at **32K** context, not 4K.
  - **Prefill target:** a **20K-token prompt in about 7 s, and a 32K one in about 12 s.**
    - The old target was 2K in 4 s, which is 500 tokens/s. The new one is about 2,800 tokens/s, more than 5× harder.
  - **Capacity:** the base card holds a working context of about 100K tokens. The long-context card holds the full 1M.
- **Why 32K is a fair headline:** with 32K of KV the chip reads 5.11 GB per token, against 4.63 GB at 4K. So the score drops by only 9% (101.5 → 92.1 tok/s), because DeltaNet layers keep no KV cache. GPUs pay the same KV cost per token.

## 1. Memory type: GDDR7 in 2 GB chips (decided)

### The prices we could cite

- **LPDDR5X:**
  - Q2 2026 contract: a 12 GB package went from $77.1 to $145.9, about **$12.16/GB** (SigmaIntel, via TheElec and TweakTown).
  - TrendForce projects Q3 2026 mobile DRAM contract prices up another 8–13% (about $13.1–13.7/GB).
  - Spot, 16 Gb die, 2026-09-09: $30.18, about **$15.09/GB** (memoryindex.io).
- **GDDR7:** no tracker or memory maker publishes a contract price.
  - Supply-chain reports from July 2026 (Taiwan's Economic Daily News, via Notebookcheck, summarised by Capital and Compute) put a **2 GB chip at about $20 ($10/GB)** and a **3 GB chip at $60–70 ($20–23/GB)**.
  - We use $10–12/GB for 2 GB chips, to leave margin.
- **GDDR6 for reference:** spot rose from about $2.50 to about **$7.50/GB** between autumn 2025 and July 2026 (TrendForce, via Capital and Compute). The 8 Gb part was $12.45 on 2026-10-05 (DRAMeXchange).
- **The big caveat:** 2026 memory prices are moving fast. Conventional DRAM contract prices rose 93–98% quarter on quarter in Q1 2026 (TrendForce). We redo this with fresh prices before any real build.

### Candidate cards

*Card cost = memory + die ($12.4, 80 mm² N6) + package ($15) + board. Retail-equivalent = 2 × card cost, as in D-010. Speeds: 32K context, 1,024 × 8 rows, k = 3, 4K draft window.*

- **GDDR7 4 × 2 GB, 8 GB, 512 GB/s:**
  - card cost $152–168, retail-equivalent $305–337
  - 92.1 tok/s plain, 250 with speculation
  - **0.27–0.30 tok/s per $ plain, 0.74–0.82 with speculation**
  - holds up to 105K context with int8 KV, 190K with int4
- **GDDR7 8 × 2 GB clamshell, 16 GB:** card cost $242–274. 0.17–0.19 plain, 0.46–0.52 with speculation. Holds 590K with int8 KV, **1.06M with int4**.
- **GDDR7 4 × 3 GB, 12 GB:** card cost $312–352. 0.13–0.15 plain. 3 GB chips cost 2–2.3× as much per GB.
- **GDDR7 8 × 3 GB, 24 GB:** card cost $562–642. 0.07–0.08 plain.
- **LPDDR5X 256-bit, 16 GB, 273 GB/s:** card cost $257–304. 49.1 tok/s plain, 138 with speculation. **0.08–0.10 plain, 0.23–0.27 with speculation.**
- **LPDDR5X 512-bit, 32 GB, 546 GB/s:** card cost $462–555. 98.2 plain, 264 with speculation. 0.09–0.11 plain. Its smallest useful size is 32 GB (8 packages), and it needs twice the memory interface on the die.

### The pick

- **GDDR7, 32 Gb/s, 2 GB (16 Gb) chips, on a 128-bit bus (4 chips), 512 GB/s.**
- **Base card: 8 GB.** **Long-context card: 16 GB clamshell** (8 chips, two per channel), with int4 KV for anything past about 590K.
- **Why:**
  - Right now GDDR7 in 2 GB chips is *cheaper* per GB than LPDDR5X ($10–12 vs $12–15) and nearly twice as fast per bus width.
  - LPDDR5X loses on both counts: about 3× worse on tok/s per $ at the same capacity.
  - LPDDR5X's one advantage, large cheap capacity, doesn't matter here: 8 GB already holds 100K context and 16 GB holds 1M with int4 KV.
- **Dropped:** the 24 GB card and 3 GB chips, at today's prices. If 3 GB chips fall to the 2 GB price per GB, a 12 GB base card (340K context) becomes attractive.
- **Against the D-010 bar** (0.273, defined at 4K plain): the base card scores 0.30–0.33 at 4K and 0.27–0.30 at 32K. With speculation, 0.74–0.82. Speculation is what makes the win large.

## 2. The prefill target: a much bigger matrix unit

### The problem

- Prefill means reading a prompt of N tokens. Each weight has to be multiplied by every prompt token, so the work is N × 7.94B multiply-adds. A 32K prompt is 2.6 × 10¹⁴.
- At 1,024 × 4 (4,096 multipliers) that's 63 s for the weights alone. The model gives **78 s** in total for 32K.

### What each size gives (prefill time for 8K / 20K / 32K tokens; 32K split by block)

- **1,024 × 4, 256 DeltaNet lanes (today):** 17.7 / 46.5 / 78.1 s (weights 63.5, attention 9.7, DeltaNet 4.8)
- **1,024 × 8:** 9.5 / 25.0 / 42.1 s
- **1,024 × 16, 1,024 DeltaNet lanes:** 4.5 / 12.0 / 20.3 s
- **1,024 × 32, 256 DeltaNet lanes:** 3.4 / 8.9 / 15.0 s. DeltaNet takes 4.8 s of that, running token by token.
- **1,024 × 32, 1,024 DeltaNet lanes:** **2.4 / 6.6 / 11.3 s** (weights 7.9, attention 2.2, DeltaNet 1.2)
- **1,024 × 64, 1,024 DeltaNet lanes:** 1.4 / 3.9 / 6.8 s
- **1,024 × 64, 2,048 DeltaNet lanes:** 1.2 / 3.5 / 6.2 s

### What a bigger unit costs

- **Area (estimate):**
  - Google's first TPU fit 65,536 int8 multiply-adds into about a quarter of a ≤331 mm² die at 28 nm (Jouppi et al., ISCA 2017). That's ≤ about 1,260 µm² per multiply-add, including accumulators and wiring.
  - Moving from 28 nm to a 6 nm-class process shrinks logic about 4–8× (wire-heavy arrays shrink less than the best case). That gives **about 160–320 µm² per multiply-add.**
  - So 32,768 multiply-adds is **about 5–10 mm²**, against about 0.7–1.3 mm² for 4,096. The die grows from about 80 to about 85–90 mm².
  - Die cost rises by about $1. **That's under 1% of the card.**
- **Energy (estimate):**
  - An 8-bit multiply costs about 0.2 pJ and an 8-bit add 0.03 pJ at 45 nm (Horowitz, ISSCC 2014). At 6 nm that's roughly 0.05–0.08 pJ per multiply-add.
  - 32,768 of them at 1 GHz is about 2–3 W of arithmetic. With data movement, roughly **5–8 W while prefilling**.
  - That fits easily in a 75 W slot card (note 23). Decode uses only 8 of the 32 rows, so the other rows can be clock-gated.
- **1,024 fp32 DeltaNet lanes:** about 4× the 256 we had. This is an **unsourced** estimate of a few mm². It needs the open-PDK synthesis check.
- **Not free:**
  - each lane needs 32 token rows of activations per cycle, so a small register file per lane
  - the scale unit must handle 32 rows: 64 group sums × 32 rows = 2,048 per cycle
  - prefill chunks must be at least 64 tokens
  - These are design work, not blockers.

### The pick (proposed)

- **Matrix unit: 1,024 weight lanes × 32 token rows (32,768 int8 multiply-adds). DeltaNet unit: 1,024 fp32 lanes.**
- It meets the target: **20K tokens in 6.6 s, 32K in 11.3 s.**
- It also helps decode at long context, because verifying k + 1 tokens never runs out of rows:
  - 32K: 92.1 → **259.8 tok/s** at k = 3
  - 128K: 69.9 → **202.8**
  - 262K: 52.9 → **156.9**
- **The 64-row option** halves prefill again (32K in 6.8 s) for about another 5–10 mm² and about twice the prefill power. It's kept open until synthesis gives real area numbers.
- **Still to do before logging it as decided:** a rough open-PDK synthesis of one lane with 32 rows, to replace the scaled TPU estimate (open item 2).

## What this changes

- **D-004:** decided, GDDR7 in 2 GB chips, 512 GB/s.
- **D-008:** the score is measured at 32K context. The prefill floor is now 20K in about 7 s and 32K in about 12 s.
- **D-012:** an 8 GB base card and a 16 GB long-context card. The 24 GB card is dropped.
- **D-015:** 1,024 × 32 proposed, replacing 1,024 × 4.
- **Note 18's DeltaNet unit:** 1,024 lanes for prefill. Decode needed only 256.
- **The SRAM budget** needs redoing for 32-row activation tiles (note 19, item 6).

## Sources

- LPDDR5X Q2 2026 contract, $77.1 → $145.9 per 12 GB (SigmaIntel via TheElec): https://www.tweaktown.com/news/112358/dram-prices-surged-by-up-to-89-percent-in-q2-2026-destroying-the-consumer-segment/index.html
- TrendForce mobile DRAM 3Q26 outlook (+8–13%): https://www.trendforce.com/research/download/RP260803NC
- LPDDR5X 16 Gb spot, $30.18 on 2026-09-09: https://memoryindex.io/lpddr-prices
- GDDR7 2 GB ≈ $20 and 3 GB $60–70; GDDR6 spot $2.5 → $7.5/GB; GDDR6 8 Gb $12.45 on 2026-10-05: https://capitalandcompute.net/blog/gpu-prices-2026/
- Conventional DRAM contract +93–98% QoQ in Q1 2026 (TrendForce, cited by): https://www.compute-market.com/blog/dram-shortage-2026-gpu-prices-what-to-buy
- Memory price history (GDDR6, LPDDR5X): https://memory-report.com/compare
- TPU v1: Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit", ISCA 2017, doi 10.1145/3079856.3080246 (65,536 8-bit MACs, 28 nm, ≤331 mm², matrix unit about a quarter of the die)
- Horowitz, "Computing's Energy Problem (and what we can do about it)", ISSCC 2014, doi 10.1109/ISSCC.2014.6757323 (8-bit multiply 0.2 pJ, add 0.03 pJ)
