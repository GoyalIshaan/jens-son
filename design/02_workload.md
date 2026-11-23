# 02. The workload: who is this chip for?

*Status: decided 2026-11-10. See `DECISIONS.md`, D-007 (which replaces D-003) and D-008 (the metric).*

## Why this comes first

The workload decides almost everything else. The same model on the same chip behaves completely differently depending on how many people use it at once.

- **One user:** each step generates one token for one person. The chip reads all 4.46 GB of weights to make one token, so the memory bus sets the speed, and the multipliers mostly sit idle.
- **Many users at once (batching):** the chip reads the weights once and uses them for every user in the batch. The weights are shared, so adding a user costs little bandwidth, and arithmetic becomes the limit instead.

So "how many users" really decides whether this is a memory chip or a compute chip.

## The four options we considered

### Option A: one person, chatting (batch 1)

Like running the model in LM Studio on a laptop. One conversation, typing speed matters, prompts are mostly short.
- **Metric:** decode tokens per second.
- **Bottleneck:** memory bandwidth.
- **The array can be tiny,** about 500 MACs per cycle at 1 GHz (budget script: batch 1 at 273 GB/s needs 492).

### Option B: one person, running an agent (batch 1 to 4, long prompts)

Like a coding agent on your own machine. The prompt is long (files, tool outputs, search results), the agent may run a few sub-tasks in parallel, and it reads far more than it writes.
- **Metrics:** time to first token for a long prompt (prefill), then decode speed.
- **Bottlenecks:** prefill is compute-bound, and decode is memory-bound.
- **The array must be big enough for prefill,** about 16K MACs per cycle (roughly 128×128) to read a 2,048-token prompt in one second.

### Option C: a small group (batch 8 to 32)

Like a box serving one office or one team. Many users at once, each wanting a decent speed.
- **Metric:** total tokens per second while each user still gets an acceptable speed.
- **Bottlenecks:** a mix. Weights are shared, but each user's KV cache and DeltaNet state still have to be read every step.

### Option D: datacenter throughput (batch 64 and up)

Like a cloud API.
- **Metric:** tokens per second per dollar and per watt.
- **Bottleneck:** compute, plus KV cache capacity.
- **Needs HBM,** a big array and lots of memory. Here we would be competing directly with H100/B200-class GPUs and TPUs, which are built for exactly this.

## The numbers behind the comparison

All from `calc/qwen35_budget.py`, with LPDDR5X on a 256-bit bus (273 GB/s), int4 weights (4.46 GB per step including scales) and bf16 DeltaNet state.

At a context of 4,096 tokens:
- **Batch 1:** 59.6 tok/s for the one user. The array needs about 500 MACs per cycle.
- **Batch 4:** 55.3 tok/s per user, 221 tok/s total. About 1,800 MACs per cycle.
- **Batch 8:** 50.5 tok/s per user, 404 tok/s total. About 3,300 MACs per cycle.
- **Batch 32:** 33.2 tok/s per user, 1,063 tok/s total. About 8,800 MACs per cycle.

At a context of 32,768 tokens, each user's KV cache is 537 MB and must be read every step, so batching helps much less:
- **Batch 1:** 54.0 tok/s.
- **Batch 8:** 29.8 tok/s per user, 238 tok/s total.
- **Batch 64:** 6.5 tok/s per user, 416 tok/s total. The memory bus is now flooded with KV reads, not weights.

## What the numbers tell us

- **Going from batch 1 to batch 8 is almost free at moderate context.** Total output rises 6.8× while each user's speed drops only 15%. The weights are read once either way; the extra cost is each user's KV and state.
- **The prefill array already pays for small batches.** Batch 8 needs about 3,300 MACs per cycle; a 1-second 2K prefill needs about 16,500. If we build the array for prefill (Option B), batch-8 decode fits inside it with room to spare. No extra hardware needed.
- **Long context kills large batches on LPDDR.** At 32K context, batch 64 gets only 6.5 tok/s per user. Serving many long-context users needs HBM, which pushes us into Option D's world.
- **Qwen3.5's design helps here.** Only 8 of 32 layers keep a KV cache. A plain transformer of the same size with KV in all 32 layers would need 4× the KV bytes, and batching would break down even sooner.

## How the decision went

**First recommendation (Claude, 2026-11-10):** Option B, one person's agent, with prefill speed as a main target and up to 8 streams. Recorded as D-003.

**The user's revision (2026-11-10):** one person, local inference, and the chip should be **as cheap as possible while being as fast as possible**. In other words, optimize **tokens per second per dollar** for Qwen3.5-9B, and beat every existing accelerator on that measure. On top of that, the chip should expose its VLIW instructions so anyone can write their own programs for it. Recorded as D-007 and D-009.

The difference matters: D-003 treated prefill speed as a goal in itself, which would have made the array big. D-007 makes **cost** a first-class goal, so every square millimetre of silicon has to earn its place.

## The decision: single-user local inference, optimized for tokens per dollar

### What we optimize

> **Score = decode tokens per second ÷ unit cost**

- **Decode tokens per second:** Qwen3.5-9B, int4 weights, batch 1, 4K context, measured (in simulation) end to end.
- **Unit cost:** what it costs to make one more accelerator at volume: the die, the package, the memory chips and the board. One-time design costs (NRE: masks, IP licences, engineering) are reported separately, because every chip company has them and they say nothing about the design itself.
- **Constraints so the score can't be gamed:**
  - decode at least 30 tok/s (a very cheap but slow chip doesn't count)
  - context at least 32K for the score. The chip must also handle the model's full 1,010,000-token context (D-011, decided 2026-11-10). See `11_memory_capacity_and_1m_context.md`
  - a prefill floor, proposed at a 2K-token prompt in 4 s or less (see below)

### Why tokens per dollar works out to "bandwidth per dollar"

At batch 1, decode speed ≈ memory bandwidth × (fraction of it we actually use) ÷ 4.46 GB. So:

> tok/s per $ ≈ (bandwidth × utilization) ÷ (4.46 GB × (memory cost + silicon cost + package and board cost))

That gives four levers, and every design choice should pull at least one of them:
1. **Bandwidth per memory dollar.** Pick the memory part that gives the most GB/s per dollar while holding at least about 5.5 GB (`04_memory.md`).
2. **Utilization.** Get as close to 100% of peak bandwidth as possible. GPUs running batch-1 inference typically reach well below peak. A statically scheduled chip with a perfect weight layout should do better. This is the lever the RL compiler improves.
3. **Small silicon.** Decode needs only about 500 MACs per cycle at 273 GB/s, or about 1,000 at 512 GB/s. Everything beyond that is there for prefill, or is wasted.
4. **No over-provisioning.** Don't buy 32 GB of memory when the model needs about 5.5 GB.

### What this does to prefill

Prefill is no longer a goal in itself. It's a cost. A 128×128 array (for a 2K prompt in 1 s) is mostly idle during decode, and its area raises the unit cost. So we set a prefill **floor** and size the array to just meet it. Proposed floor: a 2K-token prompt in 4 s or less. That needs about 8.2 TOPS, roughly 4,100 MACs per cycle at 1 GHz, about a 64×64 array. That's 4× smaller than the 1-second target. The exact floor is an open question.

### What "better than any accelerator" has to mean

The fair comparison is against the cheapest hardware that runs Qwen3.5-9B well at batch 1 today. That's probably consumer GPUs and Mac minis, not datacenter accelerators (which are built for large batches and cost tens of thousands of dollars).

The ones to beat, with specs and prices still to verify:
- an RTX 5060 Ti-class GPU: 128-bit GDDR7, about 448 GB/s
- an M4 Pro Mac mini: 273 GB/s

To compare fairly we need the same kind of cost for both sides, ideally an estimated unit cost (bill of materials) for theirs and ours, not our unit cost against their retail price. Where we can't get that, we'll say so and also compare on **bandwidth utilization**, which needs no prices at all.

### What we give up

- **Throughput at large batch.** Explicitly not a goal.
- **Fast prefill.** Long prompts will take noticeably longer than on a big GPU. That's the price of a small die.
- **Peak decode speed beyond the score.** If doubling the memory doubles the speed but more than doubles the cost, we don't do it.

## Programmability: the chip is not a hardwired model

The user's second point: anyone should be able to write their own programs for the chip using its VLIW instructions. This turns out to fit the design naturally:
- **Qwen3.5-9B will itself be a program** on the instruction set, not a fixed state machine. So the instruction set has to be general enough for loops, any matrix shape, DMA with strides, and a menu of vector operations anyway.
- **The same instructions then run other things:** fine-tunes of Qwen3.5-9B, the 0.8B model, other transformer-style models that fit in memory, and general calculations built from matrix and vector operations.
- **The RL goal needs it too:** a model can only learn to write programs if there is a public, documented instruction set and a simulator to test them on.

The rule for generality under a tokens-per-dollar goal: **add a general feature only if it costs almost no area, or if Qwen3.5-9B needs it anyway.** For example, supporting any matrix shape is nearly free. A floating-point matrix unit just in case is not.

## The performance targets this implies

All proposed, to be confirmed with the memory choice (D-004) and the cost model:
- **Decode:** at least 30 tok/s, and as high as the chosen memory allows at about 90% bandwidth utilization or better.
- **Prefill floor:** a 2K-token prompt in 4 s or less.
- **Context:** 32K tokens.
- **Batch:** 1. Small batches are allowed if they're free, but are not a goal.

## Questions this leaves open

- **The prefill floor.** It sets the array size, the biggest area knob. What prompt lengths does local use actually see?
- **Memory and silicon prices.** We need cited estimates for LPDDR5X and GDDR7 cost per chip and for wafer cost per mm², or the score can't be computed.
- **The MTP head.** At batch 1 the multipliers sit idle, so checking 2 to 3 drafted tokens per weight read is nearly free and could raise tok/s with no extra memory cost. For a tokens-per-dollar goal this is one of the cheapest speedups available, and it moves up the priority list.
