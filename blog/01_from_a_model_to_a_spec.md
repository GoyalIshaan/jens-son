# Designing a chip for one model, part 1: from a goal to a spec

*A worklog. Draft, not published.*

**I'm designing a chip whose only job is to get the cheapest tokens per second out of one open model, Qwen3.5-9B.**

## The goal: the cheapest tokens per second

I absolutely love chip design and always wanted to get my hands dirty with it, which is why I started this project. It started with one question: **what's the cheapest piece of hardware that can generate tokens from a good open model as fast as possible?** Not the fastest chip, and not the cheapest, but the most tokens per second for every dollar spent on the hardware. The target is to beat every accelerator you can buy today on that number, for one specific model.

That's written down as a single score:

> **score = decode tokens per second ÷ the unit cost of the card**

<details>
<summary>Note: what exactly the score counts</summary>

- **Decode tokens per second:** how fast the chip generates the model's output for one person, measured end to end.
- **Unit cost:** what it costs to make one more card at volume: the die, the package, the memory chips and the board. One-time design costs (masks, licences, engineering) are reported separately, because every chip company pays them and they say nothing about the design itself.
- **Guard rails, so the score can't be gamed:** at least 30 tokens per second (a very cheap but slow chip doesn't count), and the chip must handle the model's full context length.

</details>

Everything in chip design is downstream of the workload you're serving. I wanted to make something unique that has a real hope of being competitive, so this chip is hyper-specific to its workload.

### Who it's for: one person, batch 1

Because the goal is the cheapest tokens per second, a chip like this makes the most sense **running locally, on its own (not in a cluster), serving one user at a time: batch size 1.** It can technically serve a few streams at once, but everything is tuned for batch 1.

At batch 1, the speed is set almost entirely by the memory, not the arithmetic:

> decode tokens/s ≈ memory bandwidth × (fraction of it actually used) ÷ bytes read per token

<details>
<summary>Background: why batch 1 is limited by memory (skip if you know your basics)</summary>

To generate one token, the chip has to multiply the current activation vector by every weight matrix in the model. A model of about 9 billion parameters reads nearly all of them for every token, and each weight gets used for exactly **one** multiply-add for that token, then it's done. So the chip moves gigabytes of weights from memory to do a few billion multiply-adds, and the multipliers spend most of their time waiting.

With many users at once (batching), each weight that arrives is reused for every user in the batch, and arithmetic becomes the limit instead. That's the datacenter regime, where GPUs and TPUs with expensive HBM already shine, and there's no point competing there.

</details>

Putting the score and that formula together shows exactly where the levers are:

> tokens/s per $ ≈ (bandwidth × utilization) ÷ (bytes per token × cost)

- **Bandwidth per dollar:** pick the memory that gives the most GB/s per dollar at the capacity we actually need.
- **Utilization:** get as close to 100% of peak bandwidth as possible.
- **Bytes per token:** store each weight in fewer bits, without hurting quality.
- **Small silicon:** every square millimetre that isn't needed is cost.

Every decision in the rest of this post pulls on at least one of these four.

### Why one model

Knowing the exact model ahead of time lets the chip skip what general accelerators pay for:

- **The shapes are fixed,** so buffers are sized exactly and the schedule is worked out offline.
- **The weights are fixed,** so they're stored in memory in the order they're read, and memory streams near its peak (utilization).
- **Nothing the model doesn't use gets hardware** (small silicon).

The cost: if the model changes, the chip gets worse.

It's still **programmable**: the model runs as a program on a public VLIW instruction set, which others can use too. A general feature only goes in if it's nearly free in area, or the model needs it anyway.

### What this part covers

"We" in this post means me and my dawg Claude.

This first part covers everything up to now: choosing the model, measuring the competition, building a reference model, settling the number formats, choosing the memory, adding speculative decoding, pivoting to long context, and then writing and synthesizing the first real pieces of RTL. The theme that ties it together turned out to be simple: **at batch 1, the memory sets the speed, and everything else is about not wasting it.**

## Picking the model: Qwen3.5-9B

With the goal fixed, the next question was which model to build for. The goal shaped the choice. I wanted something around 8 to 10 billion parameters: big enough to be genuinely useful, small enough that a 4-bit copy fits in about 5–6 GB, so it can be compared against consumer GPUs and Macs. It had to be recent, open-weight under a permissive licence (so everything can be published and fine-tuned later), and ideally architecturally interesting.

We looked at Granite 4.1 8B, Qwen3-8B, Ministral 8B, a mixture-of-experts 8B and a few others. **Qwen3.5-9B** won on all counts: it's widely regarded as the strongest open model in its size class, it's Apache 2.0, many newer models are fine-tunes of it (which a chip for it would also run, since the shapes are the same), and its architecture is not a plain transformer.

<details>
<summary>Background: the model's shape</summary>

- **Width 4,096, 32 layers.** The layers repeat in a pattern: three **Gated DeltaNet** layers (a form of linear attention), then one full-attention layer. That's 24 DeltaNet layers and 8 attention layers.
- **The attention layers** use grouped-query attention: 16 query heads share 4 key/value heads, each head 256 wide.
- **The FFN** is SwiGLU with an inner width of 12,288: three matrices (gate, up, down) per layer.
- **The vocabulary is huge:** 248,320 tokens, with a separate input embedding and output head.
- **It ships with an extra layer** for multi-token prediction (MTP), which turned out to matter a lot later.

</details>

Put through the goal's formula, the shape gives the first hard number: the chip reads about **7.94 billion weights per token.** The 1-billion-entry embedding table doesn't count, since only one row of it is read per token. The 1-billion-entry output head does count, because all of it is read every token. The FFNs are 61% of those reads and the output head 13%. At 16 bits per weight that's about 16 GB per token, and at about 4 bits about 4.5 GB. That's the "bytes per token" in the score, and much of this post is about shrinking it without hurting quality.

<details>
<summary>Background: what DeltaNet is, with small numbers</summary>

A normal attention layer keeps a **KV cache**: a key and a value vector for every past token. It grows with the context, and every new token reads all of it.

A DeltaNet layer instead keeps a fixed-size memory per head, a matrix S (128 × 128 in this model). For every token, each head does four things:

1. **Fade** the memory by a decay g between 0 and 1.
2. **Read with the key:** what the memory currently "thinks" the value for this key is, `kv_mem = Sᵀk`.
3. **Write a correction:** `delta = β·(v − kv_mem)`, then `S ← S + k·deltaᵀ`. It writes only the *difference* between what it wanted to store and what it already had. That's the "delta rule".
4. **Read the output with the query:** `o = Sᵀq`.

A 2 × 2 example. Start with S = [[0.2, 0.4], [0, 0]], key k = [1, 0], value v = [0.5, 0.2], write strength β = 0.5 and decay g = 0.9.

- Fade: S becomes [[0.18, 0.36], [0, 0]].
- Read with the key: kv_mem = [0.18, 0.36].
- Correction: delta = 0.5 × ([0.5, 0.2] − [0.18, 0.36]) = [0.16, −0.08]. Row 1 becomes [0.34, 0.28], halfway from where it was toward v.
- Output with q = [1, 0]: [0.34, 0.28].

</details>

What DeltaNet means for the hardware: 24 of the 32 layers keep a fixed-size memory (24 layers × 32 heads × 128 × 128, about 12.6 million numbers) that **does not grow with context.** It's the same size at token 10 or token 1,000,000. Only the 8 attention layers keep a growing KV cache, and with 4 KV heads of 256 each, that's about 16 KB per token at 8 bits. A plain transformer of the same size, with KV in all 32 layers, would need four times as much.

The catch: the DeltaNet memory is read **and written** every token, unlike weights, which are only read. That shows up again and again in this design.

## What we're up against

To claim a better score than existing hardware, we need to know what existing hardware scores. The fair comparison isn't datacenter chips. It's the cheapest hardware that runs a 9B model well for one user: consumer graphics cards and small desktop computers.

The first estimates borrowed bandwidth utilization from public llama.cpp benchmarks of an older, plain-transformer model, which suggested consumer GPUs use about 86% of their peak bandwidth at batch 1. The cheapest strong card, an RTX 5060 8 GB, has 448 GB/s, which suggested roughly 82 tok/s for Qwen3.5-9B.

Rather than trust that, we measured. We rented an RTX 4090 and an RTX 5090, converted Qwen3.5-9B to llama.cpp's 4-bit format, and ran its standard benchmark. Dividing the measured tokens per second × the bytes actually read per token by the spec-sheet bandwidth:

- **RTX 4090:** about 151 tok/s at 4K context, **about 77% utilization.**
- **RTX 5090:** about 224 tok/s at 4K context, **only about 64%.**

Qwen3.5-9B uses GPU bandwidth noticeably worse than the older model did. A likely reason is that fixed per-token overheads (kernel launches, the 24 small DeltaNet steps, picking the next token) weigh more when a token takes only about 4 ms. The faster GPU suffers more, which fits that explanation.

Two things came out of this. First, the RTX 5060's real score is probably lower than the first estimate. We still kept the original, harder bar as our minimum target until a real RTX 5060 is measured, because a conservative bar can't be accused of being too easy. Second, **utilization is a real lever.** A chip that streams weights in a fixed, pre-planned order should get much closer to peak than a GPU running a general software stack.

## Step one: a golden model we can trust

Before any hardware decision, we needed a way to answer questions like "if we store weights in this format, how much worse does the model get?" That requires a **golden model**: our own from-scratch implementation of Qwen3.5-9B, with every operation written out in the order the chip will do it, and with pluggable number formats so each part (weights, activations, KV cache, DeltaNet memory) can be rounded exactly the way the chip would round it.

A golden model is only useful if it's right, so the first job was proving it matches the reference implementation. Both ran in full fp32 on the same weights:

- The largest difference in any output logit was about **2 × 10⁻⁵**, where typical logits are around 2 and the largest about 23.
- The top-1 and top-5 predictions matched at every position, and 32 tokens of greedy generation were **identical** on both test prompts.
- The hidden state after each layer drifted smoothly from about 1.6 × 10⁻⁷ to 8.7 × 10⁻⁷ relative error. That's fp32 rounding noise growing with depth. No single layer jumped, which means no single operation was implemented wrong.
- Running a prompt all at once (prefill) and one token at a time (decode) agreed to the same precision. Those are the two modes the chip will run in, so both paths are covered.

From here on, any difference between the golden model and the original can be blamed on the number format being tested, not on bugs.

## Number formats: the biggest quality decision

Since speed ≈ bandwidth ÷ bytes per token, the number of bits per weight is the most direct speed knob there is. The original model stores weights in 16-bit bf16, about 18 GB. Squeeze them to about 4 bits and the chip reads about 4.5 GB per token instead, roughly 3.5× faster on the same memory. The question is how much quality that costs.

<details>
<summary>Background: how we measured quality</summary>

Mostly with **KL divergence.** At every position in a test text, both the original and the quantized model produce a probability for every one of the 248,320 possible next tokens. KL divergence measures how different those two distributions are: 0 means identical. It's more sensitive than perplexity, which can hide changes that cancel out. We used two kinds of text, English prose and Python source code, and reported the top-1 agreement (how often both models pick the same most likely token) alongside.

Then, for the decisions that mattered, we checked **real tasks**: grade-school maths word problems (GSM8K, all 1,319) and two coding benchmarks where the generated code is actually executed against tests (HumanEval and MBPP, 664 problems together).

</details>

### The first surprise: code is far more sensitive than prose

With plain 4-bit rounding, prose perplexity barely moved, while code perplexity rose by about **20%**. The model is extremely confident on code (its perplexity there is about 1.47, so it's nearly certain of most next tokens), and small weight errors chip away at that confidence. From then on, every format was judged on code too.

### What the final weight format looks like

After three rounds of experiments, the format is:

- Each weight is a **4-bit code** that picks one of 16 values from a small **table.**
- Each matrix has **its own table** of 16 levels, learned from that matrix's weights.
- Each group of **16 weights shares one 8-bit floating-point scale** (E4M3).
- Total: **4.5 bits per weight.**

A worked example. Take a group of 16 weights, starting `0.12, −0.30, 0.05, 0.01, −0.02, 0.21, …`

1. **The group scale.** The largest absolute value is 0.30. Store it once as an 8-bit float, and divide every weight by it: `0.40, −1.00, 0.17, 0.03, −0.07, 0.70, …`. Everything now sits between −1 and +1.
2. **The table lookup.** This matrix's 16 allowed values might be `−1.00, −0.72, −0.53, −0.38, −0.26, −0.15, −0.06, 0.00, 0.06, 0.15, 0.26, 0.38, 0.53, 0.72, 0.87, 1.00`. Replace each weight with the index (0–15) of its nearest entry. 0.40 becomes index 11 (0.38). −1.00 becomes index 0. Those indices are the 4-bit codes.
3. **On the chip,** a weight is rebuilt as table[code] × scale. For the first one: 0.38 × 0.30 = 0.114, close to the original 0.12.

The bit count: 16 codes × 4 bits = 64 bits, plus one 8-bit scale = 72 bits, over 16 weights = 4.5 bits each.

### The three ideas that make it work

**A learned table instead of evenly spaced levels.** Plain 4-bit quantization uses 16 evenly spaced levels. Real weights are bell-shaped: most sit near zero, few at the edges. Running k-means clustering on each matrix places its 16 levels where its weights actually are, with more levels near zero. Same 4 bits, less error.

**Calibrated rounding (GPTQ) instead of round-to-nearest.** GPTQ rounds one column of a matrix at a time, measures the error that causes on real input activations, and nudges the not-yet-rounded weights to cancel it. Errors cancel instead of piling up. Interestingly, this *increases* the plain weight error (for the FFN matrices, from 8.3% to 10.6% RMS), while the output error falls by 20–60%. GPTQ deliberately puts its error where the inputs don't look. It runs once, offline, and the chip pays nothing for it.

**Small groups with 8-bit-float scales.** Groups of 16 with an E4M3 scale beat groups of 32 with a 16-bit scale at the same 4.5 bits.

Stacked together, these reduced the code KL from **0.153** (plain int4) to **0.050**, a 3× reduction in output error for the same bytes and the same speed. We also checked the noise: rerunning the same setup with a re-sampled table moved the score by about 0.004, so we treat differences smaller than that as noise.

### Where the error comes from

Rounding one part of the model at a time showed:

- **The FFN** is 61% of the bytes and the biggest single source of error.
- **The DeltaNet projections** are about **twice as sensitive per byte** as the FFN.
- **The output head** is remarkably robust: 4-bit there barely registers, and even 3-bit is tolerable.

That killed the hope of going to 3 bits to cut bytes further. Even with GPTQ, the cheapest 3-bit option cost about +25% code perplexity, and 3-bit DeltaNet projections were catastrophic (+66%). The only exception is an optional 3-bit output head, which buys about 3% more speed for a tiny quality cost.

### Can the table levels be integers?

The software kept the 16 table values in fp32, but hardware wants something small. If they can be stored as 8-bit integers, the multiplier on the chip is a plain int8 × int8 integer multiplier. We stored each level as the nearest k/127 and measured:

- fp32 levels: code KL 0.0500
- 8-bit integer levels: 0.0500
- 6-bit: 0.0504
- 5-bit: 0.0638, a real jump

The explanation is neat. Across all 249 learned tables, the smallest gap between two neighbouring levels is 0.077. An 8-bit grid places each level within 0.004 of its ideal spot, tiny next to that gap. A 5-bit grid has steps of about 0.067, roughly the size of the gap itself, which pushes every table back toward a uniform grid. A uniform grid *is* plain int4, which is why 5 bits loses most of the benefit. **int8 table entries are free, with margin.**

### Do the answers still come out right?

Low KL is encouraging, but what matters is whether the model still solves problems. Three versions answered the same problems with greedy decoding: the original bf16 model, plain 4-bit (the same scheme as llama.cpp's common Q4_0 format), and our chip format.

- **GSM8K maths:** original 89.3%, ours **87.4%**, plain 4-bit **81.8%**
- **HumanEval code:** original 69.5%, ours 70.7%, plain 4-bit 65.9%
- **MBPP code:** original 55.8%, ours 55.4%, plain 4-bit 54.2%

To tell a real difference from luck, we used a **paired comparison**. Since both models answer exactly the same problems, only the problems where they disagree carry information. On GSM8K, the original solved 63 problems ours missed, and ours solved 38 the original missed: a net −25 of 1,319, which is −1.9 points. If the difference were pure luck, those two counts would be about equal. A 63-vs-38 split happens by chance about 1.7% of the time, so **the 1.9-point maths loss is real, but small.** Plain 4-bit lost 133 and gained 34 (−7.5 points), a split that essentially never happens by chance. On code, our format showed no detectable change.

So the chip format recovers about three quarters of what plain 4-bit loses on maths, at the same 4.5 bits, and nothing measurable on code. One detail I find telling: plain 4-bit and our format have almost the **same** per-weight error, about 10%, yet differ by 5.6 points on maths. Maths is a long chain of steps, one wrong token early ruins the answer, and a per-weight error number can't see where the error lands.

Why not use more bits to close the last 1.9 points? Because 8-bit weights would read about 9 GB per token instead of 5, roughly halving the speed on a chip whose whole score is speed per dollar. If the maths gap ever matters, there are fixes that keep the same format (fine-tuning the model into it).

### Activations, KV cache and the DeltaNet memory

- **Activations** (the vectors flowing between layers) are int8 with a scale per 32 values. This costs nothing measurable. One scale for a whole vector, though, was bad (+14% code perplexity): a few large outlier values force a coarse scale on everything else.
- **The KV cache** in int8, with just one scale per 256-value head, is free. We checked it out to **128K tokens** of context, and the error doesn't grow with distance. A 4-bit KV mode costs a little and stays flat too, so it's kept as an opt-in for very long contexts, where it nearly halves the KV memory.
- **The DeltaNet memory** was the one place where fewer bits clearly hurt, and the way it hurts is instructive. In bf16, the KL by position was small at first and then grew: about 9× worse in the 4–8K range than at the start, with perplexity +1.4% at 8K tokens and +3.4% at 32K. The memory is updated every token, so each rounding error is carried forward and compounded. At a million tokens it would be far worse. **The DeltaNet memory stays in fp32.** That's 50 MB, read and written each token (about 100 MB of traffic), which costs about 1% of decode speed. Worth it.

## Choosing the memory

With the format settled, a token reads about **4.6 GB** at moderate context: about 4.5 GB of weights plus the DeltaNet memory and KV cache. The memory system sets the ceiling, and no amount of clever logic gets past it.

The options:

- **3D-stacked SRAM** fails on capacity: you'd need about 70 stacked dies to hold 4.5 GB. That's the multi-chip, all-in-SRAM approach some datacenter companies take, the opposite of one cheap chip.
- **HBM** needs an expensive silicon interposer and is built for datacenter batching.
- **LPDDR5X** (the memory in phones and Macs) is cheap per gigabyte and low power.
- **GDDR7** (the memory on graphics cards) is much faster per pin.

The useful way to compare them is to **count memory chips, not bus widths**, because you buy chips. One LPDDR5X package gives about 68 GB/s and comes in 8 or 16 GB. One GDDR7 chip gives about 128 GB/s in 2 or 3 GB. To reach about 270 GB/s with LPDDR5X you end up with 32 GB or more, when the model needs about 6 GB at 32K context. You'd be paying for capacity you never use. Four 2 GB GDDR7 chips give 512 GB/s and 8 GB, which is about the right size.

Then prices. 2026 is a bad year to price memory: a datacenter-driven shortage has pushed DRAM prices up sharply and they move month to month. The best figures we could cite put 2 GB GDDR7 chips at about $10/GB and LPDDR5X at about $12–15/GB. At those prices, GDDR7 is both cheaper per gigabyte *and* almost twice as fast per bus width, and the LPDDR5X options come out about 3× worse on tokens per second per dollar.

**The pick: GDDR7 at 32 Gb/s per pin, four 2 GB chips on a 128-bit bus, 512 GB/s.** That makes an 8 GB base card that holds about 100K tokens of context. A long-context version puts eight chips in "clamshell" mode (two per channel) for 16 GB, which with 4-bit KV reaches the model's full million-token context. We'll redo the prices before anything real is built.

This is the same memory setup as the RTX 5060. That's deliberate, and it's honest about the challenge: NVIDIA buys memory at least as cheaply as we could, and memory is most of a card's cost. Winning by a lot can't come from the memory alone. It has to come from using it better and from reading fewer bytes per *useful* token.

### Why the weight order matters

<details>
<summary>Background: how a DRAM read works</summary>

A DRAM chip is split into independent channels, each channel into banks, and each bank into rows of 1–2 KB. To read, the controller first **opens** a row (copying it into the bank's row buffer, about 18 ns), then **reads** columns from it as bursts. More reads from an already open row are cheap. Switching to a different row means closing the current one first, another ~18 ns. So memory runs fastest when reads stay in already open rows, and when many banks work at once.

</details>

This is where knowing the model pays off. We know every weight and the exact order the chip will read them, so we can lay them out in DRAM in that order, spread across channels and banks. Nearly every read then lands in an already open row, and while one bank switches rows, another keeps streaming. There's one more trick: the weights are read-only after loading, while the DeltaNet memory and KV cache are written. Reserving a few banks in every chip for the written data means writes never close a weight row, without reserving any whole chip's bandwidth. (We considered dedicating one of the four memory chips to DeltaNet traffic and rejected it: that traffic is only about 10 GB/s, so the chip would sit 92% idle while the weights lost a quarter of their bandwidth. When memory sets the speed, never reserve bandwidth that might sit idle.)

## The biggest lever: speculative decoding with the model's own drafter

At batch 1 the multipliers are mostly idle, waiting for weights. **Speculative decoding** puts them to work: a cheap drafter guesses the next few tokens, and the real model checks all of them in one pass, for about the cost of generating one.

<details>
<summary>Background: how speculative decoding works</summary>

1. A cheap **drafter** guesses the next few tokens, say 3.
2. The real model processes all of them in **one** pass. Each weight read from DRAM is multiplied by 4 token vectors instead of 1. The weights are read once either way, so this step costs about the same as a normal one.
3. The guesses are accepted up to the first wrong one, plus one token the real model produces itself.

Every step yields at least one token, and the output is **exactly** what the model would have produced without speculation. Only the speed changes.

</details>

Qwen3.5-9B ships with its own drafter: the extra MTP layer, trained alongside the model. So no separate draft model is needed, with its own weights and memory traffic.

### The cost of a draft, and a trim

Each draft runs the MTP layer and then has to turn its output into a token through an output head. The full output head over 248,320 tokens is 572 MB per draft, about 12% of a whole main pass. That eats much of the gain. But drafts don't need the whole vocabulary. Using only the 32K most frequent tokens cuts the draft's output head to about 75 MB. If the right token isn't in that list, the guess is simply wrong and verification catches it. Output is still exact.

### Measured acceptance

The early plan assumed the drafter would be right 60–70% of the time, which is typical in published work. We measured it on the model's own generated text for chat, maths and code, using the exact chip number formats:

- **Draft 1** is right **90.0%** of the time.
- **Draft 2**, given draft 1 was right: **84.1%.**
- **Draft 3**, given both before it were right: **78.2%.**

Expected tokens per step with 3 drafts is 1 + p₁ + p₁p₂ + p₁p₂p₃ = 1 + 0.90 + 0.90 × 0.841 + 0.90 × 0.841 × 0.782, which is about **3.26 tokens per step** instead of 1. Maths and code are more predictable (about 3.5), chat less so (about 2.8). The chip's 4.5-bit format changed acceptance by less than 0.01 tokens per step. Comparing vocabulary sizes, 32K is the sweet spot: a bigger draft vocabulary costs more bytes than it gains in acceptance, and a smaller one misses too often.

A caveat that belongs in the headline: this was measured on the model's own greedy output, so it's an upper bound. Text written by people, or sampling with temperature, would accept less. Greedy decoding is the main use case for this chip, though.

### The hard part: rolling back DeltaNet memory

The KV cache is easy to roll back: write entries for all candidate tokens, then move the end pointer back past any rejected ones. The DeltaNet memory is harder. It's modified by every token, so if the chip verifies 4 tokens and only 2 are accepted, the memory must end up as it was after token 2, not token 4. Saving a full copy after every candidate is about 150 MB of extra writes per step.

Our approach is a **lazy commit**. During verification, don't write the new memory at all. Instead, log only the small inputs each token would have used to update it: k, v, β and the decay, about 6,000 numbers per layer per token. At the next step, when the memory is being loaded anyway, replay only the accepted updates, then carry on. It costs almost no extra bytes and a little extra compute, which is exactly what we have spare.

### A drafter that only looks at recent text

At long context, the drafter has a problem of its own: it's a full-attention layer, so it keeps its own KV cache, and at a million tokens each draft would read 2 GB of it. We tested whether the drafter still guesses well if it only sees the last few thousand positions. With a 4,096-position window, tokens per step dropped by just **1.8%**. The drafter mostly uses recent text. Its cache becomes an 8 MB ring buffer instead of 2 GB.

## The pivot: long context is the real workload

Partway through, I changed the target. The way I actually use models locally is long context: a 20–30K-token prompt (code, documents, tool output), and then working inside it. Almost nothing useful happens at 4K tokens. So:

- The headline score is now measured at **32K context.**
- Prefill (reading the prompt) became a hard target: **a 32K-token prompt in about 12 seconds.** The earlier floor was a 2K prompt in 4 seconds, about 500 tokens/s. The new one is about 2,800 tokens/s, more than 5× harder.

Decode at 32K barely suffers, thanks to the architecture: only 8 of 32 layers have a KV cache, so at 32K a token reads 5.1 GB instead of 4.6 GB, about 9% slower. Prefill is a different story, because **prefill is compute-bound.** Every weight has to be multiplied by every prompt token. A 32K prompt is 32,768 × 7.94 billion ≈ 2.6 × 10¹⁴ multiply-adds.

The matrix unit we'd sized for decode had 1,024 weight lanes × 4 token rows, 4,096 multiply-adds per cycle. At 1 GHz, 2.6 × 10¹⁴ multiply-adds takes 63 seconds for the weights alone, and the performance model said 78 seconds in all. To hit 12 seconds, the unit needs to grow to **1,024 lanes × 32 token rows** (32,768 multiply-adds per cycle), with a DeltaNet unit of 1,024 fp32 lanes. That gives about 11 seconds for a 32K prompt.

Why grow the *rows* and not the *lanes*? Because of decode. The lanes are matched to the memory. At 512 GB/s and about 92% utilization, about 471 GB/s arrives, and each weight is 9/16 of a byte, so about **837 weights arrive per nanosecond.** 1,024 lanes at 1 GHz takes 1,024 per cycle, with 20% headroom. More lanes would sit idle in decode. More rows instead let each arriving weight be reused across more prompt tokens, which only prefill and speculative verification need. In decode, the extra rows can be clock-gated.

The bigger unit also helps speculation at long context. Verifying 4 tokens in the attention layers needs more query rows than a 4-row unit has, so at long context the old design would have been compute-bound during verification. With 32 rows that never happens.

### A DeltaNet trick that halves the memory passes

The DeltaNet update as written needs several passes over each head's 128 × 128 memory: read it with the key, update it, read it again with the query. Since that memory lives in DRAM (50 MB doesn't fit on a cheap die), each pass is real traffic. But the algebra allows a shortcut. The new memory is g·S + k·deltaᵀ, so the output can be computed from the **old** memory:

> o = (g·S + k·deltaᵀ)ᵀq = g·(Sᵀq) + (kᵀq)·delta

That means a single pass over the old memory gives both Sᵀk and Sᵀq. Then compute delta, write the new rows, and correct the output with one number, k·q. One read and one write per head instead of three passes. We checked this reordering in the golden model over 32K tokens: the largest difference was at fp32 rounding level, and the top token was identical at every position.

## A performance model before any RTL

Before freezing the architecture, we wrote a performance model that runs the decode step as a list of operations: embedding row, then per layer the projections, convolution, DeltaNet or attention, output projection and FFN, then the output head. Each operation says how many bytes it reads, how many it writes and how many cycles its block needs. The schedule has two resources, DRAM and compute, with DRAM allowed to run ahead by a prefetch buffer.

A small example: the FFN down matrix is 12,288 × 4,096 = 50.3 million weights.

- **Memory:** 50.3M × 9/16 bytes = 28.3 MB, which takes 60 µs at 471 GB/s.
- **Compute for one token:** 50.3M ÷ 1,024 lanes = 49,152 cycles = 49 µs at 1 GHz. Memory is slower, so the operation takes 60 µs and the lanes are idle about 18% of the time.
- **Compute for verifying 4 tokens:** still 49 µs, because the 4 token rows share each weight as it arrives. Still 60 µs total. **The 3 extra tokens are free.** That one line is the whole case for speculative decoding on this chip.

Before trusting it, we checked that with compute made free, the model reproduces the hand-calculated numbers exactly. (The check found a bug in the model itself on the first try, which is the point of checking.) Its predictions for the base card:

- **Plain decode: about 101 tok/s at 4K context and 92 tok/s at 32K.** Memory is busy 100% of the time and compute limits under 2% of the time: memory-bound, as designed.
- **With speculation (3 drafts): about 277 tok/s at 4K and roughly 230–260 at 32K,** depending on the final row count.

These are predictions from a model that assumes 92% memory utilization, not measurements of silicon. The point of the model is to make design choices with numbers, and it settled several:

- **256 DeltaNet lanes are enough for decode.** 512 gains 2.6% for twice the lanes.
- **Keeping the DeltaNet memory on chip** (50 MB of SRAM, a big chunk of die) gains only 1.4%. Not worth it.
- **The short convolution's history** can stream from DRAM instead of living in SRAM, at no measurable cost.
- **Memory bandwidth is the biggest lever by far.** Swapping GDDR7 for 273 GB/s LPDDR5X roughly halves every number.

## First RTL: one slice of the matrix unit

With the architecture mostly settled by measurement and modelling, it was time to find out what the hardware actually costs. The first block was one **group** of the matrix unit: 16 weight lanes × R token rows. The full unit is 64 of these groups side by side (1,024 lanes). Each group is a 3-stage pipeline:

1. **Lookup and multiply.** Each 4-bit weight code picks an int8 value from the matrix's 16-entry table, then is multiplied by the int8 activation of every token row.
2. **Adder tree.** The 16 products of each row are summed exactly: 16 → 8 → 4 → 2 → 1.
3. **Scale step.** The integer sum is multiplied by the group's weight scale and the row's activation scale, giving a floating-point result.

This arrangement, summing the 16 products of a scale group across space in the same cycle rather than accumulating over time inside each lane, means scales are applied once per 16 multiplies in one shared place. The lanes themselves never touch a scale.

The width of the adder tree falls out of the number formats. Both weights and activations live in −127..127 (never −128), so each product is at most 127 × 127 = 16,129, and 16 of them at most 258,064. That fits in a 19-bit signed integer (limit 262,143). The test bench is what pointed this out: the tree's levels need only 16, 17, 18 and 19 bits, one bit less per level than first written.

### Testing: bit-exact or it doesn't count

Every hardware block gets a Python reference that computes the same thing, and the RTL output is compared against it **bit for bit**: tens of thousands of random vectors per configuration, plus deliberate worst cases (all +127 × +127, all +127 × −127, which push the adder tree to its limits). Result: zero mismatches.

A test that passes only proves something if it can fail, so Claude also planted bugs in copies of the RTL: rounding halves up instead of to even, an adder level one bit too narrow. The tests caught every one. The rounding conventions fixed here (round to nearest even, tiny results flush to zero, and so on) are now part of the chip's **bit-exact contract**: the golden model and the hardware must agree on them exactly.

### Synthesis: what it really costs

We synthesized with **ASAP7**, an open academic model of a 7 nm FinFET process, close to the class a real version of this chip would use, and scaled the results to TSMC's N6 with the published density difference.

<details>
<summary>Background: standard cells and how chip area is counted</summary>

Synthesis turns Verilog into a netlist of **standard cells**, the pre-made gates (NAND, XOR, full adder, flip-flop) a chip factory provides. The unit of size is a 2-input NAND gate, about 0.0875 µm² at 7 nm. For a sense of scale, an int8 × int8 multiplier is about 300 NAND-equivalents, and a square millimetre holds about 37,000 of them.

</details>

Three findings:

**The multipliers are cheap. The scale step isn't.** Per token row in one group: 16 multipliers are about 425 µm², the adder tree about 120 µm², and the scale step with a full fp32 activation-scale multiply about 320 µm². **The scale step was 37% of each row's logic,** more than the whole adder tree. It was also too slow: about 2.3 ns, against a 1 ns budget at 1 GHz, so it would need three pipeline stages. The multiply-adds everyone thinks of as "the AI chip" were not the expensive part.

**Fan-out can wreck timing.** The first synthesis of a 32-row group reported a 27 ns worst path (70 ns at 64 rows). The cause: each looked-up weight drives 32 or 64 multipliers, and plain gate mapping inserts no buffers to drive that load. With buffering and gate sizing switched on, the worst stage dropped to well under 1 ns. The lesson for the real design: high fan-out has to be planned for, with buffer trees or a register per block of rows.

**Our area estimate was 2–4× too pessimistic.** Before synthesis, we'd scaled Google's first TPU (65,536 int8 multiply-adds in about a quarter of a ≤331 mm² die at 28 nm) to 6 nm and estimated 5–10 mm² for a 32-row unit. Synthesis gave about **2.4 mm²**. Either way, it's a small part of an ~80 mm² die (the die size itself is still an assumption). This changed a decision: 64 token rows costs only about 2.5 mm² more than 32, roughly 40 cents of silicon, and halves prefill time. So 64 rows is now the leaning (pending a power measurement), giving a 32K prompt in about 6–7 seconds, well inside the 12-second target.

## A hardware finding that went back into the model

The expensive scale step raised an obvious question: does the activation scale really need full fp32 precision?

Here's what that scale is. Take 32 activations whose largest value is 3.81. Today's format stores scale = 3.81 ÷ 127 = 0.0300 as a full fp32 number, and each activation as round(value ÷ 0.0300), so 3.81 → 127. Two cheaper ideas change only how precisely 0.0300 is stored:

- **A bf16-precision scale** keeps an 8-bit mantissa: 0.0300 becomes 0.03003, off by 0.1% (at worst 0.4%). The multiplier shrinks to about a third.
- **A power-of-two scale** rounds up to the next power of two: 2⁻⁵ = 0.03125. Now 3.81 → round(3.81 ÷ 0.03125) = 122 instead of 127, so a little of the int8 range goes unused. In exchange, "multiply by the scale" becomes "shift the binary point by 5 places", which costs almost nothing. Rounding *up* means nothing ever clips.

Back to the golden model and the GPUs. On its own, with full-precision weights, the power-of-two scale **doubled** the activation error on code (KL 0.00102 → 0.00216), a real and clearly measurable effect. But with the chip's actual 4.5-bit weights, the change was **+0.0002 ± 0.0006**, against a total of 0.056: invisible. Errors add up roughly by size, and the weight error is about 50 times larger than the extra activation error, so doubling the small one barely moves the total. Real tasks agreed: GSM8K 86.88% with power-of-two scales vs 86.66% with fp32 (no significant difference), and HumanEval within a couple of problems.

So the activation scales became powers of two. In hardware, the scale step shrank by **84%** (320 → 51 µm² per row) and now fits in a single 1 GHz cycle instead of three. A 32-row group got 27% smaller.

It also unlocked a second saving. After the groups, the 64 group results of each 1,024-lane beat have to be added into one number per row. With fp32 scales, that's a tree of 63 floating-point adders per row, rounding at every step. With power-of-two scales, every group result is an exact integer times a power of two, so they can be lined up and added **exactly** in fixed point. A small example with 4 groups: 300 × 2⁻⁸, −45 × 2⁻⁶, 7 × 2⁻⁸ and 120 × 2⁻⁷. Line everything up to 2⁻⁶: 75, −45, 1.75 and 60, which sum exactly to 91.75 × 2⁻⁶. Convert to floating point once at the end. The hardware version keeps 16 extra fraction bits in a 46-bit adder tree, and drops anything below that by rounding down (a rule that only matters when groups in one beat differ in scale by more than 2¹⁶, and that's now part of the bit-exact contract).

The fixed-point **beat reducer** came out **3.2× smaller and 3.2× faster** than the floating-point tree. Its first version also taught a lesson about writing hardware: the max search and the sum were written as loops, which synthesized into a 64-step chain taking 15–25 ns. Rewritten as explicit balanced trees six levels deep, the same bits come out in about 4 ns. Together, the power-of-two decision made the whole matrix unit about **a third smaller** and much easier to clock at 1 GHz.

This is the loop working the way it should: a cost discovered in synthesis turned into a question for the model, the model answered it with measurements, and the answer changed both the number format and the hardware.

## Smaller pieces, settled the same way

A few more decisions used the same pattern of measuring first.

**The special-function unit** computes eˣ, sigmoid, SiLU, 1/√x, 1/x and softplus for softmax, gates and normalization. We emulated each at reduced precision in the golden model. eˣ was by far the most sensitive, and its sensitivity grows with context: softmax adds up thousands of eˣ values, so small errors accumulate. A 512-token test hid this entirely, and only the 8K-token run showed it. With the chip's numerics, 10 bits everywhere was invisible and 6 bits was a clear loss. These units are tiny next to the matrix unit, so we chose generously: 16 bits for eˣ and 1/√x, 12 for the rest, with eˣ at 128 results per cycle so long-context attention and prefill don't stall on it.

**Attention runs on the same int8 matrix lanes** as everything else, rather than a separate floating-point attention unit. That needs the softmax probabilities P to be int8. The obvious choice, a fixed grid of steps of 1/127, failed badly: at 32K context the KL was over 1.0 and top-1 agreement fell to 69%, getting worse with distance. At long context most attention probabilities are tiny, and a fixed grid rounds them all to zero. Giving P a power-of-two scale per 32 positions, the same scheme as every other activation, fixed it completely: KL 0.0007 at 32K, flat with distance. No second datapath needed.

**On-chip SRAM comes to about 6.5 MB.** The dataflow keeps it small: weights stream in tiles of 64 outputs so each input slice stays in the lanes for 64 beats (SRAM then supplies about 1 KB per cycle instead of an impossible 64 KB), projections are produced head by head so only one head's worth is on chip at a time, and the FFN's gate and up rows are interleaved so the 12,288-wide intermediate only ever exists in int8. Prefill runs in 128-token chunks. At 7 nm-class density, 6.5 MB is roughly 2–2.5 mm².

**The output head never stores its 248,320 logits.** For greedy decoding, the chip keeps a running best value and its token index per row as the logits come out of the matrix unit: one compare per logit. Ties go to the lowest token ID, as part of the bit-exact contract.

## Where things stand

Up to now, the work has gone from a model choice to a mostly settled spec, with nearly every decision backed by a measurement:

- **Goal and score:** the cheapest tokens per second. Decode tokens per second per dollar of card cost for one user, measured at 32K context, with a 32K-token prompt in about 12 seconds as the prefill target.
- **Model:** Qwen3.5-9B, text only, with its own MTP drafter.
- **Numbers:** 4.5-bit weights (GPTQ, a learned int8 table per matrix, E4M3 scale per 16), int8 activations with power-of-two scales, int8 KV, fp32 DeltaNet memory. −1.9 points on maths, no measurable change on code.
- **Memory:** four 2 GB GDDR7 chips, 512 GB/s, an 8 GB base card and a 16 GB long-context card.
- **Speculation:** 3 drafts per step from the model's own MTP layer, about 3.26 tokens per step, lazy-commit rollback for DeltaNet, a 4K-position window for the drafter.
- **Matrix unit:** 1,024 weight lanes × 32 or 64 token rows at 1 GHz, with the first group and the beat reducer written, bit-exact and synthesized at 7 nm.
- **Predicted speed:** about 92 tok/s plain at 32K context and well over 200 with speculation, from the performance model.

What isn't known yet matters as much:

- **Power** hasn't been measured. That needs place and route and switching-activity simulation, not just synthesis. It's also what will settle 32 vs 64 rows.
- **The die size and cost** are still estimates built on an assumed ~80 mm² die and fast-moving memory prices.
- **The competitor bar** rests partly on an unmeasured RTX 5060.
- **Speculation acceptance** under sampling and on human-written text, and quality past 128K context, are still open.

Next comes the instruction set: Qwen3.5-9B written as a program for the chip, and a second version of the golden model that copies every rounding rule in the bit-exact contract, so the RTL can be checked against it token by token. That's the subject of part 2.
