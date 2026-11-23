# 20. The weight format, explained from scratch

*Written 2026-11-18, as a blog draft. It explains decision D-005 (decided 2026-11-12): what the format is, which experiments back it, and why we picked it. Raw numbers and rerun commands are in `15_measurements.md` (experiment sets 1–3).*

## The one-line version

- Each weight is a 4-bit code that picks one of 16 values from a small table.
- Each matrix has its own table, and its 16 values are 8-bit integers.
- Each group of 16 weights shares one 8-bit scale.
- Cost: 4.5 bits per weight.
- Result: −1.9 points on school maths (GSM8K) against the original model, no measurable loss on code. Plain 4-bit loses 7.5 points on GSM8K.

## Why the bit count matters

- Qwen3.5-9B has about 9 billion weights. The original release stores each one in 16 bits (bf16), about 18 GB in all.
- For every generated token, the chip reads almost all of the weights from memory. Memory reads are the slow part, so tokens per second is roughly memory speed ÷ bytes per token.
- Fewer bits per weight means more tokens per second. The catch is quality: squeeze the weights too hard and the model starts getting answers wrong.

## How one weight is stored: a worked example

Take a group of 16 weights from one matrix:
- `0.12, -0.30, 0.05, 0.01, -0.02, 0.21, ...`

**Step 1: the group scale.**
- The largest absolute value in the group is 0.30. Store it once as an 8-bit float (E4M3).
- Divide every weight by it, so they all land between −1 and +1:
  - `0.40, -1.00, 0.17, 0.03, -0.07, 0.70, ...`

**Step 2: the table lookup.**
- Each matrix has 16 allowed values, for example:
  - `-1.00, -0.72, -0.53, -0.38, -0.26, -0.15, -0.06, 0.00, 0.06, 0.15, 0.26, 0.38, 0.53, 0.72, 0.87, 1.00`
- Replace each weight with the number of its nearest entry, 0 to 15. That number is the 4-bit code.
  - 0.40 becomes entry 11 (0.38).
  - −1.00 becomes entry 0.

**Step 3: rebuilding a weight on the chip.**
- weight ≈ table[code] × group scale
- For example, 0.38 × 0.30 = 0.114, close to the original 0.12.

**Where 4.5 bits comes from:**
- 16 codes × 4 bits = 64 bits
- plus one 8-bit scale = 72 bits
- 72 bits ÷ 16 weights = 4.5 bits per weight

## The three ideas that make it work

**1. A learned table instead of evenly spaced levels.**
- Plain 4-bit uses 16 evenly spaced levels: −8, −7, …, 7.
- Real weights have a bell-shaped distribution: most sit near zero and few sit at the edges.
- So for each matrix we run k-means clustering to place its 16 levels where its weights actually are. That puts more levels near zero and fewer at the edges.
- Same 4 bits, smaller average error.

**2. GPTQ rounding instead of "round to nearest".**
- GPTQ rounds one column of weights at a time and measures the error that causes on real input text.
- It then nudges the weights it hasn't rounded yet to cancel that error, so errors cancel out instead of piling up.
- It runs once, offline. The chip pays nothing for it.

**3. Table entries stored as 8-bit integers.**
- Each value is stored as a whole number from −127 to 127. For example, 0.38 is stored as 48, since 48/127 ≈ 0.378.
- The activations (the numbers flowing through the model) are int8 too, so every multiply on the chip is a cheap 8-bit × 8-bit integer multiply.
- Within a group of 16, the products add up exactly in a 19-bit integer adder. Floating point is only needed once per group, to apply the scales.

## Experiment A: how different are the outputs? (KL divergence)

- **Method:** feed the same text (Python code and WikiText) through the original model and the quantized model.
  - At every position, each model gives a probability for every possible next token.
  - KL divergence measures how far apart those two probability lists are. 0 means identical; bigger means more different.
- **Result for our format:** KL 0.050 on code. The two models pick the same top token 95.6% of the time.

## Experiment B: how precise must the table values be?

- **Method:** keep everything fixed and change only how the 16 table values are stored.
- **Results (KL on Python code):**
  - full 32-bit: 0.0500
  - 8-bit integer: 0.0500
  - 6-bit integer: 0.0504
  - 8-bit float (E4M3): 0.0518
  - 5-bit integer: 0.0638
- **The noise check:** an exact rerun of the 32-bit setup gave 0.0516, so running the same thing twice moves the score by about 0.002. Anything within that range is noise.
- **Reading:** 8-bit and 6-bit are as good as 32-bit. 5-bit is about 25% worse, a real jump.
- **Why 5 bits breaks:**
  - The closest two table levels are 0.077 apart.
  - With 8 bits, the worst rounding error is 0.004, tiny next to that gap.
  - With 5 bits, the step is 1/15 ≈ 0.067, nearly the size of the gap itself, so levels start to blur together.
- **Why 8 over 6:** both are free on quality, and 8 matches the int8 multipliers the chip already has.

## Experiment C: do the answers still come out right?

A low KL is encouraging, but what matters is whether the model still gets answers right. So three models answered the same problems:
- the original model (bf16)
- plain 4-bit (int4 with a scale per 32, rounded to nearest)
- our chip format

**Results:**
- GSM8K (1,319 school maths word problems): original 89.3%, ours 87.4%, plain 4-bit 81.8%
- HumanEval (164 coding problems, run against tests): original 69.5%, ours 70.7%, plain 4-bit 65.9%
- MBPP (500 coding problems): original 55.8%, ours 55.4%, plain 4-bit 54.2%

**How we tell a real drop from luck: paired comparison.**
- Both models answer exactly the same problems, so we count only the problems where they disagree.
- GSM8K, ours vs the original:
  - lost 63: the original got these right, ours got them wrong
  - gained 38: ours got these right, the original got them wrong
  - net −25 out of 1,319 = −1.9 points
- If the difference were luck, lost and gained would be about equal. A 63-vs-38 split happens by chance about 1.7% of the time (p = 0.017). **So the 1.9-point loss is real, but small.**
- Plain 4-bit vs the original: lost 133, gained 34, a net −7.5 points. The chance of that by luck is about 5 in a quadrillion (p ≈ 5e-15).
- Code, ours vs the original: +1.2 on HumanEval (p = 0.82) and −0.4 on MBPP (p = 0.89). At those p-values the difference is no bigger than chance, so there's no measurable change.
- Ours vs plain 4-bit on GSM8K: +5.6 points (p < 1e-7).

**Two details worth a blog paragraph:**
- **Weight error didn't predict the maths result.** Plain 4-bit and our format have about the same weight error (around 10%), yet they differ by 5.6 points on GSM8K. Maths is a long chain of steps, and one wrong token early on ruins the final answer, which a per-weight error number can't show.
- **Identical answers:** our format produced byte-identical text to the original on 23% (GSM8K), 40% (HumanEval) and 35% (MBPP) of problems. Plain 4-bit managed 9%, 22% and 20%.

## Why this is the best choice (and what "best" means)

"Best" here means the best quality for the bits, among the options we measured, on a chip whose speed is limited by memory.

- **More bits would cost speed.** At 4.5 bits, a token reads about 5 GB of weights. At 8 bits it would read about 9 GB, so roughly half the tokens per second, just to win back 1.9 maths points. For a chip judged on tokens per second per dollar, that's a bad trade.
- **Plain 4-bit costs about the same bits but loses 4× as much on maths** (7.5 points against 1.9). The learned table and GPTQ fix most of the damage for nearly no extra bits.
- **The hardware stays cheap.**
  - Each table is 16 bytes per matrix, about 4 KB per token in all.
  - Every multiply is integer int8 × int8.
  - Each group of 16 sums exactly in a 19-bit adder.
- **Each piece was measured separately** (rounding method, table, group size, table precision). The alternatives we tried were worse, or no better for the bits.

## Honest limits

- The maths loss is real, not zero.
- All three benchmarks used greedy answers (always taking the top token). Long chats and very long contexts aren't tested yet; the 128K-context check is pending (experiment set 4).
- "Best" means best among what we measured, not best of every format that exists.
