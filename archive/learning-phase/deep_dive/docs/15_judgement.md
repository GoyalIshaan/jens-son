# Judgement

Roune's co-design chapter may be the most valuable part of his document, and it has nothing to do with transistors. It is about how smart people get fooled by their own numbers. [R]

## Derived consequences

If you want to know what X does, you must include what else changes because of X. This is the most common co-design error Roune has seen. His examples:

**Turning off op fusion.** Measuring XLA with fusion off shows a huge effect. Yet XLA breaks operations into tiny pieces precisely because fusion exists. A system without fusion would be designed differently, so the number tells you almost nothing.

**Do we need more scalar compute?** You crank up the array clock in a simulator and see a scalar bottleneck. But a compiler team faced with that bottleneck would unroll loops, and the need could vanish.

**The address unit.** TPU v2 added hardware for address math based on loops that had not been unrolled. Unrolling made those addresses constants, and the hardware went unused.

**Dynamic shapes.** "Impossible" was true for today's practices, and false for practices that would have changed in response.

The question to ask every time: *how would software change in response to this hardware decision?*

## "It doesn't work"

There is never just "it works" or "it doesn't work". There is only "it does or doesn't work when tried this way." [R] Trying 2:4 sparsity by zeroing small weights gives bad quality. Nvidia's recipe with retraining works. The reverse also applies: something that works only with heroic effort will not matter if your customers cannot do it.

## Omitted factors

How can every startup be faster than Blackwell? The numbers are usually correct. Something is left out. [R]

- One favorable benchmark out of thousands.
- The competitor's parallelism turned off.
- A 1,000-chip module compared 1:1 with one chip.
- 2× the performance at 10× the cost.
- Best of 1,000 runs against one run.
- A peak speed that lasts seconds before throttling.
- A low-accuracy mode that only works on one model.

Roune's conclusion is deliberately harsh. Numbers are never evidence on their own. You have to prove the meaning, not only the numbers. Being data-driven with no intellectual quality is a way to feel better about making random decisions.

## Reading a spec sheet

Run these checks on anything you read.

| Check | Example from these sources |
| --- | --- |
| Units | Table 1's pod HBM row says petabytes, and it is terabytes |
| Precision | "100×" compares BF16 to FP8. Same-precision gain is 50× |
| Sustained or peak | Perf per TDP watt uses peak FLOPS |
| Per direction or total | Jouppi quotes per direction, others double it |
| Dense or sparse | 2:4 doubles peak on paper |
| Clock check | Peak ÷ (2 × MACs) should give a believable clock |
| Scale | Per chip or per system |
| Who benefits | Irrational Analysis discloses a large Nvidia position |

The clock check is especially useful. It takes ten seconds, and it exposes an inconsistency in the Ironwood table. See *Source audit*.

## Hill climbing

A normal process picks a plan of record, then accepts changes that improve it. That is hill climbing, which gets stuck at local optima. [R] Two mechanisms keep you stuck.

**Bloat protects bloat.** Each overprovisioned feature looks like a small share of cost, so removing one seems unimportant. Remove them all together and cost falls a lot.

**The 1/x effect.** Value per dollar depends on 1/cost, and that is nonlinear. Two changes that each save $0.50 on a $1.01 item give 2× each, and 100× together. Three ideas rejected over three years might have been approved if presented together.

Roune's proposed answer is combinatorial search across hardware and software choices. Your model will be imperfect, and you will stop trusting it. Use it for ideas and sensitivity analysis. When it gives something ridiculous, find out why. [R]

## Working with software

- Hardware should never design a software-facing interface alone. Performance is software-facing. Every limit is software-facing.
- Do not let formal process squeeze out hallway conversations. Hardware engineers guessing what software needs is the worse outcome.
- Do not just defer to software either. Limits need both views: respectful negotiation of costs and benefits.
- Share one sheet of all capacities, bandwidths and limits. Roune's got thousands of views.

## Reading these sources critically

**Jouppi et al.** A primary source with authority on TPU facts. It is written by Google engineers and argues for TPU's design choices. It compares peak numbers against TDP, and the authors acknowledge that goodput-based measures would be better.

**Roune.** A strong practitioner with direct TPU and GPU experience, and honest about what is opinion. He calls his claims informed opinions, and agrees they should be verified. His AI-CPU thesis is a prediction, unproven at scale. Some of his positions, like the advantage of structured sparsity, are disputed by practitioners in his own objections section.

**Irrational Analysis.** Insightful framework, profane and declared unfinished, with several chapters as TODO. The author holds large Nvidia positions, and says to treat that as bias. Treat claims about Nvidia's future plans as speculation, and claims about Cerebras as one author's opinion.

**The encyclopedia.** A synthesized reference, broad and careful about limits. It explicitly does not give numbers. Good for vocabulary.

Your factbook adds dated evidence for the products after these sources, and I used it for that.

## Key habits

1. Name the dataflow before using a formula.
2. Check units before believing a ratio.
3. Compute intensity before buying compute.
4. Ask what software would do differently.
5. Compare at system level, per realized performance.
6. Keep a list of which of your numbers are measured.
