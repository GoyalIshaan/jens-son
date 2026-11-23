# Scale-out

A chip is only part of the machine. Roune calls parallelization tricky and unintuitive, and he admits he and others got it wrong for a long time, leading to an overprovisioned TPU network. [R] Take this chapter slowly.

::viz torus

## The concepts

- **Throughput**: tokens per second across all users.
- **Token latency**: ms per token for one user. Different from first-token latency.
- **Local vs global**: per-chip against total. A 1 TB model does fit on 20 chips with 100 GB each.
- **Minimum installation**: chips required before anything runs.
- **Failure domain**: what stops working if one part fails.

Latency and throughput are not related in any simple way. Almost all published material quotes one without the other, which makes the number close to meaningless. [R]

## The parallelism table

This is the core of the chapter. S = pipeline stages, P = within-layer ways, D = copies. [R]

| Type | Throughput | Token latency | Weights per chip | KV per chip | Network need | Failure domain |
| --- | --- | --- | --- | --- | --- | --- |
| Duplication ×D | ×D | Same | Same | Same | Little | Better |
| Pipeline ×S, decode, partial | Under ×S | Up | ÷S | Same | Critical | ×S |
| Pipeline ×S, decode, full | ×S | ×2 | ÷S | ×2 | Only needs to finish in a layer time | ×S |
| Pipeline ×S, prefill | ×S | ×2 if full | ÷S | ÷S | Low | ×S |
| Within layer ×P | ×P | ÷P, roughly | ÷P | ÷P | High | ×P |

## Three surprising facts

**1. Network bandwidth limits within-layer parallelism.** Between-layer traffic is far lighter.

**2. Double the network and you can as much as halve token latency,** through twice the within-layer split. So large arrays, which raise batch and latency, are a force for faster networks.

**3. Pipelining helps weights but not decode KV cache.** Each chip stores only its layers' keys, which divides by S. But you now need S times as many conversations in flight to keep all stages busy. The two factors cancel. Prefill avoids this, since stages can work on spans of one document. [R]

## Within-layer math

Attention splits by heads: GQA groups stay together, giving maybe 8 ways, and splitting the KV across chips adds maybe 8 more, for about 64 total. Feed-forward splits on the wide hidden dimension: 16k × 4 ÷ 256 = 256 chips at full array utilization. [R]

Each layer ends with an all-reduce. For data size X on P chips:

`bytes per chip = 2X(1 − 1/P)`

That barely depends on P. So why does doubling P more than double bandwidth need? Because each chip now finishes in half the time, so all-reduces arrive twice as often. Per-chip network rate scales with P. [R]

**Hiding it.** Run two half-batches: one computes while the other communicates. That doubles latency and KV, and removes network waits. Suppose you are 8-way with 25% network overhead. Pipelining lets you go to 32-way with no waits. Latency doubles, then falls 4×: net half the latency. [R]

**Precision on the wire.** 32-bit sums need twice the bandwidth of 16-bit sums. [R]

## Mixture of experts

Each token goes to a few of many experts. Mixtral uses 2 of 8. Llama 4 Maverick has 128 experts with 17B of 400B weights active, about 4%. [R] Problems:

- **Uneven load.** A 120% load on an expert needs two copies, each 60% busy. 1020% needs 11 copies at 93%. Large deployments smooth this out. [R]
- **Not enough rows.** Each expert sees only its share of the batch, so it struggles to fill an array.
- **All-to-all traffic.** Tokens travel to arbitrary chips.

On a single chip at batch 1 these problems mostly vanish, and only the storage cost stays. *Nonlinear units and MoE* works that case through.

Priority queues help: high-priority tokens go first, and batch work fills spare capacity. [R] Forced even routing gives full utilization at a cost in answer quality, since a token may get the wrong expert. [R]

## Link types

Shorter links give more bandwidth per dollar. In order: [R]

1. Same die
2. Silicon bridge or interposer
3. Package pin
4. Passive copper trace
5. Active copper trace
6. Board-to-board connector
7. Backplane
8. Flexible copper cable
9. Optical cable

Water cooling lets boards sit closer, so links get shorter and cheaper. Air cooling and routers push links longer. [R]

Be suspicious of any single bandwidth number. Is it bits or bytes? One direction or both? Giga or gibi? Line rate or payload? Theoretical or measured? [R] Jouppi quotes per-direction numbers, and explains others would double them. [J]

## The efficiency formula

`useful bandwidth per chip = link utilization × total link bandwidth ÷ average hops`

A router is a node, and it is not a useful destination. Every bit traverses chip → router → chip, so average hops is 2. **A single router wastes half its bandwidth.** [R] Multiple levels waste more. In a torus where only neighbors communicate, hops can be 1.

All-gather, reduce-scatter and all-reduce can all run with only neighbor traffic. So for dense models a torus runs twice as fast as a router for the same link hardware, with no router to buy. [R] A mesh without wraparound links runs at half the speed of a torus. [R]

## Topologies

| Topology | Hops, neighbors | Hops, random | Cost | Used by |
| --- | --- | --- | --- | --- |
| Single router | 2 | 1.5 at 4 nodes, 2 at many | Router plus longer links | NVLink domain |
| 1D torus, 4 wide | 1 | 1 | None | Early NVLink, Roune's ring boards |
| 4×4 torus | 1 | 2 | None | Roune's donut |
| 4×4×4 torus | 1 | 3 | None | Google's cube |
| 16×16×24 torus | 1 | 14 | None | TPU v5p large slice |
| Dragonfly | Groups fully connected | Small | High-radix | Groq, future TPUs |

Random traffic gets worse with torus size, since average distance per dimension is width ÷ 4. So MoE is fine on small tori and poor on huge ones. [R] Tokens going to many experts, or within-expert parallelism, shift things back toward tori.

Nvidia's NVL72 gives every GPU a single hop to every other at full bandwidth, over passive copper. It is the best in performance and flexibility, and worst in cost and reliability. [IA] Google's torus is more reliable and cost-effective, but it asks compilers to handle non-uniform topology. [IA]

## Google's optical circuit switches

Rack: 4×4×4 = 64 chips in electrically wired cubes, a size chosen because 3D cubes have the best bisection and 64 chips plus 16 hosts fit in a rack. [J] Inside a cube, links use PCB traces and copper cables, 144 in all, and only cube faces go optical. [IA, D] Each cube face has 16 links, so 96 optical links per cube. Opposing faces go to the same switch, so 48 switches connect all cubes. Each switch has 136 ports: 128 plus 8 spares. [J]

MEMS mirrors switch in milliseconds, which sounds slow. They are not routing packets. They rewire topology around failures and between jobs. Benefits: [J]

- **Fault isolation.** Route around a bad cube.
- **Scheduling.** A 128-chip job takes any two cubes, not 128 contiguous chips.
- **Incremental deployment.** A cube works once its 64 chips are installed.
- **Odd sizes.** That is why Ironwood has 9,216 chips: four 2K jobs plus 16 spare cubes.

Roune considers this more expensive than necessary, and proposes a cheaper design: 4×4 tori on copper, connected through cheap low-bandwidth routers, in an N×4×4 topology. [R] Whether that wins depends on cost data neither of us has.

## Pod shape, recovered from a table

A torus bisection is 2 × (smallest cross-section) × link bandwidth. Using that on Table 1 recovers shapes the paper never states. [D]

| Generation | Chips | Shape that fits | Bisection check |
| --- | --- | --- | --- |
| TPU v2 | 256 | 16×16 | 2 × 16 × 62 = 1,984 GB/s |
| TPU v3 | 1,024 | 32×32 | 2 × 32 × 70 = 4,480 |
| TPU v4 | 4,096 | 16×16×16 | 2 × 256 × 50 = 25,600 |
| TPU v5p | 8,960 | 16×20×28 | 2 × 320 × 100 = 64,000 |
| Ironwood | 9,216 | 16×24×24 | 2 × 384 × 100 = 76,800 |

All five match the published bisection exactly. That is a strong check, though these shapes are inferred and not confirmed by the paper.

## Reliability

Parallelism multiplies failure rates: if one of 1,024 chips stopping halts the job, the system is 1,024 times as fragile. In early TPU pods, one failed cable took down a whole pod until someone walked over to replace it. Sameer Kumar's fix made software work around missing links, at half the torus bandwidth, which happened to be available because the network was overprovisioned. [R] Accidental margin turned out to be useful.

Silent data corruption is harder, because nothing fails visibly. Ironwood's defenses: FBIST inside the MXU for functional self-test, and hardware replay on the VPU. [J]

The goodput target is 93% across multiple pods. Five features support this: better nodes, error checking, checkpoint and restore, strict deterministic repeatability, and modular isolation. [J]

## Simulate this

Roune has a PhD in computer algebra, and he got this math wrong for an embarrassingly long time. He recommends simulating parallelism, not only using a spreadsheet. [R] Even simulation is not enough: TPU v3's first silicon could not drive its cables at all, despite simulation, and a week of work by the firmware and hardware teams saved the launch. [R]
