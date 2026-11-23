# Generations

This is how real designers improved their architectures. The best way to learn reasoning is to watch the same machine move through eight years of changing constraints. Google published the numbers, so we can check what happened.

::viz generations

## The TPU table with derived columns

Published values from Jouppi's Table 1 [J], and derived ratios [D].

| | v2 | v3 | v4 | v5p | Ironwood |
| --- | --- | --- | --- | --- | --- |
| Year | 2017 | 2018 | 2021 | 2023 | 2025 |
| BF16 TFLOPS | 46 | 123 | 275 | 459 | 2307 |
| MXUs | 2×128² | 4×128² | 8×128² | 8×128² | 4×256² (see audit) |
| Implied clock, GHz | 0.70 | 0.94 | 1.05 | 1.75 | about 2.2 |
| VMEM MiB | 32 | 32 | 32 | 128 | 128 |
| HBM GiB | 16 | 32 | 32 | 96 | 192 |
| HBM GB/s | 700 | 900 | 1200 | 2765 | 7300 |
| Links × GB/s | 4×62 | 4×70 | 6×50 | 6×100 | 6×100 |
| Pod chips | 256 | 1024 | 4096 | 8960 | 9216 |
| **Ridge, FLOP/byte** | 66 | 137 | 229 | 166 | 316 |
| **VMEM MiB per TFLOP** | 0.70 | 0.26 | 0.12 | 0.28 | 0.06 |
| **HBM GiB per TFLOP** | 0.35 | 0.26 | 0.12 | 0.21 | 0.08 |
| **ICI GB/s per TFLOP** | 5.4 | 2.3 | 1.1 | 1.3 | 0.26 |
| **Per-chip TDP, relative** | 1.0 | 1.4 | 1.25 | 1.9 | 3.4 |

Read the bold rows from left to right. There is a clear pattern: compute grew faster than everything else, with one corrective generation.

## What changed at each step, and why

### TPU v1: the inference chip

A nearly pure int8 array driven by a host, with DRAM as its bottleneck. [J, R] Motivation was modest: speech synthesis would have been too expensive to run on existing hardware. [R] It gave 30× the performance per watt of contemporary GPUs and showed the idea worked. [J]

### v1 to v2: becoming a computer

Training changes everything. v2 added BF16 multiplies with FP32 accumulation, HBM for about 30× more memory bandwidth, vector units, a VLIW scalar core and ICI links. [J] v1's single 256-wide int8 array became two 128-wide BF16 arrays. The team was about 30 people. [R] Many things were overprovisioned, including the network. [R]

This block diagram still describes Ironwood. Getting that structure right was the most valuable decision in the line.

### v2 to v3: the cheap upgrade

Doubling MXUs, a clock increase of 34%, liquid cooling. HBM stacks doubled, while bandwidth rose only 29%. Pod grew 4×. Peak rose 2.7× while HBM bandwidth rose 1.3×, so the ridge doubled, using up some v2 margin.

The clock increase came with trouble: thermal surprise, a compiler fix worth 25% of clock, and a networking bug in first silicon. [R]

### v3 to v4: the scale-out generation

Doubling MXUs again, but the big changes were at system level. 3D torus, optical circuit switches, SparseCore count doubled. Link count rose from 4 to 6, with per-link speed dropping from 70 to 50 GB/s. [J] That is the kind of move that only makes sense if topology matters more than raw link speed. Megacore let the compiler treat two cores as one. [J]

### v4 to v5p: the correction

MXU count stayed the same, and clock jumped 67%. VMEM quadrupled, HBM capacity tripled, bandwidth rose 2.3×, link speed doubled. [J]

This is the rebalancing generation. Ridge fell from 229 to 166, the only time it fell. VMEM per TFLOP more than doubled. The v4 machine had drifted out of balance and v5p corrected it, perhaps because large models had grown and needed the capacity. This is my reading of the numbers; the paper does not say so.

### v5p to Ironwood: the big jump

Array side doubled to 256, with native FP8. BF16 peak rose 5×. Other changes: [J]

- Two compute dies per package. Eight HBM3E stacks.
- Vector registers from 8×128 to 16×256.
- Four general ALUs per lane, up from two restricted ones.
- Redundant MXU row for yield.
- FBIST and hardware replay.
- SparseCore 2.4× faster, and more general.

Ridge jumped back up to 316. ICI per TFLOP fell 5×, because link speed stayed at 100 GB/s. Per-chip TDP nearly doubled.

Peak performance per TDP watt rose about 6× in this one step, far more than in any previous generation. [J] That mostly reflects bigger arrays, FP8 and a newer node. It uses peak numbers, as the paper notes, and the headline 100× per-node figure compares BF16 to FP8. At equal BF16 precision, it is 50×. [J]

### After Ironwood: the split

Your factbook records that TPU 8t and 8i split training and serving into separate designs. [F]

| | 8t (training) | 8i (serving) |
| --- | --- | --- |
| Topology | 3D torus | Boardfly |
| Vector SRAM | 128 MB | 384 MB |
| HBM | 216 GB | 288 GB |
| HBM bandwidth | 6.5 TB/s | 8.6 TB/s |
| Special block | SparseCore | Collectives Acceleration Engine |

Everything earlier in this guide predicted this split: serving wants more memory and more SRAM, and MoE traffic favors something other than a huge torus.

## What stayed the same

Equally important. Across five generations, these did not change: [J]

- Two TensorCores per chip.
- Four chips per board.
- Systolic arrays, scratchpad, DMA, VLIW.
- XLA as the compiler.

The paper's central claim is that this stability is a feature. Compilers and models carry forward, and tuned models are ready on day one. [J]

## Nvidia's parallel path

| Generation | Year | Change | Source |
| --- | --- | --- | --- |
| Volta | 2017 | First tensor cores, 4×4 | [R] |
| Turing | 2018 | 16×16, int8 | [R] |
| Ampere | 2020 | 2:4 structured sparsity | [R] |
| Hopper | 2022 | FP8 | [R, O] |
| Blackwell | 2025 | 64×64, FP4, NVFP4, two dies | [R] |
| Blackwell Ultra | 2025 | 288 GB HBM3E, 8 TB/s, 15 PF dense NVFP4 | [IA] |

The array sizes are Roune's description. Nvidia does not publish tensor cores in these terms, so treat them as his model of the structure.

Nvidia's scale-up link has its own steady curve, from a slide in [IA]:

| NVLink | Year | GB/s per GPU | Domain |
| --- | --- | --- | --- |
| 1 | 2016 | 160 | Hybrid cube mesh |
| 2 | 2018 | 300 | First NVSwitch, 8 GPUs |
| 3 | 2020 | 600 | Ampere |
| 4 | 2022 | 900 | Hopper |
| 5 | 2024 | 1,800 | Blackwell, 72 GPUs all-to-all |
| 6 | 2026 | 3,600 | Rubin |

Compare this with TPU's ICI, whose per-chip total went from 248 to 600 GB/s over the same period. Nvidia spends far more on per-chip link bandwidth. Google spends on topology instead, and keeps links neighbor-only. Those are two different answers to the same parallelism problem. [D]

Where TPUs scaled a fixed design, Nvidia kept a general programmable core and added AI features to it. Over time it took on TPU-like traits:

- Larger matrix units, from 4×4 toward 64×64.
- Tensor Memory Accelerator for bulk asynchronous copies, which is effectively DMA.
- Tensor Memory, giving accumulators dedicated storage outside the register file. An SM has 4 partitions, each with a 16,384 × 32-bit register file and 64 KB of tensor memory, and 256 KB of shared L1. [IA]
- Warp specialization, which is close to separating data-movement and compute threads. [R]

Meanwhile TPUs became more programmable, with general ALUs, Pallas kernels and C++-like control. The two architectures are moving toward the same place from opposite sides. Roune sees the endpoint as a CPU with a large array. [R]

Nvidia led on numerics and sparsity from FP8 on, while Google led earlier with large arrays and BF16. [R]

## Groq's different bet

Groq chose determinism as its organizing idea. Hot Chips 2022 describes a chip viewable as a 144-wide VLIW, with 320-element vectors built from 20 superlanes of 16 lanes each, and about 220 MB of flat SRAM. [IA] LPU3 reaches about 500 MB. [R] Nvidia licensed the technology and hired the team for about $20B, per Irrational Analysis. It is positioned for premium, low-latency decode FF, with GPUs doing attention and prefill. [R, IA] Your factbook has Groq 3 LPX in production as of August 2026, with 256 LPUs per rack. [F]

## Principles you can extract

**1. Fix the interface. Change the implementation.** Software sees stable abstractions, while the implementation scales underneath.

**2. Scale the cheapest lever first.** Arrays grew. SRAM could not. Bandwidth grew when it had to.

**3. Balance drifts, then gets corrected.** Each generation spends the margin the last one left. Watch your ratios.

**4. Spend transistors on software's problems.** Google widened VLIW slots, generalized ALUs and improved DMAs as logic got cheap. Generality pays once logic stops being scarce. [J]

**5. Move reliability into silicon as scale grows.** Spare rows, self-test and replay were not needed at 256 chips. They are at 9,216.

**6. Let new constraints change the metric.** Performance per dollar became performance per watt, and then carbon per FLOP. [J]

**7. Margin has value.** Overprovisioned networks saved TPU pods when cables failed. [R] A perfectly balanced chip has nowhere to absorb a surprise.

## The six decisions

Jouppi ranks what proved most durable. [J]

1. Systolic arrays for matrix multiplies.
2. Range-oriented narrow floating point.
3. HBM for main memory.
4. Custom high-speed links.
5. Software control of the memory hierarchy, with DMAs and scratchpads.
6. Vector units for non-matrix computation.

Two other innovations remain unique to TPUs: optical switches and SparseCores. There is still no agreement across the industry on processors per accelerator, from 2 up to 1,472. [J] Cray's old question captures Google's position: would you rather plow a field with two strong oxen or 1,024 chickens? [J]
