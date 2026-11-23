# TPU / RTX source research

Read 2026-11-03. Internal research notes, not content to copy wholesale into the guide. The published guide should give short attributed case studies and link to the originals. Web source summary allowance was 200 words per PDF; keep source-derived case-study prose within that allowance, using original first-principles explanations elsewhere.

## Sources and coverage

- [Google's Training Supercomputers from TPU v2 to Ironwood: Architectural Stability, Scale, Resilience, Power Efficiency, and Sustainability Across Five Generations](https://arxiv.org/pdf/2606.15870), Norman P. Jouppi, Sridhar Lakshmanamurthy, Cliff Young, David Patterson, 13 PDF pages. Full substantive paper read: architecture, evolution, reliability, power, carbon and conclusion; references inspected as references. Figures 2 and Table 1 visually checked. This is a retrospective on five generations of Google training systems, not an independent benchmarking study or a new single accelerator algorithm.
- [NVIDIA RTX Blackwell GPU Architecture v1.1](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), 57 PDF pages. Architecture, SM, GDDR7, Tensor/RT cores, scheduling, power, and RTX5090 appendix read in detail. DLSS, neural graphics, video/display, and other product appendices reviewed for scope and architecture lessons. Figures 5 and 21, Table 3 visually checked. This is NVIDIA's vendor whitepaper for RTX GB20x. Do not conflate it with datacenter B200/GB200 or their HBM/NVLink specifications.

Local private research copies and extracted text are in `research/pdf-source/`. Do not ship those copies or screenshots in the website. Create original schematic diagrams and link to the source PDF figures.

## Short TPU case study suitable for the guide (~170 words)

Google's training TPUs pair scalar control with vector units, matrix units, local SRAM, asynchronous DMA, and a dedicated inter-chip network. A fast matrix engine needs a surrounding system that supplies operands, handles non-matrix operations, and coordinates transfers. The paper describes compiler-managed memory and DMA-completion flags: software schedules movement explicitly and waits when required data has not arrived.

Ironwood scales matrix, vector, memory, and network resources while retaining much of that programming structure. Its SparseCores illustrate why a second engine can help work that does not fit dense matrix execution. Optical circuit switches help assemble available machines and isolate failures; self-test and sampled redundant execution address silent computation errors.

The lesson is to optimize completed useful work across the entire machine. The paper's roughly 100× node comparison changes from BF16 to FP8, while its performance-per-watt chart divides peak throughput by TDP. Neither is an equal-precision application speedup or measured workload energy result. Read the units, precision, system boundary, and utilization assumptions before comparing designs.

## TPU precise anchors for a reference drawer

| Reader question | Page / location | Reliable reading |
|---|---|---|
| What surrounds a matrix array? | PDF p.4, Figure 2 | Scalar/VLIW dispatch, vector register file, VMEM, MXU, DMA, HBM, interconnect. Figure is TPU v2 and omits SparseCore, not a detailed Ironwood floorplan. |
| What does software control? | pp.4–5 | DMA between HBM and VMEM, synchronization flags, strided transfers, compiler fusion. |
| How much arithmetic per cycle? | p.4 | TPU v2's 128×128 array has 16,384 MACs/cycle =32,768 operations/cycle under multiply-plus-add counting. |
| What changed across generations? | p.2 Table 1, pp.6–7 | Matrix/vector capacities, SRAM, memory/network bandwidth, physical packaging. Do not sum BF16 and FP8 modes as independent simultaneously available arrays. |
| Why add another engine? | p.5 SparseCore | Fine-grained memory access, embeddings, collectives and small/sparse operations. Dense compute and other engines can execute in parallel. |
| Why care about topology and failure? | pp.7–9, Figure 4 | Fault isolation, scheduling available slices, incremental deployment, goodput, FBIST, sampled VPU replay. |
| What counts as efficiency? | pp.9–11 | Peak/TDP distinct from measured goodput/power; embodied plus operational carbon. |

TPU source cautions: Table 1 labels a pod HBM row “PetaBytes” with values that cannot be petabytes (e.g. 4 for TPU v2); text gives roughly 4TB and 1.77PB endpoints. The body says four HBM2 stacks for TPU v2 while Table 1 says two. Avoid reproducing either inconsistency. The p.11 worked carbon example multiplies to about 83×10^6 grams, which is about 83 metric tonnes, not 83 million tonnes as printed. These are source errors, not instructional values. The “~30× perf/W” chart uses mixed precision and TDP, and is a relative peak-system claim. Neither vendor paper proves the user's project's performance, area, power, or manufacturability.

## Short RTX case study suitable for the guide (~180 words)

RTX Blackwell is a heterogeneous graphics/AI processor: programmable lanes work alongside Tensor, ray-tracing, texture, video and display hardware. Its unified FP32/INT32 lanes can perform either operation type in a cycle; their separate peak figures cannot simply be added. A dedicated RISC-V context scheduler illustrates another design choice: moving recurring coordination work closer to the engines.

Memory specifications expose an easy consistency check. RTX5090's 512-bit interface and 28Gb/s per-pin data rate imply 1,792GB/s peak bandwidth. Useful bandwidth depends on access patterns and overhead. The full GB202 design has 192 SMs, while RTX5090 enables 170; full-die diagrams and shipping-product specifications describe different configurations.

The power chapter separates clock gating, power gating, independent voltage rails, and rapid frequency changes. Their value depends on idle duration and transition cost, not just active speed. The appendix distinguishes dense and sparse Tensor throughput and accumulator precision. TGP is a board power specification, not measured compute-die consumption. This RTX paper does not disclose enough detail to reconstruct a Tensor Core's physical systolic geometry, and its specifications should not be substituted for datacenter Blackwell specifications.

## RTX precise anchors for a reference drawer

| Reader question | Printed page / location | Reliable reading |
|---|---|---|
| What is on the chip? | pp.8–11, Figs 3–5 | Full GB202, hierarchy GPC/TPC/SM, four Tensor Cores per SM, register file and L1/shared memory. |
| Shared execution or separate pipes? | p.12 | FP32 and INT32 use unified lanes; many but not every INT operation gains throughput. |
| What sets off-chip bandwidth? | pp.13–15 | Interface width and per-pin data rate; GDDR7 signaling and controllers. |
| How does precision change throughput? | pp.15–16,47–48 | Supported types, accumulator mode, dense vs sparse distinction. |
| When does specialization help? | pp.17–24 | RT engines accelerate frequently repeated structures; SER improves coherent grouping for divergent work. |
| Can scheduling itself be hardware? | pp.25–26, Fig.17 | Programmable RISC-V AMP schedules contexts; not the same as replacing every warp scheduler or eliminating all host software. |
| What controls power? | pp.31–34, Figs.20–23 | Clock/power/rail gating, transition latency and workload-aware clock changes. “1000× faster” is switching response, not a 1000× clock rate or workload speedup. |
| What is hardware/software co-design? | pp.35–45 | Neural shading and display-side pacing require algorithms, APIs and accelerator capabilities together. |
| What does a SKU actually enable? | pp.46–48, Table3 | RTX5090 170 SMs/96MB L2 vs full GB202 192SMs/128MB L2. |

Selected specification facts for a compact table, if useful: RTX5090 2,407MHz boost clock; 32GB GDDR7; 1,792GB/s memory peak; 575W TGP; GB202 die 750mm²; dense FP4/FP32 accumulation 1,676TFLOPS vs sparse effective 3,352TFLOPS (pp.46–48). Rates use boost clock. These are a vendor snapshot, not current pricing or workload benchmarks. Keep dense/sparse and accumulator mode attached to every throughput number.

## Original teaching interactions suggested by the sources

1. **One scarce lane, two jobs.** A timeline alternates integer address work and floating-point work. Let the learner change the mix. Explain resource sharing and why nominal peaks are not simultaneously attainable.
2. **Trace an operand.** HBM → DMA → banked SRAM → local register → MAC → accumulator → vector post-processing → store. Clicking each hop shows bandwidth, capacity, latency and control responsibilities. Use an original generic accelerator diagram, not a re-traced proprietary floorplan.
3. **Read a spec honestly.** Toggle full die / shipping SKU, dense / supported structured sparsity, input / accumulator precision, peak / sustained. Explain each denominator before presenting numbers.
4. **Sleep break-even.** Compare active, clock-gated and power-gated periods. A low-power mode pays only if saved idle energy exceeds entry/exit energy and wake-up latency remains acceptable. This is a generic engineering model, not an RTX power prediction.
5. **Goodput.** Compute useful work / wall time with configurable checkpoint, recovery and idle intervals. A bigger peak engine can lose if the surrounding system wastes more time.

No RTL, synthesis, placement, clock-tree or measured-power claims can be inferred from these sources for the local project. Use them as examples of decisions and interfaces, not as validated parameters for the learner's design.
