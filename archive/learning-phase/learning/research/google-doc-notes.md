# Source notes — Designing AI Chip Software and Hardware

Source: [Bjarke Hammersholt Roune, Designing AI Chip Software and Hardware](https://docs.google.com/document/d/1dZ3vF8GE8_gx6tl52sOaUVEPq0ybmai1xvu3uk89_is).

Reviewed: 2026-11-03. This is an opinionated architecture proposal with valuable practitioner explanations, not a measured implementation or a consensus textbook. The local text export has 82,691 words and 70 section headings after the title. All substantive textual sections were examined across this review and the parallel networking/software review. Embedded figures and original hyperlink destinations were not preserved by the text export and were not visually audited. No benchmarks were reproduced.

## Teach these decisions

**Start from a product and workload envelope.** Specify the model family, supported operators and shapes, batch/concurrency, context length, precision/quality requirements, latency target, deployment scale, and software you control. An accelerator for low-latency single-request decode is a different optimization problem from high-throughput prefill. Maximize useful work within those constraints; peak arithmetic rate alone is an incomplete objective.

**Array size is a reuse–flexibility tradeoff.** A larger regular array can amortize operand delivery, control, and peripheral circuits over more arithmetic. Small or awkward dimensions can leave lanes idle or introduce padding. More small arrays offer independent scheduling and shape flexibility but duplicate some delivery/control resources. Compare fixed square, fixed rectangular, heterogeneous, and partitionable designs under matched MAC count, memory capacity, bandwidth, and frequency assumptions. Runtime partitioning has routing, buffering, arbitration, and control costs.

**Name the mapping before using an array formula.** The document often discusses a weight-stationary physical K×M array fed with N activation rows. Those meanings differ from an output-stationary R×C array that advances a reduction over time. Array dimensions, instruction tile dimensions, register dimensions, and workload M/N/K are not interchangeable. Show what stays local, what moves, and how partial sums are accumulated. Then calculate fill/drain, padding, useful MACs, and bytes at each interface.

**Provision the complete supply chain.** A MAC array is useful only if its operands arrive and results leave on time. The relevant blocks include SRAM banks, register files, DMA/address generators, input/output FIFOs, command queues, dependency tracking, vector units, reductions, nonlinear functions, permutation/layout hardware, and the fabric connecting them. Each can become the bottleneck. Do not dimension scalar/vector support from an unoptimized loop if simple unrolling or fusion changes the demand.

**FIFO depth buys decoupling, not sustained bandwidth.** Queues absorb temporary timing differences and let a controller issue work in bursts. They do not fix a consumer that is permanently slower than its producer. A stall propagates backward when queues fill. Deep queues cost storage, switching energy, and latency; capacity must cover burstiness and outstanding latency while respecting downstream service rate.

**DMA moves work out of the control loop.** A descriptor can describe many transfers instead of requiring instruction-by-instruction feeding. Multidimensional strides, gather/scatter, outstanding requests, and completion events increase flexibility. Direct memory-to-array paths can save staging traffic but need layout, flow control, and error handling. Bypassing the vector path is useful only when the data does not need transformations that path supplies.

**Separate storage, multiplication, and accumulation precision.** Packed low-bit weights can reduce capacity and bandwidth without necessarily increasing MAC throughput. Narrow multipliers can save hardware, but accumulation, rounding, scaling, saturation, unpacking, and conversion remain. Support for many formats buys model compatibility and costs verification/design effort. Choose a quality threshold and test representative models, including outliers; no bit width is universally sufficient.

**Treat sparsity as a contract.** Structured patterns simplify selection and metadata compared with arbitrary zero locations. Benefits require an eligible model, supported layout, sparse instructions, and enough time in eligible kernels. Retraining and quality evaluation are part of the cost. Report executed nonzero operations separately from dense-equivalent throughput. A 2× eligible arithmetic peak is not a 2× application speedup.

**Lossless compression is another bandwidth–compute trade.** It can compose with quantization because it exploits symbol statistics rather than deliberately removing information. It adds metadata, decoding throughput, variable block sizes, alignment, and buffering. Model incompressible/worst-case data and random access. A compression engine that cannot deliver decompressed operands at the required rate becomes the new bottleneck. The document explicitly has no empirical area/power tradeoff for its proposed compression path.

**Prefill and decode stress different resources.** Prefill processes many known query positions and often reuses operands extensively. Decode normally has a small query dimension per sequence and must revisit weights and KV state. Batching reuses weights across sequences; it does not make unrelated sequences share their KV data. GQA shares KV among query heads within its architectural contract. Speculation adds verified tokens per target invocation only according to acceptance and draft overhead; rollback and provisional state matter.

**Budget memory capacity separately from traffic.** For equal key/value dimensions, basic KV bytes = 2 × live sequences × layers × context × KV heads × head dimension × bytes/element, plus scales, metadata, padding, and fragmentation. Use actual varying sequence lengths or their sum. Sharding and replication change per-chip capacity. Offloading trades lower storage cost for bandwidth/latency and scheduling requirements. Sparse attention changes the algorithm and must pass quality validation; it is not a free replacement for dense attention.

**Overlap only independent work.** Double buffering allows transfer of a future tile while computing a current tile if banks, ports, buffers, and dependency rules permit. Fusion saves intermediate traffic but can increase live storage, resource pressure, and scheduling difficulty. Separate prefill/decode devices gain specialization but must move KV and absorb load imbalance. Mixing them can improve aggregate resource use but does not guarantee latency targets or perfect balance.

**Parallelism moves the bottleneck.** Replication scales independent requests but repeats weights. Pipeline parallelism partitions layers and creates bubbles or more in-flight requests. Tensor parallelism shortens individual layers at the cost of collectives. Expert parallelism introduces dispatch/combine and load imbalance. Evaluate total system cost and latency, including hosts, links, memories, cooling, and utilization; smaller per-chip memory is not a system-level saving if more chips are required.

**Choose a topology for its traffic.** Ring/torus/mesh, crossbar, and switch fabrics trade hop count, bisection bandwidth, wiring/ports, cost, locality, and failure behavior. Collective-friendly regular traffic and MoE all-to-all are different demands. Specify whether bandwidth means per-link, aggregate injected, useful payload, or delivered endpoint bandwidth before comparing numbers.

**Software is part of the architecture.** The compiler/runtime must support the layouts, tiling, buffers, dependencies, scheduling, and numeric contract that make the hardware useful. Hardware cost savings that assume unavailable software are not realized savings. Parameterize vector/array sizes for manageable small tests, but also test production sizes, boundary shapes, resource limits, and timing-related behavior. Small-size correctness is not proof that every parameterization works.

**Power density is not energy per operation.** Efficient dense compute can still make a hotspot. Physical placement, clock gating, activity limits, voltage/frequency choices, power delivery, cooling, and throttling are coupled. Distinguish an arithmetic energy estimate from total chip power and from board power. The TPU thermal story is the author's account; its particular 25% benefit is not a transferable design constant.

## Corrections and uncertainty to keep visible

| Source statement or framing | How the guide should handle it |
| --- | --- |
| All major matrix engines are essentially identical systolic arrays; marketing core names map directly to them. | A matrix instruction or tensor engine does not establish its physical dataflow, array geometry, or total subsystem composition. Use disclosed implementation evidence only. |
| Specific NVIDIA generations have particular physical array dimensions. | Do not infer physical PE geometry from instruction/tile dimensions or repeat these dimensions without primary evidence. |
| INT8/7 is always enough, FP16 is a drop-in inference substitute, and supporting more precision is unnecessary. | These are workload-dependent claims; range, accumulation, scaling, calibration, quality, and supported operators must be checked. |
| Always use structured sparsity; a larger sparse model dominates. | Not established universally. Quality, retraining cost, memory/KV growth, supported kernels, and end-to-end latency can overturn that choice. |
| Heterogeneous attention/FFN arrays improve every context, even at a poorly chosen ratio. | Not a theorem. Serial dependencies, resource imbalance, queueing, and software complexity can lose. Compare against a tuned homogeneous baseline. |
| Small parameterized tests establish correctness at large dimensions. | Useful for debugging, but production-size indexing, overflow, capacity, timing, and control need additional validation. |
| SSD-backed indexed attention can replace almost all HBM. | A proposed algorithm/system co-design direction, conditional on quality, retrieval overhead, miss behavior, latency, and bandwidth. The quoted 4 GB capacity is not a proven universal requirement. |
| Pipeline parallelism inherently multiplies KV by a fixed factor and doubles token latency. | These results use specific assumptions about per-stage batch, number of in-flight batches, and equal-duration network stages. With global concurrency fixed, layers can shard a request's KV across stages. |
| Router-based networks are limited to 50% efficiency and tori reach 100%. | A chosen aggregate link/port-budget accounting model. Not a universal endpoint-throughput ceiling; load, paths, contention, packet size, and bisection matter. |
| Different kernels cannot execute on one GPU SM concurrently. | Overbroad. Concurrent kernel execution is supported on capable CUDA devices when scheduling and resources allow it. Do not assume overlap, but do not rule it out categorically. |
| An N-job, S-stage pipeline has N−2(S−1) fully occupied steps. | For the illustrated equal-duration pipeline: total steps N+S−1; fully occupied steps max(0,N−S+1). For N=4 and S=3, two steps are fully occupied. |
| Hardware supporting a transformer family implies weights are burned into silicon for one model. | Distinguish architecture-specialized, operator-specialized, configurable, and weight-fixed hardware. The document's Etched/Taalas grouping and acquisition assertion should not be repeated without primary verification. |
| AI CPUs will inevitably replace GPUs; particular companies are bloated or mismanaged. | These are opinions and anecdotes. Omit from the instructional core; compare measurable designs and product constraints instead. |
| Any extra memory bandwidth or convenience feature is waste. | Flexibility, service guarantees, workload drift, programmability, utilization, reliability, and development cost are legitimate design objectives. |

## Primary cross-checks

- [NVIDIA structured sparsity and search-engine applications](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) describes the 2:4 pattern, index metadata, quality recovery through training, and application speedups below the theoretical eligible-compute multiplier.
- [Google TPU system architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm) explicitly describes MXU systolic organization. This does not establish other vendors' undisclosed physical arrays.
- [CUDA Best Practices Guide, concurrent kernel execution](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-kernel-execution) documents multiple simultaneous kernels on capable devices. Actual overlap remains resource- and scheduling-dependent.
- [Etched's own current description](https://www.etched.com/) describes an HBM/SRAM hybrid inference system and splittable math arrays, without substantiating the source document's blanket one-fixed-model characterization. Vendor performance claims there were not independently validated.

## Suggested learning interactions

1. A clock-step array view that labels stationary data, moving operands, accumulation, bubbles, and completed outputs.
2. A matched-MAC geometry comparison with shape padding, fill/drain, bandwidth, and partition overhead visible.
3. A producer/FIFO/consumer timeline showing finite burst absorption and eventual backpressure.
4. A load–compute–store pipeline demonstrating warm-up, overlap, buffer conflicts, and drain.
5. A memory worksheet with separate weight, KV, scratchpad, and accumulator budgets.
6. A resource roofline showing how changing compute or bandwidth can leave throughput unchanged.
7. A conditional precision/sparsity experiment that keeps quality and metadata beside the claimed speedup.
8. A design-review worksheet asking what workload, software, physical, and quality assumptions each choice depends on.
