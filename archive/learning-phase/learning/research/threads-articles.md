# Threads and dataflow articles — research notes

Checked 2026-11-03. Original synthesis; no article images or full text are redistributed.

## Coverage and confidence

- Both requested Substack articles were read through their substantive text, including the ending. Their images were not independently inspected, and embedded videos were not watched. The 2026 article contains numerous explicit placeholders, so reading it does not supply the missing sections.
- Direct X profile retrieval returned HTTP 403. An indexed public TwStalker profile showed a partial timeline, with an unavailable effective pagination path. Three connected architecture posts, one detailed follow-up, a patent-commentary post, and a goal post were read through public mirrors. **This is not an exhaustive account archive or all-thread review.**
- Public mirrors can duplicate the quoted post into the quote's text. Attribute only clearly identified author paragraphs. Canonical X links below identify the original posts; access was through the listed mirrors.
- Social posts and Substack opinions are prompts for investigation, not measured hardware specifications. Primary Groq documentation was checked separately.

## Requested article: the dataflow draft

[It's the dataflow, stupid](https://irrationalanalysis.substack.com/p/unfinished-draft-its-the-dataflow), February 27, 2026. Unfinished architectural commentary; about 668 lines precede reader discussion in the retrieved page.

The useful organizing questions are where data lives, who moves it, what computes on it, and how chips communicate. Cache and scratchpad describe management policies; SRAM describes a storage technology. Memory size alone does not specify accessibility, sharing, ports, locality, or delivered bandwidth. Compiler-controlled execution makes communication timing part of the programming model.

The article contrasts conventional CPUs/GPUs/TPUs with Groq's deterministic approach and speculates about clock forwarding, stacked SRAM, and cooling. These proposals are not a validated future product. The sections on several startups and the final product mock-up are unfinished. Avoid adopting its categorical judgments about companies, compiler competence, topology superiority, or thermal problems being solved. Likewise, physically distributed last-level cache slices are not necessarily private caches. Use primary architecture documentation before assigning those properties to a named chip.

Teaching use: a four-layer chip map, cache-versus-scratchpad comparison, and an experiment showing that more capacity does not fix insufficient access bandwidth.

## Requested article: VLIW writeup

[[V]ery [L]ong [I]ncoherent [W]riteup](https://irrationalanalysis.substack.com/p/very-long-incoherent-writeup), February 24, 2024. Historical opinion, not a current performance assessment; substantive text read through its conclusion.

The reusable lesson is that hardware simplicity can transfer difficulty to software. In VLIW, the compiler groups operations into legal issue slots and must respect data dependencies and resource availability. Peak issue width says little about the number of useful operations scheduled. Evaluate compiler quality, supported operators, compilation time, shape coverage, and engineering effort alongside power, performance, and area.

Do not teach that VLIW is inherently unsuccessful, that only one company can compile it, or that an employee's departure proves technical failure. The article's occupancy examples are expressly invented and its inference about hand-written assembly is not a measurement. VLIW and SIMD are different axes: multiple operations in an instruction bundle versus one operation applied across multiple data lanes. A machine can combine both.

Teaching use: a small instruction scheduling exercise where loads, multiplies, and additions share constrained slots; compare an ideal schedule with dependency stalls. Explain software maturity as a design requirement, not a footnote.

## Requested X author: accessible architecture material

Canonical links:

1. [Interconnect-first argument](https://x.com/0xBADB01E/status/2073970411138949533)
2. [Sharding and decode](https://x.com/0xBADB01E/status/2073990357398982797)
3. [Proposed LPDDR node](https://x.com/0xBADB01E/status/2073998960818360809)
4. [Eight-chip pod and high-batch clarification](https://x.com/0xBADB01E/status/2074048465366061483)

Read via [three-part mirror](https://w.twstalker.com/0xBADB01E/status/2073990357398982797) and [follow-up mirror](https://ngntipkolamrenang.twstalker.com/0xBADB01E/status/2074048465366061483).

The author argues for sizing compute, memory bandwidth, capacity, and communication together, with special attention to small messages in interactive decode. Pipeline parallelism and tensor parallelism have different critical paths. The proposed LPDDR machine is a hypothesis; its area, power, cost, clock/link latency, and rack throughput are not measured implementation results. The follow-up limits direct full-mesh connectivity to eight-chip pods, not the entire rack, and discusses the workload dependence of high-batch KV traffic. A separate indexed clarification acknowledges that high-batch inference can still benefit from HBM and high compute throughput.

Also reviewed: [Intel patent commentary](https://x.com/0xBADB01E/status/2074927205415833718) supplies no concrete design derivation; [ambitious parameter/cost goal](https://x.com/0xBADB01E/status/2074949982646607872) is an aspiration, not a specification. Neither belongs in a chip-spec facts table.

## Primary-source check: Groq

[Groq Hot Chips 34 presentation, 2022](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf).

Verified sections: printed slides 4–6 (static/dynamic boundary and deterministic memory); 14–25 (functional slices, streams, SRAM, scheduling, ISA); 40–44 (clock alignment and networking). This is a focused check, not a full presentation review.

The presentation describes matrix, vector, memory, and rearrangement units; 144 dispatch paths; directly addressed scratchpad banks; cycle-accurate compiler scheduling; and streams with one-cycle hops between units. Slide 23 explicitly describes the system as viewable as fully pipelined 144-wide VLIW and reports less than 3% area for instruction dispatch. Slides 40–41 describe link-latency estimation, clock drift compensation, aligned counters, and deskew instructions. This supports the architectural tradeoff; it does not validate later commentary about compiler failure or hypothetical optical-clock products.

Teaching use: a stream can replace repeated memory round trips, but producer/consumer timing and data layout become part of the contract. Determinism still requires machinery for synchronization and exceptional conditions.

## Independent worked examples for the guide

These are arithmetic teaching models, not benchmark claims about any cited product.

### The latency of an isolated message

For payload S bytes, delivered link rate B bytes/s, fixed startup latency L, use T = L + S/B. With S = 8 KiB and B = 32 GB/s, serialization is 256 ns. At L = 2 µs, total is 2.256 µs; doubling B saves only 128 ns. Reducing L to 0.2 µs produces 0.456 µs. Label payload, bandwidth direction, protocol overhead, and number of communication rounds. A collective operation is not a single message.

### Pipeline parallelism versus tensor parallelism

Assume 64 GB of weights and 4 TB/s per chip, ignore KV and overhead, and read each weight once per generated token. One chip's lower-bound time is 16 ms. Split layers evenly across eight chips: a single request still traverses eight stages, each taking 2 ms, so its weight-read path remains 16 ms. Multiple requests can occupy different stages and improve aggregate throughput. Split each layer across eight chips instead: ideal weight-read time can approach 2 ms, but synchronization and collective communication must be added. Reducing each chip's bandwidth by eight while layer-pipelining makes an individual request's sequential weight-read time eight times longer. Always name the objective: per-user latency or aggregate throughput.

### Weight reuse and KV traffic

For a dense linear layer and batch B, weight-only arithmetic intensity is approximately 2B/q FLOP per byte, where q is stored bytes per weight. At B = 64 and q = 0.5, that is 256 FLOP/B before scale metadata, activations, or partial sums. Full-attention KV storage is 2 × layers × batch × context × KV-heads × head-dimension × bytes/element. KV reads are per sequence, so batching does not amortize them like shared weights. Whether attention or the linear layers dominate must be computed from actual shapes and layouts.

### Port totals are not pairwise rates

Seven independent links at 32 GB/s each provide 224 GB/s aggregate injection rate, but one direct peer gets 32 GB/s unless routing stripes its traffic over other peers, adding hops and contention. Equal aggregate memory bandwidth does not guarantee equal end-to-end token throughput, latency, capacity utilization, or energy.

### Evidence a real specification needs

For area, require a process/library, SRAM macros, PHY sizes, utilization target, routing overhead, and implementation flow. For power, include switching activity, voltage/frequency, leakage, SRAM accesses, clock network, I/O, external DRAM, and cooling boundaries. For communication, define message size, topology, per-direction rate, switch count, startup costs, collective algorithm, and concurrent traffic. Do not turn a speculative social-media number into a default performance guarantee.

## Useful additional source

[NVIDIA H100 product specifications](https://www.nvidia.com/en-us/data-center/h100/) currently list H100 SXM at 80 GB memory and 3.35 TB/s memory bandwidth. These are peak specification values, not sustained application bandwidth. The thread's approximate weight-scan reasoning can be demonstrated as a lower bound, with KV, activation, launch, quantization metadata, and compute costs explicitly excluded.

