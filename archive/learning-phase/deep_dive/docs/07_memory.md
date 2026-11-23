# Memory

If the systolic array is the engine, memory is the fuel system. This chapter covers how designers size each level, and what each level is for.

::viz memory

## The pyramid

Going up the hierarchy gives exponentially better performance at much higher cost. [IA]

| Level | Typical size | Bandwidth | What it holds |
| --- | --- | --- | --- |
| PE registers | Bytes per PE | Highest | Operands in flight |
| Accumulators | KB to MB | Very high | Partial sums |
| Register file | KB to MB | Very high | Vectors |
| Scratchpad or L1 | 100s of KB to 100s of MB | High | Tiles |
| HBM | 10s to 100s of GB | TB/s | Weights, KV cache |
| Host or SSD | TB | GB/s | Cold KV cache, checkpoints |

## Cache or scratchpad

If hardware decides what stays in on-chip memory, it is a cache. If software decides, it is a scratchpad. Public discussion gets this wrong constantly, and the difference changes memory maps, hardware complexity and the programming model. [IA]

| | Cache | Scratchpad |
| --- | --- | --- |
| Who places data | Hardware, by guessing | Software, explicitly |
| Extra hardware | Tags, replacement, coherence | DMA engines |
| Timing | Varies with hits and misses | Deterministic |
| Good when | Access is irregular | Access is known in advance |

Roune adds a useful twist. Automatic caches are convenient for normal code. In high-performance code, you already think about what is in cache, so manual control is actually more convenient. [R] His recommendation is both: a cache that is automatic by default, with portions convertible to manual. Nvidia GPUs have both already, through L1 and shared memory.

Jouppi lists software control of the memory hierarchy through DMAs and scratchpads as one of the six key decisions. [J] TPU cores access memory only through DMA. [R]

## Sizing SRAM from first principles

### Step 1: the reuse you need

Take a matmul where an output tile of size Tm × Tn accumulates on chip while K streams through. Bytes moved for each step of K are (Tm + Tn) × b. Operations are 2 × Tm × Tn. So:

`intensity = 2·Tm·Tn ÷ ((Tm + Tn)·b)`

For a square tile of side T, intensity is T ÷ b. To reach the machine's ridge point, you need:

`T ≥ ridge × b`

For Ironwood in BF16, the ridge is 316 FLOP/byte, so T ≥ 632. [D] That tile's FP32 accumulators take only 1.6 MB. Square tiles of that size are small next to 128 MiB, so raw matmul reuse is not what forces large SRAM.

### Step 2: double buffering

To overlap compute with data movement, you hold the current tile and load the next one. [E]

`SRAM ≥ 2 × (Tm·Tk + Tk·Tn) × b + Tm·Tn × acc_bytes`

Multiply by the number of arrays sharing the SRAM. The software pipeline's stages decide how many buffers you need. Each live operation costs real storage, which is why fewer pipeline stages are preferred. [R]

### Step 3: what actually consumes capacity

Most of a real scratchpad goes to things beyond one tile:

- Weights resident across many tokens.
- Fused operations holding intermediate activations, which saves round trips to HBM.
- KV tiles during attention.
- Staging for collectives, so that network traffic never touches HBM. [R]

Your factbook shows Google's split: TPU 8t has 128 MB of vector SRAM, while the serving-focused 8i has 384 MB. [F] Serving spends silicon on SRAM because decode has less reuse to exploit.

## Bank organization

A 128 MiB scratchpad is many thousands of small macros. Macro size trades density against speed. Larger macros have less peripheral overhead per bit, but longer bitlines and more energy per access. [E] Each access only needs a small piece, so many small banks are typical, and they also supply parallelism.

Bandwidth needed at the array edge for an R×C weight-stationary array per cycle: R activation elements in, C results out, plus weights while loading. Provision ports for that, and put the banks physically at the edge.

## Why HBM

Standard DRAM was the bottleneck on TPU v1, so TPU v2 moved to HBM. [J] Jouppi ranks HBM third among the six key decisions, and most accelerators have converged on it.

HBM stacks DRAM dies on a base die and connects it through a silicon interposer with a very wide, relatively slow interface. Width gives you bandwidth at reasonable energy per bit. The costs are interposer area, packaging difficulty and price.

An important distinction from Roune: HBM is expensive per GB of storage, but not per GB/s of bandwidth. He estimates around 3× regular RAM per GB. [R] So ask whether you need the capacity or the bandwidth. They come bundled, and those are different questions.

## How much capacity

Two things fill HBM: weights and KV cache.

`KV bytes = 2 · batch · layers · seq · groups · width · bytes · idle · pipeline`

Roune's example: batch 64, 32 layers, 1M context, 8 groups, width 128, 1 byte per element gives 4,194 GB. Split 32 ways, that is 131 GB per chip. [R, D] This is why chips carry about 200 GB.

Weights are less of a problem than people assume. With 1,024 chips, 1,000 GB of weights is 1 GB per chip. [R]

The two respond differently to parallelism.

| Parallelism | Weights per chip | KV cache per chip |
| --- | --- | --- |
| Duplication | Same | Same |
| Pipelining across layers | ÷S | Same, for decode |
| Within layers | ÷P | ÷P |

## How much bandwidth

Decode reads all active weights and each user's KV cache every step. So:

`step time ≥ (weight bytes + batch × KV bytes per user) ÷ bandwidth`

Note that batching amortizes weights but not KV. Roune also points out that splitting across C chips does not change per-chip bandwidth need, because each chip reads 1/C of the data, and you want C times the tokens. [R]

## Reducing the need

Roune argues this is where the biggest savings hide. [R]

- **Sparse or indexed attention** replaces dense reads of ancient history with retrieval. Going from 1M to 1k tokens of dense attention takes KV from 4,194 GB to 4 GB. It could even allow KV on SSD.
- **Grouped-query attention** cuts KV by the group factor.
- **KV quantization** to 8 or 4 bits. 8 is standard for serious work.
- **Lossless compression** on top of lossy quantization. JPEG does both. One reported result is about 20% size reduction. This needs Huffman decoding at memory speed in the DMA path, and Nvidia's existing decompressor runs at only about one tenth of memory speed. Lossy mapping can be folded into the same lookup tables, so a per-layer custom format can be nearly free. Roune is careful to say he has not quantified the area cost.

## Stacking more SRAM

Hybrid bonding can add SRAM directly above logic. AMD's V-Cache adds 64 MB per chiplet with only a small latency cost, placed above existing SRAM, while blank silicon covers the hot cores. [IA] For a statically scheduled design, added latency hardly matters because the compiler accounts for it in advance, so Irrational Analysis sees this as a natural next step for Groq-style chips. [IA]

## Latency, bandwidth and Little's law

To keep a memory pipe full, you must have enough requests in flight.

`bytes in flight = bandwidth × latency`

With 7.3 TB/s and a 500 ns round trip, you need 3.65 MB outstanding. At 4 KB per request, that is about 890 requests in flight. [D] Every queue on that path has to hold them: DMA descriptors, memory controller queues, response buffers. This is the calculation behind Roune's warning that one undersized queue can destroy performance. [R]

## SRAM-only machines

Groq stores all weights in on-chip SRAM: 220 MB in an early part at a slide-quoted 55 TB/s, and about 500 MB in LPU3. [IA, R] Your factbook's 128 GB across 256 LPUs per rack agrees with that figure. [F, D] A 100B FP8 model needs at least 200 chips just for weights. [R]

Irrational Analysis insists this is not fundamentally about SRAM. The SRAM enables a deterministic dataflow. [IA] Roune's view is more economic: low batch needs enormous bandwidth, which forces SRAM, which is expensive, and that makes sense only for premium low-latency tokens. [R] Both are true at different levels.

## Tuning reference

| Knob | Effect | Cost |
| --- | --- | --- |
| SRAM capacity | Bigger tiles, more fusion, less HBM traffic | Area that barely shrinks |
| SRAM bandwidth and banks | Feed arrays, fewer conflicts | Macro overhead, wiring |
| Buffering depth | Hides latency | SRAM capacity |
| HBM stacks | Bandwidth and capacity | Beachfront, cost, power |
| Compression in DMA | Effective bandwidth and capacity | Engine area, variable sizes |
| Queue depth | Sustains bandwidth | Control area |
| Cache mode | Easier programming | Tags, less determinism |
