# 05. First architecture sketch

*Status: thinking out loud, nothing here is decided yet. This turns into Spec v1 once the model (D-002), memory (D-004) and metric (D-008) are decided. The workload (D-007) and programmability (D-009) are decided. Note: under D-007 the matrix unit is sized to a prefill floor, roughly 64×64, not 128×128.*

## The blocks

Here's the chip as a list of blocks, following the path of one token.

- **Host interface.** A connection to a host computer (PCIe or something simpler in simulation). The host sends prompt tokens in and receives generated tokens back. That's a few bytes per token, so bandwidth here doesn't matter.
- **Control processor.** A small scalar core that fetches instructions from its own instruction memory and runs loops: which layer, which tile. Instructions are VLIW bundles: one wide instruction holding several independent operations (see below).
- **DMA engines.** These move data between DRAM and on-chip SRAM. The control processor gives a DMA engine a *descriptor*, a small record of about 32 bytes: "copy N bytes from DRAM address X to SRAM bank Y, with these strides, then set flag F." The engine then issues the hundreds of 64-byte requests itself, without bothering the control processor again.
- **Memory controller and PHY.** These turn DMA requests into DRAM commands (activate, read, precharge, refresh) and drive the pins.
- **SRAM scratchpad.** On-chip memory split into banks, managed by software. Unlike a cache, the program decides exactly what lives where. It holds:
  - incoming weight tiles, double-buffered so one tile is being used while the next arrives
  - the current activation vector (4,096 values, 4 KB at int8)
  - a slice of the KV cache or DeltaNet state being worked on
  - partial results
- **Matrix unit.** Multiplies activations by weights. For decode (batch 1 to 8) it's a matrix-vector multiply. For prefill it's a full matrix multiply over many tokens at once. Its size is set by the prefill target (`02_workload.md`).
- **Vector unit.** Everything that isn't a big matrix multiply: requantization, RMSNorm, SiLU and sigmoid gates, rotary embeddings, softmax, the DeltaNet convolution and state update, residual adds, and choosing the next token over 248K logits.
- **Sync flags (semaphores).** Small hardware counters. A DMA engine sets one when its copy finishes, and the control processor can wait on it. This is how "the data has arrived" reaches the program.

## What passes between the blocks

- **Host → control:** a token ID (4 bytes) and commands like "start prefill" and "generate".
- **Control → DMA:** descriptors, about 32 bytes each.
- **DRAM → SRAM, via DMA:** weight tiles. This is the main stream, 4.46 GB per token. Also KV cache rows and DeltaNet state.
- **DMA → control:** completion flags.
- **SRAM → matrix unit:** the activation vector plus the weight tile.
- **Matrix unit → vector unit:** 32-bit partial sums.
- **Vector unit → SRAM:** the next layer's activation vector in int8, plus updated state and new KV entries.
- **SRAM → DRAM, via DMA:** the updated DeltaNet state and new KV entries, written back.

## How the VLIW control keeps everything busy

### What "busy" means here

At batch 1, the matrix unit needs only about 500 MACs per cycle to keep up with the memory. So the goal isn't keeping the multipliers busy. It's keeping the **memory bus** busy every single cycle. Any cycle where DRAM is idle is lost forever, because the next token's speed is set entirely by how quickly the weights stream in.

### Software pipelining

The program overlaps three stages, each working on a different tile:

- **DMA** fetches tile i+2 (it runs ahead)
- **Matrix unit** multiplies tile i
- **Vector unit** finishes the result from tile i−1

One VLIW bundle in the steady-state loop looks like this:

```
scalar: next loop index | dma: start tile i+2 | matrix: tile i | vector: finish i-1 | wait: flag for tile i+1
```

- All five slots run in the same cycle.
- **The wait slot is the only place the program can stall:** it stalls only if tile i+1 hasn't arrived yet.
- DRAM latency varies (refresh, row misses), so we don't schedule DRAM by exact cycle. We schedule it by **queue and flag**: start requests early enough to cover the worst case, then wait at the flag. The compute side is fully deterministic, so it can be scheduled exactly.

### Sizing the overlap

- **Little's law sets how far ahead DMA must run:** about 27 KB in flight at 273 GB/s and 100 ns. A tile of 16 to 64 KB, double- or triple-buffered, covers it.
- **SRAM needed for buffering is therefore small:** a few hundred KB for weight tiles. The big SRAM questions are about KV and DeltaNet state (see `04_memory.md`).

## Decisions this sketch raises

- **How dynamic is control?** Fully static VLIW (the compiler decides everything, like Groq) vs a static core plus queued DMA (like the TPU). Proposal: static compute, queued DMA, because DRAM timing isn't predictable.
- **How many DMA engines,** and how many requests each can keep in flight.
- **SRAM size and banking:** how many banks, how wide, and what lives where.
- **Matrix unit shape:** one square systolic array used for both prefill and decode, or a square array plus a separate wide, short unit tuned for matrix-vector multiply.
- **Precision:**
  - int4 weights with per-group scales
  - int8 activations
  - 32-bit accumulators
  - bf16 or fp32 in the vector unit and DeltaNet state
- **Where the DeltaNet state and KV cache live:** DRAM with streaming (the current default) or partly in SRAM.
- **How to sample from 248K logits:** a top-k or argmax unit, streaming the logits as the output head computes them so they never all need storing at once.
- **The instruction format:** how many slots, how wide. This is also the "language" the RL model will later learn to write, so it must be simple enough to learn and simulate quickly.

## How this connects to the RL goal

The control program is the thing the RL-trained model will eventually write. That means:
- **We need a fast, exact simulator** of the control program's behavior: correct results plus a cycle count. Then RL can try thousands of programs and get rewarded for correct output in fewer cycles.
- **A clean, documented instruction set** matters twice: once for us writing the first programs by hand, and again for the model learning to write them.
