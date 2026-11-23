# 04. The memory system

*Status: reopened 2026-11-10. The tokens-per-dollar goal (D-007, D-008) changes the question. See `DECISIONS.md`, D-004.*

## Why memory is the biggest decision

At batch 1, every token reads all 4.46 GB of int4 weights. Nothing else the chip does comes close to that much data. So the tokens per second you can ever reach is simply:

> decode tok/s ≤ memory bandwidth ÷ 4.46 GB

No amount of clever logic gets past that line. Clever logic can only get you close to it.

## The three questions, in order

1. **Capacity: does everything fit?** Weights 4.46 GB (int4) plus KV cache (537 MB per user at 32K context) plus DeltaNet state (25 MB per user in bf16) plus working buffers. For one to eight users that's roughly 5 to 10 GB, so a 16 GB memory leaves comfortable room.
2. **Bandwidth: how fast can it go?** Divide bandwidth by about 4.6 GB per step (weights plus one user's state and KV at 4K context).
3. **Cost, power and packaging: can it actually be built,** and does it fit the workload's story?

## The options

### 3D-stacked SRAM

SRAM stacked directly on the logic die, like AMD's V-Cache, which adds about 64 MB per stacked die.
- **Capacity fails.** The weights need about 4,460 MB, so about 70 stacked dies' worth. Not possible.
- **Where it does fit:** as a large on-chip buffer. For example, 25 MB could hold one user's DeltaNet state.
- **This is the Groq and Cerebras approach:** keep all weights in SRAM by spreading the model across many chips, or using a whole wafer. That's the opposite of a single, cheap chip.

### LPDDR5X

The memory in phones and Apple Silicon Macs. Soldered right next to the chip, low power, cheap per gigabyte.
- **Bandwidth:** about 136.5 GB/s on a 128-bit bus, 273 GB/s on 256-bit (M4 Pro class), 546 GB/s on 512-bit (M4 Max class).
- **Capacity:** easily 16 to 64 GB.
- **Decode ceilings from the script** (batch 1, 4K context): 29.8, 59.6 and 119.2 tok/s for the three bus widths.
- **Cost:** low. **Power:** low. **Packaging:** ordinary.

### GDDR7

The memory on graphics cards. Much faster per pin, but hotter and pricier.
- **Bandwidth:** about 1 TB/s on a 256-bit bus. Script ceiling: about 224 tok/s.
- **Capacity:** 16 to 32 GB is typical.
- **Trade-offs:** more power, needs careful board design for its very fast signals.

### HBM3E

Stacks of DRAM dies sitting next to the chip on a silicon interposer (2.5D packaging such as TSMC CoWoS).
- **Bandwidth:** about 1.2 TB/s per stack. Script ceiling: about 268 tok/s for one stack.
- **Trade-offs:** expensive, supply-limited and needs advanced packaging. It's built for datacenter chips serving large batches.

## First recommendation: LPDDR5X on a 256-bit bus (now reopened)

The first version of this note (2026-11-10) recommended LPDDR5X on a 256-bit bus (273 GB/s, about 60 tok/s). The reasoning: it's faster than reading speed, cheap, low power, and in the same class as an M4 Pro Mac, which makes a clean comparison.

That was before the goal became **tokens per second per dollar** (`02_workload.md`). Under that goal the question changes from "what is fast enough?" to "what gives the most bandwidth per dollar, in about 5.5 GB?".

## Counting memory parts instead of buses

You don't buy "a 256-bit bus". You buy memory chips, and each one adds pins, bandwidth and capacity. From the budget script:

- **One LPDDR5X x64 package (8533 MT/s):** 68.3 GB/s, 8 or 16 GB.
- **One GDDR7 x32 chip (32 Gb/s per pin):** 128 GB/s, 2 or 3 GB.

What you get with a given number of parts, at batch 1 and 4K context:
- **2 LPDDR5X packages:** 136.5 GB/s, 29.8 tok/s, 16 to 32 GB.
- **4 LPDDR5X packages:** 273 GB/s, 59.6 tok/s, 32 to 64 GB.
- **2 GDDR7 chips:** 256 GB/s, 55.9 tok/s, 4 or 6 GB. The 6 GB version only just fits 32K context, and 4 GB doesn't fit at all.
- **4 GDDR7 chips:** 512 GB/s, 111.7 tok/s, 8 or 12 GB.
- **8 GDDR7 chips:** 1,024 GB/s, 223.5 tok/s, 16 or 24 GB.

The model needs about 5.6 GB at 32K context (5.04 GB of stored weights including the embedding table, plus int8 KV and state), plus working space. An earlier version said 5.03 GB because it left out the embedding table.

## What this shows

- **Per part, GDDR7 gives about 1.9× the bandwidth of LPDDR5X.**
- **LPDDR5X over-provisions capacity.** To get 273 GB/s you end up with 32 GB or more, while we need about 5.5 GB. We'd be paying for roughly 26 GB we never use.
- **GDDR7's capacity per chip is about right for this model.** Four 3 GB chips give 12 GB and 512 GB/s. That's the same memory setup as a 128-bit GDDR7 graphics card.
- **So under tokens per dollar, GDDR7 may well win,** unless its price per chip is much higher. We need cited prices before deciding.
- **GDDR7's costs:**
  - higher power
  - a bigger and more complex PHY (more die edge and area)
  - a more demanding board
  - These all go into the unit cost, so the comparison has to include them, not just the memory chips.

## Current leaning

**Four GDDR7 chips on a 128-bit bus (512 GB/s), with LPDDR5X 256-bit as the fallback.** Updated 2026-11-10: 2 GB chips (8 GB in total) are enough, since we need about 5.6 GB at 32K context. The 3 GB chips (12 GB) would add cost for capacity we don't use. This is the same memory setup as the RTX 5060 (`07_competitors_and_goal.md`). To be decided once we have:
- prices per GB and per chip for LPDDR5X and GDDR7 (cited, dated)
- PHY area and power estimates for both
- a check on whether the real batch-1 utilization of a 128-bit GDDR7 GPU leaves room for us to beat it

### Options we keep open

- **Bus width stays a parameter** in the RTL and the script, so we can report several points on the cost-speed curve.
- **On-chip SRAM for the DeltaNet state** gains only about 1% decode speed at batch 1 (59.6 → 60.3 tok/s at 273 GB/s), and costs 25 MB of SRAM, which is a lot of die area. Under tokens per dollar it's very likely not worth it.

## How a DRAM fetch actually works

You need this picture to design the DMA engine and the weight layout.

### The structure

- A DRAM system is split into **channels**, which work independently. Each channel has its own command and data wires. LPDDR5X channels are 16 bits wide, so a 256-bit bus is 16 channels.
- Each channel has several **banks** (16 in LPDDR5X). Each bank can be doing something different at the same time.
- Each bank is a grid of **rows**. A row holds about 1 to 2 KB in total.

### One read, step by step

1. **Activate.** The controller tells a bank to open a row. The whole row is copied into the bank's *row buffer*. This takes about 18 ns (the parameter called tRCD).
2. **Read.** The controller reads a column from the open row. After a delay (the CAS latency, also around 15 to 20 ns), data streams out as a burst. On LPDDR5X a burst is 16 transfers on a 16-bit channel = 32 bytes, and controllers usually fetch 64 bytes at a time.
3. **More reads from the same row are cheap.** That's a *row hit*: no new activate needed, just the column delay.
4. **Precharge.** Before opening a different row in that bank, the controller must close the current one. That's another ~18 ns (tRP).
5. **Refresh.** Every few microseconds, DRAM must pause banks to refresh their charge. That's one source of timing uncertainty, discussed in the VLIW notes.

**End to end,** from the chip sending a request to data arriving is about 100 ns, once you include the memory controller and the PHY (the circuit that drives the actual pins).

### What this means for our design

- **Read weights in long, contiguous runs.** If the next 2 KB we need sits in the already-open row, every read is a row hit and the bus stays near peak. We know every weight and its read order ahead of time, so we can store them in DRAM in exactly that order. That's a direct advantage of a model-specific chip.
- **Spread consecutive data across channels and banks.** Then while one bank precharges, another is streaming. The address mapping (which address bits pick channel, bank and row) is a design choice we control.
- **Keep enough requests in flight.** By Little's law, bytes in flight = bandwidth × latency. At 273 GB/s and 100 ns that's 27.3 KB, about 427 requests of 64 bytes always outstanding. At 150 ns it's 640. If the DMA engine can't keep that many in flight, the bus sits idle no matter how fast the rest of the chip is.
- **Leave room for refresh.** Peak bandwidth is never fully reached. Expect roughly 85 to 95% of peak in practice. We'll measure this with a DRAM timing model in simulation.

## Terms worth knowing

- **Burst length:** how many transfers one read command returns.
- **tRCD, CAS latency (CL), tRP:** the delays to open a row, to read from an open row, and to close a row.
- **Row-buffer hit rate:** the fraction of reads that find their row already open.
- **Bank-level parallelism:** how many banks are busy at once.
- **PHY:** the analog circuit on the chip that drives and samples the memory pins.
- **Beachfront:** the length of die edge used up by memory pins. A wider bus needs more of it, which is one reason buses can't just get wider forever.
- **Memory controller reorder queue:** the controller reorders requests to get more row hits. With our fixed, known access pattern we may barely need it.

## Open items

- **Real LPDDR5X timing numbers** for our DRAM model: we need a cited datasheet or JEDEC summary, not the rough values above.
- **The DRAM model for simulation:** probably DRAMsim3 or Ramulator 2, connected to Verilator. To be decided in `06_build_and_verify_plan.md`.
- **Energy per bit** for LPDDR5X vs SRAM, for the energy story.
