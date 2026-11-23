# The frame: what you are optimizing

Chip design has more interacting decisions than anyone can hold in their head. This chapter gives you the frame that keeps them organized. Every later chapter hangs off it.

## Start from a goal you can measure

A chip is a bet that a certain workload will run at a certain cost. Before any block diagram, you need four things written down.

| Decide first | Why it controls everything |
| --- | --- |
| The workload: model family, sizes, context length, precision | Sets matmul shapes, which set array size |
| The service target: tokens per second per user, first-token time | Sets batch ceiling, which sets arithmetic intensity |
| The cost metric: tokens per dollar, or per watt | Tells you what a saved mm² or watt is worth |
| The software you control | Sets how much overprovisioning you must buy |

Roune's central tension is between maximizing tokens per dollar and overprovisioning so that customers get something familiar. [R] His claim is that large HBM, big routed networks and host computers exist partly to protect unsophisticated software. If you control the whole stack, you can buy less hardware. If customers bring arbitrary software, you need margin.

Jouppi's team adds that the metric itself moved. Performance per watt has overtaken performance per total cost of ownership, because electricity for new data centers is now the constraint. [J] Your metric depends on what is scarce for your customer.

## The four-question description of any computer

Irrational Analysis proposes that any machine can be compared on four attributes, however strange it is. [IA] Use this every time you read about a new chip.

**1. Memory hierarchy.** Is on-chip SRAM a cache or a scratchpad? What does a miss cost? How many levels are there? What is the ratio of SRAM to DRAM?

**2. Memory access and routing.** Given a piece of memory and a piece of compute, how does data get between them? Ring, mesh, crossbar, or a mix? Memory is a node on the network, so this matters as much as the memory itself.

**3. Compute structure.** How fancy is each core (speculation, out-of-order)? How many operands per instruction (SIMD, warps, systolic arrays)? Which data formats?

**4. Chip-to-chip structure.** All-to-all through switches, a torus, dragonfly? This is often where the real differences hide.

| | Memory | Routing | Compute | Chip to chip |
| --- | --- | --- | --- | --- |
| AMD Genoa-X | Caches, large L3, stacked SRAM | Nested rings | Very fancy, weak on matrices | Dual socket, no scale-up fabric |
| Intel Sapphire Rapids | Caches, sliced L3 | Mesh | Very fancy | Up to 8 sockets |
| Nvidia GB300 | Caches plus scratchpad, HBM | Mesh inside GPC, crossbar above | Medium fancy, warp-32 plus tensor cores | 72-way all-to-all over copper |
| Google Ironwood | Scratchpad, HBM | Crossbar | Medium, big systolic arrays, VLIW | 3D torus with optical switches |
| Groq LPU | Scratchpad only, no DRAM | Streaming lanes, statically scheduled | Simple, 144-wide VLIW, 320-wide SIMD | Dragonfly, software-scheduled |

Adapted from the summary table in [IA].

## Think in levels

The encyclopedia makes a point that is easy to forget: its 40 categories are not all at the same level. "A matrix engine contains arithmetic units, gates, and transistors." [E] A useful stack, from bottom to top:

1. **Devices and wires.** Transistors, metal layers, vias. Sets delay, leakage and energy per switch.
2. **Cells.** Gates, flip-flops, SRAM bitcells. What your synthesis tool sees.
3. **Datapath blocks.** Multipliers, adders, register files, FIFOs.
4. **Engines.** Systolic arrays, vector units, DMA engines.
5. **Core.** An engine set with its control and local memory.
6. **Die.** Cores plus NoC, memory controllers, PHYs.
7. **Package.** Dies plus HBM, interposer, power delivery.
8. **System.** Boards, links, switches, cooling.
9. **Software and model.** Which decides how much of all the above you use.

When someone argues about a chip, ask which level they are arguing at. A claim true at level 3 can be irrelevant at level 8.

## The encyclopedia's five questions

For any block you consider adding, the encyclopedia ends with a test that is worth memorizing. [E]

- What work does it perform?
- What supplies its data?
- What limits its throughput?
- How much state and routing does it require?
- Does the target process or IP library support it?

## Balance, stated precisely

A balanced chip is one where, on the target workload, no resource is far more idle than the others. Roune lists the things that must be balanced together. [R] Systolic compute, vector compute, scalar compute, memory bandwidth, memory capacity, network bandwidth, watts and thermals, thermal throttling, and the latencies of various operations.

Three ratios give you most of the picture.

| Ratio | Formula | What it tells you |
| --- | --- | --- |
| Ridge point | Peak FLOP/s ÷ memory bytes/s | The reuse software must achieve per byte |
| Capacity ratio | HBM bytes ÷ peak FLOP/s | How much state each unit of compute can hold |
| Network ratio | Link bytes/s ÷ peak FLOP/s | How much parallelism you can afford |

The whole history of TPUs can be read from these three numbers, which we do in the *Generations* chapter.

## System performance is the only performance

Per-chip numbers can mislead you. A chip half as fast with half the HBM costs the same HBM per unit of work. A network one-eighth the price on a chip one-eighth as fast saves nothing. [R] Always divide by realized system performance. This is also the force that pushes chips to be large: fixed costs per chip, such as a host, a package and network ports, amortize better over more compute.

Goodput is the system-level version of this idea. A system that spends 10% of its time recovering has a goodput of 0.9× its throughput. Google reports 93% goodput for Gemini 2.5 across multiple pods. [J]

## How to use the rest of this guide

Read *Area, clock, power* next for the physical budget. Then read *Systolic arrays* and *Matmul shape* together, since they are the center of everything. The chapters on reduction trees, crossbars, memory, control and vector units cover what surrounds the array. *Workloads* and *Scale-out* cover how software and systems set requirements. *Generations* shows how real designers moved through all of this. *Method* turns it into a procedure, and *Your chip* applies it to your RTL.
