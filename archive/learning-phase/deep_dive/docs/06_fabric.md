# Crossbars and networks on chip

Data rarely starts where it is needed. On-chip fabrics move it. The most useful sentence on this topic comes from Irrational Analysis: memory, whether a block of SRAM or an HBM stack, is just a node on the network on chip, so how you set up that network matters a lot. [IA]

Roune adds the mental tool that makes this tractable: think of the math, and remember it has to happen on a 2D plane. [R]

::viz fabric

## The three basic topologies

| | Ring | Mesh | Crossbar |
| --- | --- | --- | --- |
| Wires | Scale with P | Scale with P | Scale with P² |
| Average hops | P/4 | About (2/3)·√P | 1 |
| Bisection | 2 links | √P links | P/2 links |
| Latency | High for large P | Medium | Lowest |
| Example | AMD CCD | Sapphire Rapids, inside an Nvidia GPC | Ironwood core NoC, Nvidia above GPCs |

**Rings** are simple and cheap. Each stop adds a hop, so they suit a small number of agents. AMD's Genoa-X uses nested rings: a ring within each 8-core chiplet, and those chiplets connect through a central I/O die. [IA]

**Meshes** scale well, and suit a tiled floorplan. Wires are short and regular. Sapphire Rapids uses a mesh with a memory controller and accelerators as mesh nodes, so DRAM data takes several hops to reach middle cores. [IA]

**Crossbars** connect any input to any output in one step. They are flexible and fast. They are also expensive, as the encyclopedia notes: their wiring and area costs grow quickly with port count. [E]

Real chips mix these. Nvidia uses meshes inside graphics processing clusters, with a crossbar tying them together. A crossbar can group a cluster that then joins a ring or mesh. Not every element needs a path to every other element. [IA]

## Why crossbars grow as P²

A crossbar has one crosspoint for every input-output pair. With P ports of width W bits, you need P² × W crosspoints, or an equivalent tree of multiplexers. The worse problem is wiring: all P × W bits must physically cross the middle of the block.

Estimate the block's width from wiring alone:

`width ≈ P × W × wire_pitch ÷ routing_layers`

Example: 64 ports × 512 bits = 32,768 wires. At an 80 nm pitch on 4 layers, that is 0.66 mm of pure wire before any logic, in each direction. [D, O] It also means long wires and more cycles of latency. Doubling ports doubles that width and quadruples the area.

## Where crossbars appear in AI chips

- **SRAM banks to lanes.** Letting any lane read any bank.
- **Inside every router.** A NoC router is a small crossbar with buffers and an allocator. [E]
- **Permute units.** Lane shuffles and transposes.
- **Between cores.** Ironwood's memory and DMA interconnect. [IA]
- **In the system.** NVSwitch is a crossbar for whole GPUs. [R]

## Banks and conflicts

SRAM with many ports is expensive, so a wide memory is built from many single-port banks with a fabric in front of them. Now two requesters can collide on the same bank.

Under uniform random access, P requesters over B banks serve this many requests per cycle:

`served = B × (1 − (1 − 1/B)^P)`

| Requesters | Banks | Throughput per requester [D] |
| --- | --- | --- |
| 16 | 16 | 64% |
| 16 | 32 | 80% |
| 16 | 64 | 89% |

Random access wastes a third of your bandwidth with equal banks and requesters. Designers fix this in several ways:

- **Provision extra banks.** Two banks per port reaches 80%.
- **Hash the address.** XOR bits so that strides spread across banks. [E]
- **Pad arrays.** A row stride that is not a multiple of the bank count avoids systematic conflicts.
- **Make access deterministic.** Tile layouts so that software guarantees no conflicts. This is how a compiler-managed scratchpad earns its keep.

In a well-scheduled matmul, there should be no random access at all. Conflict-free layouts let you drop much of the fabric. This is one more place where software capability buys hardware simplicity.

## Cheaper than a crossbar

| Option | Cost | Limits |
| --- | --- | --- |
| Full crossbar | P² | None |
| Clos, three stages | About P^1.5 | More latency |
| Beneš or butterfly | P log P | Permutations only, needs routing setup |
| Fixed shuffle | P | Only the patterns you chose |
| Time-multiplexing | Lower width | Lower throughput |
| Private banks per client | None | No sharing |

Ask what traffic you truly need. An array that only needs transposes and row shifts can use a permutation network far cheaper than a crossbar. [E]

## Router anatomy

A NoC router contains input buffers, route computation, a virtual-channel allocator, a switch allocator and a small crossbar. [E] Each element exists for a specific reason.

- **Virtual channels** stop one stalled packet from blocking others on a link, and can break deadlock cycles.
- **Credits** stop a sender from overrunning buffers when feedback takes several cycles. [E]
- **Escape channels** give a proven path to forward progress. [E]
- **Allocators** resolve contention each cycle.

Each is hardware that exists because traffic is unpredictable.

## Determinism removes the machinery

Groq's slides list what it dropped: no arbiters, no reactive components, no reordering, and no link-layer replay. Functional units are stateless and side-effect free, and data moves on fixed stream registers in one of two directions. Instruction dispatch takes under 3% of the area. [IA]

That is the logical end of this line of thought. If a compiler knows every data movement on every cycle, you need no buffers for contention, no allocators and no flow control. The hardware becomes simple and fast. The cost is that every chip must stay in lockstep, and the compiler must schedule the entire system.

Most designs sit somewhere between. TPUs use compiler-controlled DMAs with hardware queues. GPUs use full dynamic scheduling. The right point depends on how predictable your workload is, and AI workloads are predictable: few branches, and the ones that exist are rarely data-dependent. [R] Dynamic routing exists for unpredictable traffic, which matmul generally is not. MoE is the exception.

## The NoC as a power consumer

Moving a byte a millimeter costs energy on every hop, and data-movement energy adds up to a large fraction of chip power. Keep producers close to consumers.

- Place SRAM banks right next to the array edges they feed.
- Give each array its own local memory, with the global fabric carrying only what must be shared.
- Size fabric bandwidth to steady-state flow, then add margin for bursts.

## Tuning reference

| Knob | Effect | Cost |
| --- | --- | --- |
| Port count | More parallel clients | Quadratic in a crossbar |
| Port width | More bandwidth | Wiring, linear in width |
| Hierarchy levels | Keeps crossbars small | Non-uniform latency |
| Banks per port | Fewer conflicts | Area overhead per bank |
| Address hashing | Removes stride conflicts | Small logic, harder debug |
| Router buffers | Absorb bursts | Area, leakage |
| Virtual channels | Avoid blocking | More buffers and allocators |
| Static schedule | Removes arbitration | Compiler effort, brittle |
