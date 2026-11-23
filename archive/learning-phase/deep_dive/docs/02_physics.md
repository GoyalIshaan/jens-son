# Area, clock, power

The PDFs mostly assume you already know the physical budget, so much of this chapter is **[O]**: standard engineering, with ballpark numbers you should replace using a real PDK. What the PDFs add is how designers use these constraints. Treat every number marked [O] as a starting estimate. The method matters more than the values.

::viz die

## The three budgets are one budget

Area, clock and power trade against each other through a few equations. You rarely get to choose all three.

- **Throughput** = number of MACs × clock × utilization.
- **Power** ≈ activity × capacitance × voltage² × clock + leakage.
- **Clock** rises with voltage, and falls with logic depth and wire length.
- **Area** sets how many MACs and how much SRAM you can have, and how much leakage you carry.

Because you can reach the same throughput with many slow MACs or few fast ones, the real question is always which budget binds first.

## Area

### The hard ceiling

A single exposure on a lithography tool covers about 26 × 33 mm, or roughly 858 mm². [O] No monolithic die can be larger, apart from wafer-scale tricks like Cerebras. [R] Designs that want more use several dies in a package. Ironwood uses two compute dies, and Blackwell Ultra pairs two dies with a 10 TB/s link. [J, IA]

### Yield makes big dies expensive

With a simple Poisson model, the fraction of good dies is `exp(−area × defect_density)`. At a defect density of 0.1 per cm²:

| Die area | Good dies [D] |
| --- | --- |
| 200 mm² | 82% |
| 400 mm² | 67% |
| 600 mm² | 55% |
| 800 mm² | 45% |

The usual answer is redundancy. Memory has spare rows and columns. GPUs ship with some cores disabled. Ironwood applied the memory idea to compute by adding a redundant row in the MXU. [J] A regular array is ideal for this, since one spare row can cover a defect anywhere in the array.

### Where area goes

| Block | How it scales | Rough size [O] |
| --- | --- | --- |
| 8-bit MAC with registers | Shrinks with logic density | About 100–150 µm² at 5 nm |
| 16-bit float MAC, FP32 accumulate | About 5× an int8 MAC | About 500–800 µm² at 5 nm |
| SRAM macro | Shrinks slowly | About 0.3–0.4 mm² per MiB at 5 nm |
| HBM PHY | Barely shrinks, needs die edge | About 15 mm² per stack |
| High-speed SerDes | Barely shrinks, needs die edge | About 0.3–0.5 mm² per lane |

Calibration point: TPU v1 spent about 24% of a 331 mm² die at 28 nm on 65,536 int8 MACs, about 1,200 µm² per MAC. [O] Arrays are a smaller share of real dies than people expect. Vector units, NoC, control and I/O add up quickly, so the lab budgets those at 50% of array area plus a fixed amount. Even so, expect real dies to come out larger than the lab shows.

### SRAM stopped scaling

Jouppi states it plainly: SRAM density grew more slowly than logic density, so VMEM only quadrupled from 32 to 128 MiB in eight years despite a much larger die. [J] This one fact drives a lot of current design. It is why AMD stacks SRAM dies on its CPUs, why hybrid bonding is attractive for Groq-style designs, and why SRAM chiplets on older nodes make economic sense. [IA]

AMD's lesson is instructive. It stacks extra cache only on top of existing SRAM and puts blank structural silicon over the CPU cores, because stacking on hot compute is hard. [IA]

### Beachfront: the budget people forget

HBM PHYs and SerDes must sit at the die edge. Edge length grows with the square root of area, while compute grows with area. So as dies get bigger, each mm² of compute has less edge available for I/O. The area question becomes a perimeter question. This is one reason multi-die packages exist: two dies have more total edge than one die of the same total area.

Ironwood's eight HBM stacks sit on either side of its two compute dies, with interconnect SerDes on a separate chiplet. [J, IA]

## Clock

### What sets the period

`period = clock-to-q + logic delay + setup + skew + jitter`

Everything except logic delay is overhead. If overhead is about 80 ps [O], a 500 ps period gives you 420 ps of useful work. Halve the period and useful work falls to 170 ps, so you need more than twice the pipeline stages, with more flip-flops and more power. That is why clock frequency has diminishing returns.

### Wires are the real limit

A repeated wire on mid-level metal takes on the order of 100 ps per mm. [O] Crossing a 20 mm die therefore takes about 2 ns. At 2 GHz that is four cycles, before you do any logic. Three consequences follow.

- **Local wiring wins.** A systolic array only talks to its neighbors, so its critical path does not grow with array size. This is its deepest physical advantage.
- **Global signals must be pipelined.** Anything broadcast to every processing element, like `enable` or `clear_acc`, cannot reach 65,536 cells in one cycle.
- **Big cores have a limit.** TPUs use two large TensorCores because that balanced longer wire latencies in one huge core against exposing many tiny cores that software must lash together. [J]

### Should the array run fast?

Early TPUs ran arrays at lower clocks so that the scalar and vector side could keep up. [R] Look at the implied clocks from Jouppi's Table 1 [D]:

| Generation | Implied clock |
| --- | --- |
| TPU v2 | 0.70 GHz |
| TPU v3 | 0.94 GHz |
| TPU v4 | 1.05 GHz |
| TPU v5p | 1.75 GHz |
| Ironwood | About 2.2 GHz, see the audit chapter |

From v2 to Ironwood, BF16 peak rose 50×. About 3× came from clock, and the rest came from more MACs. Clock was never the main lever.

Reasons to run an array slower:

- Control and feeding logic have more slack.
- Lower voltage cuts energy per operation.
- Hotspots shrink.

Reasons to run it faster:

- Fewer MACs for the same throughput, saving area and leakage.
- Lower latency per tile.

## Power

### The equation to memorize

`P_dynamic = α · C · V² · f`

α is the fraction of nodes switching per cycle, C is total switched capacitance, V is the supply voltage and f is the clock. Two things stand out. Voltage is squared. And α depends on your data, which is why worst-case power is so much higher than typical power.

Leakage power flows even when nothing switches, and grows with temperature. Roune notes that a hotter chip draws more power, causing more heat. [R]

### Energy per operation: the numbers that shape architecture

Horowitz's 45 nm numbers are the classic reference. [O] Absolute values have improved since, but ratios have held up reasonably well.

| Operation | Energy at 45 nm |
| --- | --- |
| 8-bit add | 0.03 pJ |
| 8-bit multiply | 0.2 pJ |
| 32-bit float multiply | 3.7 pJ |
| 32-bit read, 8 KB SRAM | 10 pJ |
| 32-bit read, 1 MB SRAM | About 100 pJ |
| 32-bit read, DRAM | About 640–1300 pJ |

Read that table as an architect would. A DRAM read costs thousands of times more than an 8-bit multiply. Reading a value from a large SRAM costs far more than computing with it. So every byte you fetch must be used many times. A systolic array earns its efficiency because a value fetched once is used across a whole row or column of MACs, moving only between neighbors. Systolic arrays produce far less heat per operation than other structures. [R]

Modern ballparks for planning [O]:

| Action | Energy |
| --- | --- |
| int8 MAC with local registers | About 0.2 pJ |
| BF16 MAC with FP32 accumulate | About 0.4–0.6 pJ |
| SRAM read per byte | About 1–3 pJ |
| HBM read per byte | About 30 pJ |
| Chip-to-chip copper per byte | About 15–40 pJ |

### The wide-and-slow trade

Because voltage is squared, you can trade area for energy. Take a teaching curve where frequency follows the alpha-power law, with threshold 0.3 V at a nominal 0.75 V. Dropping to 75% of the clock allows about 0.61 V, cutting dynamic energy per operation to about 65%. [D] You need 1.33× the MACs for the same throughput.

This is Roune's point about larger clocked-down arrays, and it is what Etched's low-voltage inference pitch relies on. [R, F] It pays off when power is the binding limit and area is available. It stops paying when leakage from the extra area, or the cost of that area, outweighs the saving.

### Heat is about density

Average power density for a 1 kW chip on 800 mm² is 1.25 W/mm². [D] Arrays can run several times hotter than that, since they pack so much computation into a small area. The TPU v3 story shows what happens if you get this wrong. [R]

- The team underestimated array heat, and it showed up when silicon arrived.
- The hardware throttle measured temperature after the fact, which was too slow for spikes inside the array.
- The fix was a compiler static analysis, validated against an empirical heat model, that refused to emit instruction patterns that could overheat the array.
- That allowed a 25% higher clock. Roune called it compiler throttling and later wished he had called it Turbo Mode.

The lessons are general.

**You cannot provision for the worst case.** Worst-case data and instruction patterns draw far more power than anything useful does. Every high-performance chip therefore throttles. Without it, a malicious cloud tenant could destroy hardware. [R]

**Throttle on current, not only temperature.** Measuring electrical power reacts faster than temperature sensors do. [R]

**Peak and sustained are different numbers.** A chip that throttles quickly can advertise a clock it cannot hold. Compare long-running benchmarks instead. [R]

**Placement is a thermal tool.** Spread arrays apart, or interleave them with cooler logic. [R]

### Delivering the current

A 1 kW chip at 0.75 V draws 1,333 amps. [D] Resistance in the power network turns directly into lost voltage, and every millivolt of droop eats timing margin. When a large array starts all at once, current jumps and the supply sags. Two responses appear in the sources: more decoupling and better delivery paths such as vertical power and backside power [J, E], and droop detectors that throttle before a timing failure [E].

A useful detail of systolic arrays: their diagonal wavefront ramps activity up gradually, which softens that current step compared with a design where everything starts on the same cycle. [O]

### Power-saving structures

Each saves something different, and each has a cost in the encyclopedia's terms. [E]

| Technique | What it saves | What it costs |
| --- | --- | --- |
| Clock gating | Clock tree and flop switching in idle logic | Enable timing, a gating cell |
| Operand isolation, zero gating | Datapath toggling on zero operands | A small amount of control |
| Power gating | Leakage in sleeping blocks | Wake latency, state loss unless retained |
| DVFS | Energy at lower demand | Regulators, characterization |
| Multiple Vt cells | Leakage on paths with slack | Library and sign-off complexity |

## Reasoning procedure

1. Compute required MACs from throughput ÷ (clock × target utilization).
2. Convert MACs to area and power using your node's numbers.
3. Add SRAM area for tiles, PHY area for bandwidth and control overhead.
4. Check perimeter against PHY and SerDes needs.
5. Check total watts against cooling, and peak density against the array.
6. If power binds, go wider and slower. If area binds, go narrower and faster, or use more dies.
7. Compute yield and decide what needs redundancy.

The lab above lets you do this interactively. Its constants are defaults you can replace, and they are not measurements of any product.
