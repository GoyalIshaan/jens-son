# 09. The background machinery: clocks, on-chip networks and everything else

*Started 2026-11-10. Status: explainer plus first thoughts. Nothing here is decided.*

The architecture sketch (`05_architecture_sketch.md`) shows the blocks that do the work: control, DMA, SRAM, the matrix unit and the vector unit. A real chip also needs a lot of machinery nobody draws on the block diagram: something to keep time, wires to move data, power, startup, a way to talk to the host, a way to see inside, and a way to test it at the factory. This note goes through each piece: what it is, a small worked example, and what it means for our chip.

---

## 1. The clock

### What it is

Almost everything on the chip is built from **flip-flops**: tiny 1-bit memories that copy their input to their output only at the moment the clock signal rises. Between flip-flops is **combinational logic**: adders, multiplexers, comparators. Each clock tick, every flip-flop captures a new value at once, and the logic between them has one clock period to compute the next value.

At **1 GHz**, the period is **1 nanosecond**. So the rule is:

> The slowest path from one flip-flop, through logic, to the next flip-flop must finish within 1 ns, with margin.

That slowest path is the **critical path**, and it sets the maximum clock speed.

### A worked example

Say one stage of our matrix unit multiplies an int4 weight by an int8 activation, then adds the product into a 32-bit accumulator.
- **Multiply:** about 0.35 ns
- **32-bit add:** about 0.30 ns
- **Flip-flop overhead** (time to capture, plus time for the output to appear): about 0.10 ns
- **Clock uncertainty margin** (skew and jitter, see below): about 0.05 ns
- **Total: 0.80 ns.** That fits in 1 ns with 0.20 ns of *slack*.

These delays are illustrative for a modern node. Synthesis will give the real ones.

If a path needs 1.3 ns, you have three choices:
- **Slow the clock** to about 770 MHz. Everything gets slower.
- **Pipeline:** put a flip-flop in the middle. The result now takes 2 cycles, but you still get one new result per cycle.
- **Make the logic faster:** a better adder design, or less work per stage.

This trade-off is called **timing closure**, and it's most of the work after RTL is written. The synthesis tool reports the critical path and its slack.

There's a second rule, **hold time:** a path must not be *too fast* either, or the next flip-flop sees the new value before it has finished capturing the old one. The tools fix this automatically by adding delay.

### Where the clock comes from: the PLL

Nothing on a board produces a clean 1 GHz signal directly. A crystal or the PCIe connector gives a slow, very stable **reference clock**, typically 100 MHz. A **PLL** (phase-locked loop), an analog block, multiplies it: 100 MHz × 10 = 1 GHz. A PLL takes time to *lock* (settle) after power-up, so the rest of the chip must wait for its "locked" signal before starting. PLLs are licensed IP, like PHYs.

### Getting the clock everywhere: the clock tree

The clock has to reach every flip-flop on the chip, potentially millions, at almost the same moment.
- **The clock tree:** a tree of buffers that fans one signal out across the die. The tools build it automatically.
- **Skew:** the difference in arrival time between two flip-flops. Skew steals from the 1 ns budget.
- **Jitter:** the clock edge wobbling slightly from cycle to cycle. That also steals time.

The clock tree switches every single cycle, so it uses a lot of power, often quoted as a sizable fraction of a chip's dynamic power. That's why chips use **clock gating:** turn off the clock to a block that has nothing to do this cycle. Our vector unit waiting for the matrix unit is an obvious candidate.

### What it means for us

- **Our target is 1 GHz,** and every number in the budget script assumes it.
- **Decode speed doesn't depend on the core clock.** The memory sets the speed. A slower core clock only matters if compute can't keep up, which at batch 1 it easily can. So we can pick a modest clock (say 800 MHz to 1 GHz) that closes timing easily and uses less power.
- **Prefill speed does scale with the clock.** A 64×64 array at 1 GHz is about 8.2 TOPS; at 800 MHz it's about 6.6 TOPS.

---

## 2. Several clocks on one chip: clock domains

### Why there's more than one clock

Different parts of the chip have to run at different speeds:
- **The core** (control, DMA, SRAM, matrix and vector units): our chosen clock, say 1 GHz.
- **The GDDR7 interface:** each pin runs at 32 Gb/s. The PHY handles the very fast part and hands data to the memory controller on a slower, much wider bus. For example, one 32-bit chip at 32 Gb/s delivers 128 bytes per ns. At 1 GHz that's a 128-byte (1,024-bit) word each cycle per memory chip.
- **PCIe:** its own clock from the host link.
- **Debug (JTAG):** a slow clock, a few MHz, driven from outside.

Each group of logic running on the same clock is a **clock domain**.

### The danger: crossing between domains

When a signal goes from one clock domain to another, the receiving flip-flop may capture it exactly while it's changing. The flip-flop can then sit halfway between 0 and 1 for a moment, which is called **metastability**. Then it settles randomly. If that bad value spreads, the chip fails, rarely and unpredictably. Those are the worst bugs to find.

The fixes:
- **A single bit (a flag):** pass it through two flip-flops in a row in the receiving domain (a **synchronizer**). The first may go metastable, but it almost certainly settles before the second captures it.
- **Data:** use an **asynchronous FIFO**. The writer pushes data with its clock and the reader pops with its own. The read and write positions cross between domains as **Gray code**, a counting scheme where only one bit changes per step. If the receiver catches a position mid-change, it reads either the old or the new value, never a nonsense one.

Example: pointer 3 → 4 in plain binary is 011 → 100, so three bits change at once. A receiver catching it mid-change could read 111 = 7, which is badly wrong. In Gray code, 3 → 4 is 010 → 110: only one bit changes, so the receiver reads either 3 or 4.

### What it means for us

- **We keep the number of clock domains small:** core, memory, PCIe and debug.
- **Every crossing goes through a standard, well-tested FIFO or synchronizer.** No hand-made crossings.
- **In simulation, we start with one clock** and add the real domains later, with a clock-domain-crossing check tool.

---

## 3. Moving data around: buses, crossbars and networks on chip

### The bandwidth problem in wires

Bandwidth on chip = **bits per cycle × clock frequency**. Our memory delivers 512 GB/s. At 1 GHz that's **512 bytes every cycle, or 4,096 bits.** So the path from the memory controllers to the SRAM must be about 4,096 wires wide, and every block on that path must accept 512 bytes per cycle. Compare a typical on-chip bus of 64 to 128 bits: we need 32 to 64 times that.

### The three ways to connect blocks

- **A shared bus:** one set of wires everyone takes turns on. Simple and small, but only one transfer at a time. Fine for slow traffic like configuration registers.
- **A crossbar:** every sender has a path to every receiver, and several transfers can happen at once. The cost grows with senders × receivers × width, so a wide crossbar between many blocks gets huge. Our earlier learning chapter on fabrics covered this.
- **A network on chip (NoC):** routers connected in a grid, with data split into packets that hop from router to router, like a tiny internet. It scales to many blocks (large SoCs and many-core chips use these), but adds latency, area and complexity.

### What our chip needs

Our traffic is unusual: a few **huge, fixed, predictable** flows, plus a little control traffic.
- **Flow 1, the big one:** memory controllers → weight SRAM → matrix unit. 512 bytes per cycle, every cycle.
- **Flow 2:** activations and partial sums between SRAM, the matrix unit and the vector unit. Small at batch 1.
- **Flow 3:** KV cache and DeltaNet state, both directions between DRAM and SRAM. Moderate.
- **Control traffic:** the control processor writing descriptors and configuration registers, and reading status. Tiny.

So the first-pass proposal is **no general network at all:**
- **Dedicated wide paths** for flows 1 to 3. The connections are known at design time, so they're just wires and registers, laid out so each flow is short and direct.
- **A small, simple bus** for control traffic: an APB- or AXI-Lite-style register bus.

This is a model-specific chip's advantage again. General chips need flexible networks because they don't know their traffic ahead of time. We do.

### SRAM banking: feeding 512 bytes per cycle

A single SRAM block can only read or write one word per cycle, and SRAM blocks come in limited widths. To read 512 bytes per cycle, we split the SRAM into **banks** that work in parallel. For example, 16 banks × 32 bytes each = 512 bytes per cycle.
- **The catch, bank conflicts:** if two reads want the same bank in the same cycle, one must wait.
- **Our way out:** we control the data layout, so the compiler places data so the hardware never needs two things from one bank at once. A general chip has to handle conflicts in hardware; we avoid them in the schedule.

### Wires take time too

A signal can only travel so far in one clock period. Across a large die, a wire can take more than a cycle. The fix is to put flip-flops along long wires (**pipeline registers**, also called register slices), so each segment fits in a cycle. That adds a few cycles of latency, which DMA queues absorb easily.

This is also why the **floorplan** (where each block physically sits) matters:
- **memory controllers and PHYs** along the die edges, near the memory chips
- **weight SRAM** right next to them
- **the matrix unit** next to the SRAM

The longest, widest flows get the shortest distances.

---

## 4. Power

### Where it comes from

The board's voltage regulators supply several **rails**, at different voltages:
- **core logic:** under 1 V, typically around 0.75 V on modern nodes
- **the PHYs:** need their own clean supplies
- **the IO pads**

Each rail reaches the die through package bumps. On the die, a grid of thick metal wires (the **power delivery network**) spreads it to every transistor.

### Things that go wrong

- **IR drop:** current through the grid's resistance lowers the voltage. A busy block far from the bumps might see 0.70 V instead of 0.75 V, and runs slower. Timing must still close at that lower voltage.
- **Sudden load changes:** when the whole matrix unit switches on at once, the current jumps, and the voltage dips for a moment. On-chip capacitors (**decaps**) act as small local reservoirs to smooth this out.

### Saving power

- **Clock gating**, covered above.
- **Power gating:** cut power completely to a block that's idle for a long time, for example the matrix unit between requests.
- **DVFS** (dynamic voltage and frequency scaling): lower voltage and clock when full speed isn't needed. During decode, the core doesn't need full speed, so this fits our workload well.

### What it means for us

For a memory-bound chip, much of the energy goes into **moving data from DRAM**, both in the memory chips and in our PHYs. Our logic is comparatively small. We need cited energy-per-bit figures for GDDR7 and SRAM to make this precise (an open question). Lower power also means a cheaper cooler and power supply, which feeds back into unit cost.

---

## 5. Starting up: reset and boot

When power comes on, every flip-flop holds a random value. The chip has to be brought into a known state in a fixed order:

1. **Power is stable.** A power-on-reset circuit holds everything in reset until the supply voltages are good.
2. **The PLL locks.** Wait for its "locked" signal.
3. **Release reset per clock domain.** Reset is released in sync with each domain's clock, so all flip-flops leave reset on the same edge. (Releasing it at a random moment would cause the same metastability problem as a clock crossing.)
4. **Memory training.** At 32 Gb/s, tiny differences in wire length on the board matter. The PHY and controller run a **training** routine: they send test patterns and tune timing and voltage levels on each pin until reads and writes are reliable. Every boot, before any real data moves.
5. **Load firmware.** A small **boot ROM** (fixed at manufacture) lets the host load our control program and configuration over PCIe.
6. **Load the model.** The host copies the weights into DRAM, in our chosen layout.
   - Over PCIe Gen4 x4, about 8 GB/s, 4.46 GB takes about 0.6 seconds.
   - This happens once per session, not per token.
7. **Ready.** The host sends a prompt.

### What it means for us

- **The reset and boot sequence is a small state machine we write.**
- **The training is part of the licensed PHY and controller.**
- **In simulation we skip training** and start with memory already loaded, but we still test the reset sequence.

---

## 6. Talking to the host

- **PCIe** connects the card to the host computer. We'd buy the PCIe controller and PHY as IP.
- **Registers (MMIO):** the host reads and writes our chip's control registers as if they were memory addresses. "Start", "stop", "status" and "token ready" are all registers.
- **Doorbells and interrupts:**
  - The host writes a *doorbell* register to say "new work is in the queue".
  - The chip sends an *interrupt* to say "your tokens are ready".
  - Each token is only a few bytes, so the host link is never a bottleneck. At 100 tok/s that's well under 1 KB/s.
- **The driver:** host software that loads firmware and weights, sends prompts and reads back tokens. We'd write a small one, or for the simulation, a testbench that plays the host.

---

## 7. The register map

Every setting and status bit the host or firmware can touch gets an address in a **register map:**
- **Control:** start, reset, mode.
- **Status:** busy, error flags.
- **Configuration:** model dimensions, if we let the program set them; base addresses of weights, KV cache and state in DRAM.
- **Counters:** see below.

We'll write the register map as one machine-readable file and generate both the RTL and the documentation from it. That keeps them from drifting apart, and it's part of the public instruction set promised in D-009.

---

## 8. Seeing inside: debug and performance counters

When something goes wrong, or is just slow, you need to see what happened.
- **JTAG:** a standard debug port, a few slow pins, that can halt the control processor, read registers and memories, and step through the program.
- **Performance counters:** hardware counters that count cycles when each unit is busy, cycles the DRAM bus is idle, cycles the control processor waits on each sync flag, bytes moved, and so on. For us these are essential:
  - They measure **bandwidth utilization**, our main design lever.
  - They're the **reward signal for the RL compiler:** "fewer cycles" and "where did the cycles go" both come from here.
- **Trace buffer:** a small memory that records the last few thousand events (instructions issued, DMA starts and ends) for replay after a failure.

---

## 9. Manufacturing test (design for test)

Every chip from the factory must be tested, because some will have defects (`08_post_design_levers.md`). Testers can only reach the chip through its pins, so the chip has to help test itself:
- **Scan chains:** in test mode, all flip-flops are linked into long shift registers. The tester shifts in a pattern, clocks the chip once, and shifts out the result. Software (ATPG, automatic test pattern generation) picks patterns that catch nearly every possible stuck wire.
- **Memory BIST:** each SRAM gets a small built-in self-test engine that writes and reads patterns at full speed. Some designs have spare rows to repair a defective SRAM.
- **The cost:** a few percent of area, plus some test logic in the clock and reset paths. The tools insert most of it automatically after we write the RTL.

For the open-source tapeout attempt, we'd include at least scan chains, because a design without them isn't credible as a real chip.

---

## 10. The physical edge: pads, bumps and ESD

- **Bumps:** the die connects to the package through solder bumps, typically on a grid across the die. Every signal, power and ground connection needs bumps. A GDDR7 interface needs many of them close to its PHY, which pushes the PHYs to the die edge (the *beachfront* in `04_memory.md`).
- **IO cells and ESD protection:** a touch of static electricity can be thousands of volts. Every pin gets protection circuitry to absorb it. That's part of the IO library we'd buy.
- **Sensors and fuses:**
  - **temperature sensors** for throttling when the chip gets too hot
  - **voltage monitors**
  - **one-time fuses** for a chip ID, or to record repairs found during test

---

## 11. What we write vs what we buy

**We write (RTL):**
- the control processor and its VLIW instruction set
- the DMA engines
- the matrix unit and vector unit
- the SRAM banking and the dedicated data paths
- the register map and control bus
- the reset sequencer
- the performance counters
- the clock-domain crossing FIFOs (from a standard design)

**We buy as IP** (or, in simulation, replace with simple behavioural models):
- the PLLs
- the GDDR7 PHY and memory controller (we may write a simplified controller ourselves)
- the PCIe controller and PHY
- the SRAM macros, generated by the foundry's memory compiler
- the IO and ESD cells
- the temperature sensors

**The tools insert:**
- the clock tree
- scan chains and memory BIST
- the power delivery network
- hold-time fixes

## How this affects the simulation plan

- **Phase 1:** one clock, behavioural memory model, no PCIe (the testbench pokes registers directly). Goal: correct tokens.
- **Phase 2:** real clock domains with async FIFOs, a detailed DRAM timing model, performance counters. Goal: realistic cycle counts and bandwidth utilization.
- **Phase 3, for the tapeout attempt:** synthesis with real SRAM macros from the open process kit, scan insertion and timing closure for one tile.

## Open questions

- **Core clock:** 800 MHz or 1 GHz? Depends on the prefill floor and the critical path of the matrix unit.
- **Can the control traffic** really stay on a simple bus, or does the DMA engine need something more?
- **The GDDR7 controller-side interface:** how wide is it per channel, and at what ratio to the core clock, for the PHY we'd license?
- **Energy per bit** for GDDR7 and SRAM, with sources.
