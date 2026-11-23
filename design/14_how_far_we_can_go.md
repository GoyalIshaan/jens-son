# 14. How deep can we go before we have to stop?

*Started 2026-11-10. Status: under consideration (D-014, proposed). This refines the milestones in `01_goal.md`. Costs and services marked "to verify" haven't been checked.*

## The short answer

We can take the **digital chip all the way to a verified design and real layout on an open process,** and tape out a **small slice of it** cheaply. We have to stop at three walls:
- **The commercial process node** (TSMC N6 or similar), because of NDAs and foundry contracts.
- **The analog IP:** the GDDR7 PHY, the PCIe PHY and the PLLs. These are licensed for a lot of money.
- **Money:** masks, licences and commercial sign-off tools.

Past those walls, we replace the real thing with **models** (a GDDR7 timing model instead of a PHY) and **estimates** (area scaled from our open-process layout).

## The levels, in order

*Update 2026-11-10: `16_design_flow_and_verification.md` revises the order of the first steps. The golden model comes first, then architecture and the performance model in a loop, and only then the frozen spec. The levels below still describe how far each kind of work can go.*

Each level gives you something real to show. Each one also tells you whether the next level is worth it.

**1. Spec v1.** Block diagram, ISA, number formats, memory map. *Fully doable.*

**2. Golden model** (Python/NumPy, bit-exact with our number formats).
- Runs Qwen3.5-9B and measures quality for every format choice in `10_number_formats.md`.
- *Fully doable, and the most important early step:* several decisions are waiting on its quality numbers.

**3. Performance model.** Cycle-approximate, with a simple DRAM model, then DRAMsim3 or Ramulator 2.
- Gives real tok/s and utilization before any RTL.
- *Fully doable.*

**4. RTL of each block, with unit tests** (cocotb or SystemVerilog testbenches). *Fully doable.*

**5. Whole chip in Verilator, running a small model** (the tiny test model in `03_model.md`). *Fully doable.*

**6. Whole chip running the real 9B model in simulation.**
- About 33M cycles per token at 1 GHz with today's numbers.
- At 10,000 to 100,000 simulated cycles per second, that's 5 minutes to 1 hour per token. We'll know the real speed once the RTL exists.
- *Doable for a few tokens.* Those few tokens, matching the golden model bit for bit, are the headline result.

**7. FPGA prototype** (optional).
- Cheap FPGA boards have DDR4, not GDDR7. A board with 25 GB/s would run the 9B model at about 5 tok/s: 25 ÷ 4.6 GB per token.
- It's slow, but it's **real hardware generating real tokens,** and it's far faster than simulation for testing the ISA and the RL compiler.
- Needs a board with at least 8 GB of DDR, a few hundred to a few thousand dollars (to verify).

**8. Synthesis and layout on an open process** (Yosys and OpenROAD/OpenLane on SKY130, GF180 or IHP SG13G2, 130 to 180 nm).
- Real timing, area and power numbers for our blocks: the matrix lane, the vector unit, the DeltaNet unit.
- We scale them to N6 with published density ratios. That scaling is approximate, and we'll say so.
- *Fully doable for blocks.* The whole 4,096-MAC chip at 130 nm would be very large, so we might lay out a scaled-down version (say 1/16th of the lanes).
- SRAM comes from OpenRAM or the PDK's macros.

**9. Sign-off checks on the open layout.**
- DRC (layout rules) and LVS (layout matches schematic) with Magic, Netgen or KLayout. Static timing with OpenSTA.
- These are the same checks a real tapeout passes, on a real process, with free tools. *Fully doable.*

**10. A small real tapeout.**
- **Tiny Tapeout:** a shared-chip service with tiles of a fraction of a mm², roughly low hundreds of dollars per tile (to verify). Enough for one matrix lane plus a few PEs, or a small dot-product engine. You get a real chip back after several months.
- **Multi-project wafer shuttles:** wafer.space (GF180), IHP's research shuttles, ChipFoundry. These give a few mm² to about 20 mm² for thousands to tens of thousands of dollars (to verify). Enough for a scaled-down version of the whole digital core with a slow memory interface.
- *Doable if we want to spend the money.* Which one depends on budget.

**11. The real chip on a commercial node.** This is where we stop:
- **PDK access** for TSMC, Samsung or Intel needs an NDA and a business relationship.
- **GDDR7 PHY and controller, PCIe, PLLs:** licensed IP, typically hundreds of thousands to millions of dollars (to verify).
- **Sign-off tools:** commercial Synopsys or Cadence licences.
- **Mask set:** millions of dollars at N6 (to source; see `08_post_design_levers.md`).
- **Then packaging, board design, bring-up and manufacturing test.**

Our contribution at this level is a **paper design:** the RTL, ready for a commercial flow. Area and cost come from our estimates, and the analog blocks are behavioral models.

## Where we should actually aim

- **Must reach (the main claim):** levels 1 to 6. A public ISA plus RTL that runs Qwen3.5-9B bit-exact in simulation, with a performance model saying how fast it would be on a real card.
- **Should reach (credibility):** levels 8 and 9 for the main blocks. Real layout numbers on a real process beat "we think it's 80 mm²".
- **Nice to have:** level 7 (an FPGA running real tokens) and level 10 (a Tiny Tapeout tile of one matrix lane).
- **Out of reach without funding:** level 11. We say so plainly, and we publish what a funded team would need.

## What we model instead of building

- **GDDR7 PHY and DRAM:** a timing model of the memory chips (banks, refresh, read and write turnarounds), driven by our controller RTL.
- **PCIe:** a simple host model that pushes commands and data into our host-interface registers.
- **PLLs and clocks:** ideal clocks in simulation. Real clocks matter only for the timing numbers at level 8.
- **Pads, ESD and power delivery:** left out of simulation. Described in `09_chip_infrastructure.md`.

## The RL compiler milestone fits anywhere after level 5

The RL environment needs a way to run a program and get a score: cycles, utilization, correctness. That's either the performance model (fast) or the RTL in Verilator (exact). An FPGA (level 7) would make it much faster. So the RL work doesn't have to wait for layout or tapeout.

## Where the RTL work runs: laptop, CI and cloud (added 2026-11-11)

All our tools (Verilator, cocotb, Yosys, OpenROAD) are open source, so there are no licence limits and any machine can run them. The plan:

1. **Laptop:** write the code, run quick unit tests, look at waveforms.
2. **CI (GitHub Actions):** on every change, the fast tests run automatically (a few minutes) and changes that break them are blocked.
3. **A cloud CPU (Beam, which we already use, or similar), nightly or on demand:**
   - the full test set in parallel: 64 cores run about 64 tests at once
   - the full 9B-token simulation (about 33 M cycles per token)
   - synthesis and layout on the open process
   - comparison against the golden model
4. **Later, optionally, a cloud FPGA** (for example AWS F2). It runs the RTL as hardware, thousands of times faster than simulation, which the RL compiler work needs.

Notes:
- **A faster CPU helps compile time and single simulations, but only up to a point.** Verilator's speed-up from threads flattens out around 8 to 16 threads. The big win from the cloud is running many tests in parallel.
- **Use one container image everywhere,** so the laptop, CI and cloud run identical tool versions.
- **Only record waveforms when debugging.** Full-chip waveform files are huge.
- **Make sure cloud jobs shut down when they finish.**
