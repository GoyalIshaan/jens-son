# 16. The design flow after the golden model, and how it compares to industry

*Written 2026-11-10 from a design discussion. Status: proposed (D-016). This corrects the order of the early levels in `14_how_far_we_can_go.md`, which listed the spec before the golden and performance models.*

## The order: decide, model, specify, then build

Each step answers a question that would be expensive to get wrong at the next one. Steps 1 to 4 are all software, and each is cheaper to change than the one after it. The aim is to find mistakes there, not in RTL.

**1. Use the golden model to settle the open decisions** (days, on Beam GPUs)
- Agree the quality pass mark first, then test the options against bf16:
  - weight formats: round-to-nearest vs GPTQ vs AWQ, int4 vs other 4-bit tables, FFN at 3 bits
  - int8 vs bf16 activations
  - int8 vs int4 KV cache at long context
- Measure the real MTP acceptance rate.
- **Done when** D-005 (formats) and D-013 (speculative decoding) rest on measured numbers.

**2. Architecture and performance model, in a loop**
- Start from the draft architecture we already have (notes 05, 11 and 13, D-015).
- Build a cycle-approximate simulator of the blocks and a DRAM timing model. It does no maths, only timing: when weights arrive, when each unit is busy, where it stalls.
- Questions it answers:
  - Do we really reach 92% of bandwidth?
  - Is 1,024 × 4 the right array shape?
  - Is 4 MB of SRAM enough?
  - Do we make the 4 s prefill floor?
- Fix the architecture where it falls short, rerun, repeat.
- **Done when** it predicts tok/s, utilization and prefill time for a design point that meets our targets.

**3. Freeze spec v1**
- Everything down to the bits:
  - block diagram
  - instruction set: every instruction, its encoding, what it does
  - weight layout in DRAM (the 18-byte blocks of 32 codes plus a scale)
  - memory map and register map
- **Done when** someone else could write the RTL from the document alone.

**4. Instruction-level simulator and the Qwen program**
- A simulator that executes our instructions one by one, with the golden model's arithmetic.
- Write all of Qwen3.5-9B as a program in our instruction set. Its output must match the golden model exactly.
- This proves the ISA can express the whole model. It's also the environment for the RL compiler work, which can start here.
- **Done when** the program generates the same tokens as the golden model.

**5. RTL, block by block, each checked against the golden model**
- Order:
  - matrix lanes plus the scale unit (the biggest block; the early systolic RTL is a starting point)
  - vector unit
  - DeltaNet unit
  - attention path
  - DMA and the memory controller, against a DRAM model
  - the VLIW control unit
- Each block gets golden-model inputs and expected outputs, plus random and extreme inputs (see below), and must match bit for bit.

**6. Integration, small to big**
- One layer, then the small test model end to end in Verilator, then a few tokens of the real 9B model.
- The 9B tokens matching the golden model exactly is the headline result.
- The performance model's predicted cycle counts get checked against the RTL's real ones. If they disagree, one of them is wrong, and we find out which.

**7. Physical design on an open process**
- Synthesis and layout of the main blocks with Yosys and OpenROAD.
- Real area, timing and power replace the "about 80 mm²" guess in the cost model.

**8. RL compiler and publishing**
- RL training uses the instruction simulator and performance model as its environment, with cycles and correctness as the reward.
- Publish the notes, code and results.

Steps 2 and 3 can start while step 1 runs. The performance model needs only the shapes and bandwidths, which we already have.

## "Doesn't the performance model need the design on paper first?"

Partly. "The design on paper" means two different things:
- **The architecture:** which blocks exist, how wide they are, how they connect, buffer sizes, rough latencies. For example: "1,024 lanes × 4 rows, weights arrive at 512 bytes per cycle, a 102 KB weight FIFO, the DeltaNet unit does one head at a time, DRAM latency about 100 ns."
- **The full specification:** every instruction's bits, the register map, the byte layout in DRAM, error handling.

The performance model needs the architecture, not the full spec. The architecture and the performance model are built **together, in a loop.** That loop is where most real design decisions get made. The spec then records where the loop ended.

An example of why the loop comes before the spec:
- Say the model shows the DeltaNet unit takes 3,000 cycles per layer while that layer's weights stream in 1,500 cycles. The matrix unit would sit idle half the time.
- In the performance model, adding a second DeltaNet unit or overlapping it with the next layer's weights is a one-line change.
- Found after the instruction encoding and RTL are written, it means redoing both.

## Is this the standard chip design process?

Yes. If anything, ours is a lighter version of what chip companies do.

### What industry does that we also do

- **A reference model** (often called a "C model"), compared automatically against the RTL in every test.
- **An architecture team with performance models,** exploring designs before RTL.
- **An instruction-set simulator** for any programmable chip.
- **Block-by-block RTL** with a testbench per block, then integration.

How seriously industry takes this: verification is commonly said to take more than half of a chip project's effort, and verification engineers often outnumber designers. A missed bug that forces a new set of masks costs millions and months.

### Where we're less careful than industry

- **No coverage-driven random testing** (UVM-style), which measures which corner cases the tests actually reach.
- **No formal verification** across the whole design (mathematically proving a block can't misbehave).
- **No emulation hardware** to run billions of cycles.
- **No gate-level simulation, no full sign-off on a commercial node, no test insertion, no testing of real silicon.**

### Where we're more ambitious than usual

- **A public instruction set plus an RL-trained compiler.** Few chip projects publish their ISA, and we don't know of any that train a model to program it.
- **Running a full 9B model bit-exact in RTL simulation** on a laptop-and-cloud budget, without emulators.
- **Making quantization decisions inside the hardware project.** Usually an ML team does that and hands the hardware team a format.

## Two safeguards we're adding because of that comparison

**1. Random and extreme inputs for every block, not only real model data.** Real Qwen data never reaches the worst cases. For example, our 20-bit group sum can reach 516,128 in the worst case, but real weights and activations never get close. A test using only real data would never catch an adder that overflows at 300,000. So every block's testbench also runs:
- all maximum values and all minimum values
- alternating signs
- zero-length and single-element inputs where the block allows them
- long random sequences checked against the golden model's function for that block

**2. Formal checks on the small, tricky blocks,** using SymbiYosys (free and open source). Candidates:
- FIFOs
- the clock-domain-crossing FIFOs (Gray-code pointers, see `09_chip_infrastructure.md`)
- the arbiters and the instruction-issue logic

These blocks are small enough to prove correct, and they're where hard-to-reproduce bugs usually hide.

## The point in one line

The golden model proves the chip is right on the inputs we tried. Random and extreme inputs, plus formal checks, cover the inputs we didn't think to try.
