# 01. The goal

*Started 2026-11-09. Last updated 2026-11-10.*

## In one sentence

Design an inference chip built for exactly one recent open model, Qwen3.5-9B, prove in simulation that it runs the model correctly end to end, take it as far toward tapeout as open tools allow, publish all of it, and then train the same model with RL to write fast low-level code for the chip.

## Why one model

Most AI chips are general: they must run any model anyone might write, so they pay for flexibility everywhere. A chip for one model can make choices a general chip can't.

- **The shapes are known.** Every matrix size, every layer type and the exact order of operations are fixed before the chip is designed. Buffers can be sized exactly, and the schedule can be worked out ahead of time.
- **The weights are known.** We know every weight the chip will ever read, so we can lay them out in DRAM in exactly the order they'll be read. That keeps the memory running near its peak speed (see `04_memory.md`).
- **Unused features cost nothing.** If the model never uses some operation, the chip doesn't need hardware for it.

The cost is that the chip gets worse, or useless, if the model changes. That's an acceptable trade for a project whose point is to show how far specialization can go. There is also an industry trend here: companies like Etched and Taalas have bet on chips specialized for one architecture or even one model. We should check and cite their public claims before the blog relies on them.

## Why this matters for a portfolio

- **Recent:** the model is from 2026, not a 2019 GPT-2.
- **Real:** it is a model people actually use, with real quirks: a hybrid of linear and full attention, a 248K vocabulary, and a multi-token prediction head.
- **End to end:** the project crosses three layers that rarely appear together: model analysis, hardware design and verification, and compiler/RL work.

## How we work: the design loop

1. **We agree the spec together.** The user makes the calls. Claude lays out the options, does the arithmetic and gives a recommendation.
2. **Claude writes the RTL and the tests.** The RTL is SystemVerilog. The reference model is written in Python or JavaScript, and the RTL output is compared against it bit for bit.
3. **Claude reports back** with what passed, what failed, the measured cycle counts and what it found surprising.
4. **We revise the spec** and go again. Each turn of the loop gets a short entry in `DECISIONS.md`, so the blog can show how the design evolved.

This replaces the old plan's rule that the user writes all the code. The learning phase is over, and the goal now is the best chip as fast as possible.

## Milestones, in order

1. **Spec v1:** workload, model, memory system, precision, top-level block diagram and the performance target. The "done" test: every block has an interface and a size, and the numbers add up in the budget script.
2. **Golden model:** a bit-exact software model of the chip's arithmetic (int4/int8 with our exact rounding), checked against the real model in floating point. This is how we know whether quantization damages quality before any RTL exists.
3. **RTL for one layer** of a scaled-down model with the same architecture, passing against the golden model.
4. **Whole model in simulation:** a small model of the same family (likely Qwen3.5-0.8B, or our own scaled-down config) generating real tokens in Verilator, with a DRAM timing model attached.
5. **Full-size check:** the 9B configuration run for at least a few tokens, or validated by a cycle-level performance model where full simulation is too slow. A full token is about 33M cycles (see the budget script).
6. **Toward tapeout:** synthesis and place-and-route with open tools (Yosys and OpenROAD) on an open process kit (likely SkyWater 130 nm or GF180), at least for one tile or core. Report area, timing and power. A full chip with a DRAM interface is beyond what open tools and open PDKs can do; we'll say exactly where we stopped and why.
7. **Publish:** the code, the notes and the blog.
8. **RL for the compiler:** fine-tune Qwen3.5-9B with reinforcement learning to write the chip's low-level VLIW programs. The reward is "correct output, in fewer cycles", measured by our own simulator. The model writes code for the chip that runs it.

## What success looks like

- **Correctness:** the chip produces the same tokens as our golden model, bit for bit, and the golden model's answers stay close to the floating-point model's.
- **Performance:** cycle counts in simulation that match the budget script's predictions closely, or a clear explanation of the gap.
- **Honesty:** every claim in the blog traces back to a simulation, a calculation or a cited source.
