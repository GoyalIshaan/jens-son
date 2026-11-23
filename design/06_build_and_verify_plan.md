# 06. How we build and verify it

*Status: proposed 2026-11-10. Tools still to be confirmed by trying them.*

## The layers of models

*See also `15_golden_model_and_quantization.md` for a worked explanation of the golden model, and `16_design_flow_and_verification.md` for the order of work and the added verification safeguards.*

We build several models of the chip, each checking the next.

1. **The budget script** (`calc/qwen35_budget.py`). Arithmetic only: bytes, operations, bandwidth. Answers "is this design in the right ballpark?" in a second.
2. **The golden model.** A software model that does the exact integer arithmetic the chip will do: int4 weights, int8 activations, our rounding rules, our lookup tables for nonlinear functions. Checked against the real Qwen3.5 in floating point, using perplexity and a few benchmark questions. This tells us whether our number formats are good enough **before** we write any RTL.
3. **The performance model.** A fast, cycle-approximate simulator of the chip's blocks, queues and DRAM. It runs the full 9B schedule in seconds or minutes and predicts tokens per second. It is also the fast simulator the RL step needs.
4. **The RTL.** SystemVerilog, simulated in Verilator. Its outputs must match the golden model bit for bit, and its cycle counts must match the performance model closely.

## How each RTL block is tested

- **Unit tests per block:** random inputs, compared against the golden model. Include edge cases: maximum and minimum values, zero sizes, and back-to-back operations of different sizes. (The old `tile_engine` has an unconfirmed risk here: it clears accumulators globally at the first beat, which assumes the pipeline holds zeros between tiles.)
- **Layer tests:** one whole DeltaNet layer and one whole full-attention layer on the small model, compared against the golden model.
- **End-to-end test:** the small model generates tokens from a prompt, and the RTL's tokens must equal the golden model's tokens exactly.

## Simulation speed, and why we use a small model

- A full 9B token is about 33M cycles (budget script).
- Verilator might simulate a design this size at somewhere between 1,000 and 100,000 cycles per second, depending on how big the RTL is. That's between 6 minutes and 9 hours per token. We will measure the real rate.
- So day-to-day testing uses a small model of the same architecture (`03_model.md`), and the full-size design is checked with a few tokens plus the performance model.

## The DRAM model

The RTL needs something on the other side of the memory controller. Options:
- **A simple model we write:** fixed latency plus a bandwidth limit. Fast and easy, but it misses row misses and refresh.
- **DRAMsim3 or Ramulator 2:** detailed, cycle-level DRAM simulators that model LPDDR timing. More realistic, slower, and they need a C++ bridge into Verilator.

Plan: start with the simple model for correctness, then switch to a detailed one for performance numbers.

## Toward tapeout

What's realistic with open tools:
- **Synthesis:** Yosys turns RTL into gates.
- **Place and route:** OpenROAD (via OpenLane 2 or ORFS) on an open process kit such as SkyWater 130 nm or GF180MCU.
- **Results:** area, maximum clock frequency and power estimates for one tile or one block (for example the matrix unit or a vector lane).

What's not realistic, and the blog will say so plainly:
- **A full chip on a modern node.** Our performance numbers assume a modern process (5 nm class) and an LPDDR5X PHY. Open process kits are far older, and LPDDR5X PHYs are commercial IP.
- **So we'll report two kinds of numbers:** measured results on the open node for real blocks, and clearly labelled estimates for what the same design would do on a modern node.

## The RL compiler step

- **Environment:** the performance model plus the golden model. A program is rewarded only if its outputs are correct, then more for fewer cycles.
- **Policy:** Qwen3.5-9B itself, fine-tuned to write programs for our instruction set. The model learns to write code for the chip built to run it.
- **Baseline:** our hand-written programs, plus a simple scheduler. RL has to beat these.
- **To decide later:** which RL method, how much compute it needs, and whether to start with a smaller model.

## Open items

- Install and try Verilator plus a DRAM simulator. Measure simulation speed.
- Pick the golden model's language. Probably Python with NumPy for speed of writing, and maybe C later for speed of running.
- Check the Qwen3.5-0.8B config.
