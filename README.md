# A chip for Qwen3.5-9B

An inference chip specialized for one ~9B open model, Qwen3.5-9B, taken from spec to RTL to simulation and as close to tapeout as open tools allow.

The RTL and the Python simulators are written by Claude in a design loop: each step is planned in a numbered design note, built, checked against a bit-exact golden model, measured, and the result recorded back into the notes and the decision log.

- Start with `design/README.md` for the reading order of the design notes.
- `design/DECISIONS.md` is the running decision log (decided, proposed, open).

## Repository layout

- `design/`: the numbered design notes (`01_goal.md` onward), the decision log, open questions, `spec/` (ISA v1 reference) and `calc/` (budget, performance and area/power calculators).
- `isa/`: ISA v1 assembler and encoder, timing simulator, functional simulator, the Qwen decode and prefill program generators, and saved runs in `isa/results/`.
- `golden/`: the golden model of Qwen3.5 with pluggable number formats, the bit-exact chip arithmetic references, test-vector generation, and the Beam GPU experiments; results in `golden/results/`.
- `rtl/mxu/`: the matrix-unit group and beat reducer (Verilog), their Python references, testbenches and yosys synthesis script.
- `rtl/vec/`: the vector unit (FP32 FMA, SFU, reductions, quantization, the command controller) with testbenches and `run_tests.sh`.
- `rtl/chip/`: the whole-chip RTL (top, core, control, DMA, SRAM, MXU, weight FIFO) and `run_chip.py`, which runs a program on the RTL and on the functional simulator and compares every byte.
- `rtl/synth_timing.py`: yosys + ASAP7 timing for the main blocks; `rtl/third_party/` holds the Liberty files it needs (not in git, see below).
- `model_map/`: builds a single-page visual map of the model and the chip (`index.html`).
- `blog/`: drafts of the public write-up.
- `tests/`, `pe.sv`, `sys_array.sv`, `tile_engine.sv`, `matmul_engine.sv`: early RTL from the learning phase (an int8 systolic array).
- `claude_research/`: background research PDFs (kept out of git for now; see "What is not in git").
- `archive/learning-phase/`: earlier learning material, including the old plan, the deep-dive site and the quiz.

## Dependencies

- Python 3.11 or newer. Install the Python packages with `pip install -r requirements.txt`.
- Icarus Verilog 12 or newer (`iverilog`, `vvp`), the default simulator for every testbench. Tested with 13.0.
- Verilator 5 (optional), a faster simulator for the large tests. Select it with `SIM=verilator`. Tested with 5.048.
- Yosys (optional), for the synthesis and timing scripts. Tested with 0.63.
- The Qwen3.5-0.8B checkpoint from Hugging Face, for the functional-simulator runs on a real model. The scripts read it from `~/.cache/huggingface/hub`, so fetch it once with `hf download Qwen/Qwen3.5-0.8B`.
- A Beam account and the `beam-client` CLI, only for the GPU experiments in `golden/beam_*.py`.

On macOS: `brew install icarus-verilog verilator yosys`.

## How to run things

Design calculators (no dependencies beyond Python):

- `python3 design/calc/qwen35_budget.py` reproduces the numbers in the design notes.
- `python3 design/calc/perf_model.py` runs the cycle-approximate performance model.
- `python3 design/calc/area_power.py` prints the area and power estimate.

ISA and simulators:

- `python3 isa/run_v1.py` builds the Qwen3.5-9B program in ISA v1, checks it and times it on the timing simulator.
- `python3 isa/run_fsim.py [n_tokens]` runs the real decode program on Qwen3.5-0.8B in the functional simulator and checks it against the golden model. The first run quantizes the weights and caches them in `golden/results/q08b_chip_weights.pt`.
- `python3 isa/run_fsim_prefill.py [--chunks 2] [--layers 24] [--decode 6]` does the same for the prefill program.

Golden model:

- `python3 golden/make_vectors.py` writes the golden test vectors to `golden/vectors/`.
- `golden/bitexact.py` and `golden/bitexact_vec.py` are the bit-exact references for the chip's arithmetic.
- The GPU experiments run on Beam, from inside `golden/`: `$BEAM beam_app.py validate` (also `quality`, `long`), where `BEAM` points at the beam-client Python. Each run saves a JSON to `golden/results/`; `design/15_measurements.md` explains the numbers.

RTL simulation:

- `sh rtl/vec/run_tests.sh` regenerates the vector-unit tables and runs every vector-unit testbench with Icarus.
- `python3 rtl/vec/test_vec_unit.py` runs the randomized vector-unit tests (add `SIM=verilator` for speed).
- `python3 rtl/chip/run_chip.py smoke` runs a hand-written program touching every unit on the whole-chip RTL and compares it with the functional simulator.
- `python3 rtl/chip/run_chip.py mmfuzz [seed]` runs random matrix-unit command shapes.
- `python3 rtl/chip/run_chip.py tiny [tokens]` runs the real decode program on a 4-layer Qwen3.5-shaped model.
- `SIM=verilator python3 rtl/chip/run_chip.py ...` uses Verilator (`rtl/chip/build_verilator.sh`) instead of Icarus.
- For the matrix unit, generate vectors with `python3 rtl/mxu/ref_group.py R MODE N OUTDIR` or `python3 rtl/mxu/ref_reducer.py add|red N OUTDIR`, then run `tb_group.v` or `tb_reducer.v` with `iverilog -g2012`.

Synthesis and timing (needs yosys and the Liberty files):

- `python3 rtl/synth_timing.py` for the main vector and chip blocks, `python3 rtl/mxu/synth.py` for the matrix unit, `python3 rtl/vec/synth_vec.py` for the vector unit.

## What is not in git

Generated files are left out by `.gitignore` and can be rebuilt with the commands above:

- Simulator outputs: `*.vvp`, `*.vcd`, Verilator build folders, and the per-run folders under `rtl/chip/results/` (each holds ~60 MB DRAM and SRAM dumps).
- Model weights and caches: `*.pt` and other checkpoint formats.
- Golden test vectors in `golden/vectors/` (`python3 golden/make_vectors.py`).
- Standard-cell Liberty files in `rtl/third_party/`. Download them from the ASAP7 repository (`asap7sc7p5t_*_RVT_TT_nldm_*.lib` into `rtl/third_party/asap7/`) and the SkyWater sky130 PDK (`sky130_fd_sc_hd__tt_025C_1v80.lib` into `rtl/third_party/sky130hd/`).

Held back for now, and not regenerable:

- The research PDFs in `claude_research/` and the vendor papers in `archive/learning-phase/learning/research/pdf-source/`.
- `archive/learning-phase/learning/site/`, which is its own git repository.

## License

MIT. See `LICENSE`.
