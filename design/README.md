# Design notes: a chip for one model

These notes record how we are designing an inference chip for one open model, Qwen3.5-9B. They are the raw material for a long public blog post. So every file says what we decided, what else we considered, why we chose what we did, and which numbers the choice rests on.

## How to read these notes

- Start with `01_goal.md`, then read the rest in number order. Each file builds on the one before.
- `DECISIONS.md` is the running log. Every decision gets an entry with a status: decided, proposed (waiting for a yes) or open.
- `OPEN_QUESTIONS.md` lists what we still don't know and how we plan to find out.
- `calc/qwen35_budget.py` produces every number in these notes. Run `python3 design/calc/qwen35_budget.py` to check them. If a note and the script disagree, the script is right and the note needs fixing.

## The files

- `01_goal.md`: what we are building, why, and the milestones from spec to tapeout to the RL compiler.
- `02_workload.md`: who the chip serves (one user, many users, a datacenter) and why that choice drives everything else.
- `03_model.md`: why Qwen3.5-9B, what else we looked at, and what the model's shape means for hardware.
- `04_memory.md`: 3D SRAM vs LPDDR vs GDDR vs HBM, and how a DRAM fetch actually works.
- `05_architecture_sketch.md`: a first sketch of the blocks, the data passed between them, and how VLIW-style control keeps them busy.
- `06_build_and_verify_plan.md`: how we write, test and simulate the RTL, and how far toward tapeout we can get.
- `07_competitors_and_goal.md`: the cheapest hardware that runs Qwen3.5-9B today, its tokens per dollar, and our minimum goal.
- `08_post_design_levers.md`: node, die size, yield, NRE, packaging and how each changes the cost.
- `09_chip_infrastructure.md`: clocks, clock domains, on-chip data paths, power, reset and boot, the host link, debug and manufacturing test.
- `10_number_formats.md`: weight, activation, accumulator, state and KV formats, and what each costs.
- `11_memory_capacity_and_1m_context.md`: on-chip SRAM, card memory, and what the full 1M-token context costs.
- `12_speculative_decoding.md`: using Qwen3.5's built-in MTP drafter, and rolling back the DeltaNet state.
- `13_matrix_shapes.md`: every weight matrix, its share of the bytes, and a first mapping onto hardware.
- `14_how_far_we_can_go.md`: the levels from spec to tapeout, and where we have to stop.
- `15_measurements.md`: the measured evidence: golden model vs Hugging Face, every number format (GPTQ, table-entry precision), long-context KV and state, task benchmarks (GSM8K, HumanEval, MBPP), real GPU speed. Raw data in `golden/results/`.
- `15_golden_model_and_quantization.md`: what a golden model is, why activation quantization is cheap, and where 4-bit weights come from (no official int4 Qwen3.5-9B exists).
- `16_design_flow_and_verification.md`: the order of work after the golden model, how it compares to industry practice, and the extra safeguards we're adding.
- `17_numerics_questions_and_pushback.md`: questions raised about the measurements (plain vs smarter rounding, the 13% plateau, why not bf16, why tokens per dollar beats raw speed), with the answers and pushback.
- `18_deltanet_unit_and_memory_traffic.md`: a deep dive on the DeltaNet unit (one-pass trick, sizing), and memory-traffic ideas: dedicated lanes (rejected), on-chip state, write batching, a read-only weight region with bank partitioning.
- `19_spec_v1_plan.md`: the plan for spec v1, covering what it contains, what is settled, the 12 open items and how each closes, the order of work, and the freeze criteria.
- `20_weight_format_explained.md`: a blog-ready explanation of the weight format (D-005) from scratch, with a worked example, the three experiments behind it (KL, table precision, tasks with paired statistics) and why it is the best trade for this chip.
- `21_performance_model_v1.md`: the first cycle-approximate performance model (`calc/perf_model.py`): checks against the hand numbers, decode and speculation at 4K to 1M, sweeps of lanes, rows, clock, memory and buffers, prefill, capacity, and what it changes in the plan.
- `22_memory_choice_and_long_context_target.md`: cited GDDR7 and LPDDR5X prices, every candidate card on tok/s per $, the pick (GDDR7 in 2 GB chips, 8 and 16 GB cards), the new long-context target (score at 32K, 32K-token prompt in about 12 s), and the matrix unit it needs (1,024 × 32).
- `23_pcie_explainer.md`: PCIe from scratch (lanes, generations, slot power, M.2, OCuLink, USB4, CXL), what our card actually sends over it, and why Gen4 x4 on a physical x16 card.
- `24_matrix_unit_rtl_and_synthesis.md`: progression step 1. One matrix-unit group in RTL, bit-exact tests, synthesis on ASAP7 7 nm at 4–64 rows, the scale-step finding, and what it decides (1 GHz holds, 64 rows is cheap, test power-of-two activation scales).
- `25_cheaper_activation_scales.md`: experiment set 5. The activation scale's fp32 multiply is 37% of the matrix unit's row logic. A bf16-style scale is free in quality (−45% scale step); a power-of-two scale like MXINT8 (−84%, one pipeline stage) doubles activation-only error but is invisible with the chip's weights. Confirmed on GSM8K and HumanEval: no significant change (power of two +0.23 and −1.2 points). Recommended as D-019.
- `26_beat_reducer.md`: adding the 64 group results of a beat per row. With power-of-two scales it's an exact fixed-point sum (3,621 µm², 4.25 ns per row at 7 nm), 3.2× smaller and faster than a 63-adder fp32 tree. Includes the new rounding rule (16 fraction bits, round down) for the bit-exact contract.
- `27_budgets_sram_sfu_lm_head.md`: progression step 2. The SRAM budget from the real formats (about 6.5 MB, 128-token prefill chunks) and the dataflow choices that keep it small; the special-function unit's counts, width (128 exp per cycle) and measured precision (exp is the sensitive one); the streaming greedy LM head; and the int8-attention question (experiment set 7).
- `28_isa_v1_and_qwen_program.md`: progression step 3. ISA v1 (4-slot bundles of coarse commands, per-unit queues and semaphores, descriptors, a weight FIFO), the whole Qwen3.5-9B decode and speculative step written in it, and a timing simulator that runs it: within 1% of the performance model, after finding that the weight FIFO needs 192 KB and the vector unit 512 lanes.
- `29_golden_v2_part1.md`: progression step 4, part 1. The bit-exact matrix-unit contract (checked against the RTL reference; 14x more accurate than fp32 GEMM), the special-function unit as real table + Newton algorithms, chip-exact RoPE and tiled softmax, and experiment set 8 (their quality on real text).
- `30_chunked_deltanet_prefill.md`: the chunked delta rule for prefill. The maths with small numbers, what can go wrong, the timing (32K prompt 6.85 → 5.37 s) and experiment set 9 (3-pass split int8 operands keep quality; one pass doesn't, on code). Part 2: sequential vs fused vs no-new-hardware, part by part and layer by layer (experiment set 10); the no-new-hardware version is chosen (5.66 s, nothing else changes). D-025.
- `31_functional_simulator_first_run.md`: progression step 6 begins. The functional simulator (the timing simulator plus golden v2's exact values), the weight packer, and the first end-to-end run of the real ISA program on Qwen3.5-0.8B: it writes "Tokyo", 11 of 12 tokens match the golden model, every layer matches to fp32 rounding in isolation. Parts 2 and 3 cover two-chunk prefill, the KV-head instruction loop, speculative replay and randomized timing checks.
- `32_vector_unit_rtl.md`: progression step 6b. The vector unit's arithmetic in RTL: one fp32 FMA for every elementwise op, the int8 quantizer, the five special functions as FMA chains, the 512-lane reduction. All bit-exact with golden v2 (0 mismatches over about 70,000 vectors). ASAP7 area (about 0.8 mm² of arithmetic), the pipelined FMA, and why an 8-cycle FMA costs the speculative step 11% unless the DeltaNet unit interleaves 4 heads. Part 2 records the eight-stage FMA, quantizer mappings, pipelined special functions, decoded vector controller and verification scope. Part 3 measures four interleaved DeltaNet heads, the SRAM allocation and the checkpoint order used in speculation. Part 4: the review (every command was stop-and-wait), the fixes (a dependency-scheduled DeltaNet controller, streaming box and row commands at the D-020 unit counts), and where 9B performance stands at the RTL's real rates. Parts 5–7: the streaming softmax tile and its row overlap, row ops streamed across rows, four-context reductions; the 9B meets D-008 at the RTL's measured rates (32K prefill 11.4 s, 20K 6.5 s, decode 90.0 tok/s).
- `33_whole_chip_rtl.md`: progression step 6, second half. The rest of the chip in RTL (`rtl/chip/`): scalar core, queues and semaphores, DMA, weight FIFO, matrix unit controller, the binary descriptor format, the SRAM port model; whole-chip runs against the functional simulator byte for byte (a smoke program and the real decode program on a 4-layer Qwen3.5-shaped model); SRAM banking sized from measured traffic (D-028); Verilator working on this Mac; the prefill program too; RTL timing against the model, and the matrix unit rebuilt so loads, beats and epilogue overlap (D-029).
- `34_sram_proposal.md`: a proposal, not decided. Revisits D-028 now that packing runs the vector unit near its model rates: the arbiter livelock behind the prefill hang, what bandwidth and bank collisions cost 32K prefill (6.6 s with a free SRAM, 7.8 s with today's 64 plain banks), where the collisions come from, and the options (skew, layout, 128 banks, dual ports) with a recommendation.
- `REPORT_2026-11-22_overnight.md`: the overnight summary: what was built, decisions D-026 to D-029, numbers, what's left, and three decisions for the user.
- `spec/isa_v1_reference.md`: the ISA v1 reference (every slot, opcode, field and descriptor), for spec v1 (step 5).

Notes 07 to 16 are things we are still weighing: their leanings are logged as *proposed* in `DECISIONS.md` until we agree them.

## Rules we follow when writing these notes

- **Numbers come with their working.** Each figure either comes from the script or shows its arithmetic inline.
- **Facts about the outside world come with a source.** Unverified claims are marked as such.
- **Rejected options stay in.** The blog is more useful when it shows the roads we didn't take and why.
- **Decisions are dated.** If we change our mind, we add a new entry and keep the old one with a note.

## What happened to the old docs

The project started as a learning exercise: a tiny GPT-2-style model hardcoded into RTL, plus a deep-dive website on chip design. That material now lives in `archive/learning-phase/`. Nothing was deleted. The deep-dive site and quiz still run from there. The early RTL (`pe.sv`, `sys_array.sv`, `tile_engine.sv`) stays at the top level, because we may reuse the systolic array.
