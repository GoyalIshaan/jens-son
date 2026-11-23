# Small GPT-2 INT8 Embedded Chip - Verilator Simulation Plan

**Goal**  
Embed a small GPT-2-style model directly in the RTL (weights + control logic hardcoded).  
Run real prefill + decode in Verilator to prove the chip logic is correct.  
Measure cycle-accurate performance (prefill cycles + per-token decode cycles).

Everything stays in simulation only — no synthesis or FPGA required for this phase.

## Recommended Starting Model Spec

- Layers: 2
- d_model: 64
- n_head: 2 (head_dim = 32)
- vocab_size: 256
- block_size (max context): 32 (start small)
- FFN inner size: 256 (4x)
- Precision: INT8 weights + activations, INT32 accumulation

This size is small enough to embed as on-chip ROMs/arrays while still being a real (tiny) GPT-2-like decoder-only transformer.

## High-Level Architecture

- Reuse your existing systolic fabric (pe + sys_array + tile_engine).
- One main `matmul_engine` that supports the required shapes via tiling.
- All weights stored inside the RTL (initialized arrays or `$readmemh` from exported .mem files).
- KV cache as on-chip arrays.
- Supporting fixed-point operators: requant, LayerNorm, softmax (small), GELU approx, vector add.
- Hardcoded controller / sequencer FSM inside the top-level module. It knows the exact model shape and walks through every operation (embed → layers → final head).
- Simple top-level interface for the testbench:
  - Load tokens
  - Run prefill
  - Step decode (autoregressive)
  - Get next token + performance counters
- Verilator C++ driver is thin — mostly just drives the chip and collects results.

The "ASIC" feeling comes from having the full model + control logic inside the RTL rather than orchestrating everything from software.

## Phased Small Plan

1. **Lock model + build golden reference**
   - Define exact config (use the spec above).
   - Write a Python reference model (float first).
   - Add an integer-only simulation path that matches the exact operations the RTL will perform (including requantization rules).
   - Export a fixed set of weights + scales in a format easy to embed in RTL.

2. **Make the matmul path solid**
   - Implement / finish `matmul_engine` on top of `tile_engine`.
   - Support K-dimension tiling + accumulation for full GEMM.
   - Handle the matrix sizes needed (e.g. 64×64, 64×256, seq×d_head, 1×d for decode).
   - Create Verilator tests that compare against the golden.
   - Experiment with different systolic array sizes (ROWS/COLS/INNER_K) and record cycle counts.

3. **Implement the supporting INT8 operators**
   - Requantization after matmul (scale + shift + clip back to int8).
   - LayerNorm (fixed-point).
   - Softmax for attention scores (over ≤32 positions).
   - GELU approximation.
   - Residual add / elementwise ops.

4. **KV cache and attention dataflow**
   - On-chip storage for K and V (per layer, per head).
   - Logic to write new keys/values.
   - Support running attention at the *current* sequence length.
   - Handle both prefill (growing prompt) and decode (M=1 queries + full cached K/V).
   - Causal masking.

5. **Hardcode the model + build the controller**
   - Create the top-level module that contains:
     - Embedded weights
     - KV cache arrays
     - Activation buffers
     - Matmul engine instance(s)
     - The operator blocks
   - Write a hardcoded FSM / step sequencer that executes the exact transformer forward pass for this model.
   - Implement embed + positional, full layer stack (pre-norm style), final norm + lm_head.
   - Add simple next-token logic (argmax is enough to start).

6. **End-to-end verification + performance measurement**
   - Verilator C++ testbench that:
     - Loads a short prompt
     - Runs prefill
     - Performs several decode steps
     - Compares outputs against the golden reference
   - Add performance counters (total cycles, prefill cycles, decode cycles per token).
   - Once correct, vary the systolic array geometry and re-measure to see performance impact.

## Suggested First Steps

Start with **step 1 + step 2**:
- Get the golden + int8 reference working in Python.
- Get `matmul_engine` reliably doing tiled matmuls of the needed sizes in Verilator and matching the golden.

Everything else builds on a trustworthy matrix multiply.

## Key Technical Challenges

- Numerical matching: requantization, fixed-point LayerNorm, softmax, and GELU must be defined identically in Python and RTL.
- Sequence length handling in attention (especially the transition between prefill and decode).
- Data movement and buffering around the systolic array.
- Keeping the hardcoded controller manageable while still supporting both prefill and decode.
- Tiling strategy and accumulation when the systolic array is smaller than the matrix dimensions.

## How to Collaborate

- I will **not** write the implementation code unless you explicitly ask.
- I can:
  - Review diffs or code you write
  - Help debug specific issues
  - Suggest interfaces or data layouts
  - Refine this plan
  - Generate test vectors or help with the Python golden model
  - Answer architecture questions

Just share what you're working on or what part you're stuck on.

---

**Current status (as of 2026-07-05):**  
Core systolic pieces (pe + sys_array + tile_engine) exist and the sys_array tests pass.  
`matmul_engine` is still a stub.  
This plan focuses on turning that foundation into a complete embedded small GPT-2 that runs correctly in Verilator.

Let's go one phase at a time. What's the first piece you want to tackle?