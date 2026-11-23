# 30. The chunked DeltaNet for prefill

*2026-11-21. The user asked to try it (OPEN_QUESTIONS). Code: `golden/dn_chunked.py` (the maths and the chip's operand formats), `Numerics(state="chunked_*")` in `golden/qwen35_golden.py`, experiment set 9 (`golden/beam_exp9.py`), and `build_prefill(dn="chunked")` in `isa/qwen_prefill_v1.py` with the `DNC` vector op in `isa/sim_v1.py`.*

## Why

- **The sequential DeltaNet** updates each head's 128 × 128 state once per token: 3 passes over it on the vector unit.
  - For one 128-token prefill chunk that's 24 layers × 32 heads × 128 tokens × 3 passes, about **9.4 ms of a 26.8 ms chunk** (at 16K).
  - It only partly overlaps the matrix unit, so DRAM sat idle about 4 ms per chunk.
- **The chunked form** takes 64 tokens at once and turns most of the work into matrix products, which the matrix unit can do. Same maths, different order.

## The maths, with small numbers

Take one head and a chunk of just 2 tokens to see the idea.

- **Sequential:**
  - token 1: S₁ = d₁·S₀ + k₁·δ₁ᵀ, with δ₁ = β₁(v₁ − d₁·S₀ᵀk₁)
  - token 2: S₂ = d₂·S₁ + k₂·δ₂ᵀ, with δ₂ = β₂(v₂ − d₂·S₁ᵀk₂)
- **Substitute S₁ into δ₂:**
  - d₂·S₁ᵀk₂ = d₂d₁·S₀ᵀk₂ + d₂·(k₁·k₂)·δ₁
  - So δ₂ + β₂·d₂·(k₁·k₂)·δ₁ = β₂(v₂ − d₁d₂·S₀ᵀk₂)
- **That is a small triangular system** in the corrections δ:
  - Its right side needs only the chunk-start state S₀: one matrix product, S₀ᵀk, for all tokens at once.
  - Its left side needs only the keys' dot products k_s·k_t: another matrix product.
- **For a chunk of C = 64 tokens:**
  - G_t = g₁ + … + g_t, the cumulative log decay (always ≤ 0)
  - D[t, s] = exp(G_t − G_s) for s ≤ t, the decay from token s to token t (always ≤ 1)
  - A[t, s] = β_t (k_t·k_s) D[t, s] for s < t
  - r_t = β_t (v_t − exp(G_t)·S₀ᵀk_t)
  - **solve (I + A)·Δ = r** (forward substitution, 64 steps)
  - Δ_t is exactly the sequential rule's δ_t
  - outputs: o_t = exp(G_t)·S₀ᵀq_t + Σ_{s ≤ t} (q_t·k_s)·D[t, s]·Δ_s
  - new state: S₆₄ = exp(G₆₄)·S₀ + Σ_s exp(G₆₄ − G_s)·k_s·Δ_sᵀ
- **Where each piece runs:**
  - **Matrix unit:**
    - K·Kᵀ and Q·Kᵀ (64 × 64, 128 inputs)
    - S₀ᵀk and S₀ᵀq (64 × 128, 128 inputs)
    - P·Δ, where P = (Q·Kᵀ) ⊙ D (64 × 128, 64 inputs)
    - the state update K̃ᵀ·Δ (128 × 128, 64 inputs), accumulated into the fp32 state
  - **Vector unit, fp32:** G, D, A, r, the triangular solve, and the two sums.
  - **The state itself stays fp32.** Only the copy that goes into the matrix unit is rounded.

## What can go wrong (the user's question: what's the worst case?)

1. **The reordering itself goes unstable.** The triangular solve could amplify rounding error. In the worst case for a unit-triangular matrix, entries of the inverse can grow like 2^C.
   - **Measured:** no. All in fp32, the chunked form is as accurate as the sequential one (3 × 10⁻⁷ vs 2 × 10⁻⁷ relative error against float64).
     - Tested on random keys, keys clustered around 8 directions, and the worst case (the same key every token, β = 0.99).
     - |Δ| / |r| stays at or below 1.7.
   - **Why it stays tame:** A's entries are β × (cosine of two unit keys) × a decay ≤ 1. The decay makes far-apart tokens matter less, which is the opposite of the growth case.
2. **The int8 operands cost quality.** The matrix unit takes int8 on both sides, so the state's copy, the keys and the corrections all get rounded to 8 bits. **This is the real risk.**
   - **Measured on synthetic data:** 1.5–4% relative error in the outputs. It's stable: the same at token 2,000 as at token 100. The decay forgets old rounding errors, so they don't pile up.
   - **The biggest contributors** are the two products inside the chunk, Q·Kᵀ and P·Δ (about 1% each), then the state update (0.5–0.7%).
   - **Fallback, with no new datapath:** split each operand into a high int8 and the int8 remainder, and do 3 passes (hi·hi + hi·lo + lo·hi). The error drops 100× to 1–2 × 10⁻⁴. The matrix unit has the time (below).
3. **It's slower than it looks.** The products are small, 64 or 128 inputs, so most of each 256-input segment is idle. Measured in the timing simulator below, with that idle time counted.
4. **The worst realistic outcome:** experiment set 9 shows a quality loss that even the 3-pass version doesn't fix. Then prefill keeps the sequential DeltaNet, which already meets D-008 (32K in 6.85 s). The cost is this note, plus about one GPU hour. No hardware is committed yet.
5. **Prefill and decode no longer compute the DeltaNet in the same order.** The state after a prompt differs in its low bits from what token-by-token decode would give. That's fine: golden v2 defines both orders, and decode always continues from whatever state prefill left. It only has to be written down (order rule R8, below).

## Timing (ISA v1 simulator, 64 rows, 512 vector lanes, SiLU 64, fused softmax)

- **Program:**
  - For each value head and each 64-token half of the chunk:
    - a vector prep command (round q, k, the state copy and K̃ to int8 high and low parts; compute the decay matrix)
    - the matrix products K·Kᵀ, Q·Kᵀ, S₀ᵀk and S₀ᵀq
    - a vector solve command
    - the matrix products P·Δ and K̃ᵀ·Δ, the latter into the state
    - a vector output command
  - Every matrix command is 3 int8 passes.
  - Same SRAM (6.27 MB): the new buffers fit inside the DeltaNet phase of the shared region. 5,465 bundles (was 5,129), 40 semaphores.
- **Vector unit per chunk:** the DeltaNet goes from 9.4 ms (DN) to 2.2 ms (DNC). The matrix unit gains about 0.9 ms.
- **Chunk at 16K:** 26.8 → 21.0 ms. Matrix busy 16.5 ms, so prefill is now nearly matrix-bound.
- **Prompt:**
  - **32K: 6.85 → 5.37 s (−22%)**
  - **20K: 4.07 → 3.14 s**
  - That's about what 1,024 vector lanes would have given (5.45 s), at a fraction of the area.
- **The simulator fix this needed:** a matrix command's cycles are now beats = ceil(n_out / seg) × ceil(n_in × seg / 1,024) × hold. Before, a 64-input product was counted as if it filled the lanes. It changes nothing for the existing programs, whose n_in always fills whole beats.
- **A slip found on the way:** the simulator's default SiLU/sigmoid rate was still D-020's 16 per cycle. Last night's 6.85 s had passed 64 explicitly. The default is now 64 (D-023, decided). Decode re-checked: see below.

## What the hardware needs for this

- **A lane mask on the matrix unit:**
  - n_in shorter than one segment (64 or 128 of 256), with the lanes past n_in reading zero
  - The ISA's field check caught this: it had required whole beats.
- **A per-output fp32 scale vector in the matrix epilogue:**
  - The int8 weight-side operands (the state's columns, Δ's columns) carry one scale per output.
  - For K̃ᵀ·Δ, which accumulates straight into the state, the scale must be applied before the add.
  - The 64 epilogue multipliers already exist for the per-tensor gmax/127 (D-024). This makes their constant a vector read from SRAM.
- **A fused vector mode `DNC`** with three phases (prep, solve, output), like SMTILE. Its solve is a right-looking forward substitution: after each Δ_s, update every later r_t at once, so 64 dependent steps with the work spread across the lanes.

## Order rule R8 (golden v2), proposed

- **Cumulative decays:** G by sequential fp32 adds over the chunk.
- **Decays:** D[t, s] = exp(G_t − G_s) with the chip's eˣ.
- **Matrix products:** the matrix unit's contract (D-024), int8 operands as above, with 3 passes accumulated in the order hi·hi, hi·lo, lo·hi.
- **The solve:** for s = 1..64, Δ_s = r_s, then r_t ← fma(−A[t, s], Δ_s, r_t) for every t > s.
- **Outputs and state:** o = fma(γ, S₀ᵀq, PΔ); S = fma(γ₆₄, S₀, K̃ᵀΔ).
- The golden model uses a library triangular solve for speed. Its bits will differ from R8 in the last place, which is fine for quality but must be replaced by the R8 loop before the test vectors are made.

## Experiment set 9: quality on real text

*2026-11-21, RTX 4090, 38 minutes, `golden/beam_exp9.py`, `golden/results/exp9_chunked_dn.json`. Every prompt token runs the chunked form, in 64-token chunks. Paired KL against the sequential DeltaNet over 8,192 tokens of prose and of code, and 32,768 of prose. Two smoke runs failed first: the fp16-scale overflow guess (replaced by fp32 scales anyway), then the real cause, the activation-rounding edge case above.*

- **With bf16 weights (only the DeltaNet order changes):**
  - **all fp32:** KL below 10⁻⁶ on both sets. The reordering is free, as on synthetic data.
  - **int8, one pass:**
    - prose 0.0057, code 0.0075, 32K 0.0037
    - top-1 agreement 98.6%
    - **That's about a tenth of what the 4-bit weights cost.** Not negligible.
  - **int8, 3-pass split:** prose 0.000001, code 0.000004, 32K 0.000004. Nothing.
- **On top of the chip's full numerics** (GPTQ weights, D-019 activations, int8 KV, D-021, plus all of experiment set 8's chip-exact pieces; baseline prose 0.0560, code 0.0564, 32K 0.0448):
  - **one pass:**
    - code **+0.0045 ± 0.0010**, a real 8% increase
    - prose −0.0030 ± 0.0026 (noise)
    - 32K 0.0505
  - **3-pass split:**
    - prose +0.0019 ± 0.0024 and code −0.0003 ± 0.0005, both within noise
    - 32K 0.0475, also within noise (stderr about 0.008)
- **Error doesn't grow with distance** in either version. The 32K buckets follow the baseline's shape.
- **Conclusion:**
  - **The chunked DeltaNet with 3-pass split operands keeps the chip's quality** and makes a 32K prompt 22% faster (5.37 s).
  - **One pass is measurably worse on code.** It would save only the extra matrix passes, about 0.6 ms per chunk, so it isn't worth it.
  - **Proposed as D-025; decided by the user 2026-11-21 in its lite form (part 2).**

## Part 2: three versions compared across the whole chip (user request, 2026-11-21)

*The user asked: run the current setup against D-025, measure the effect on the matrix calculations, attention, the FFN and everything else, and keep D-025 only if that effect is small. Weigh the system design, the complexity and standard chip-design practice too. Code: `build_prefill(dn="lite")`, `chunked_lite` in `golden/dn_chunked.py`, the layer probe in `qwen35_golden._probe`, experiment set 10 (`golden/beam_exp10.py`, `golden/results/exp10_dn_layers.json`), `rtl/mxu/results/synth_asap7_reducer_mask.json`.*

### The three versions

- **Sequential (current):** one DN command per head per 128 tokens. That command exists anyway, because decode and the speculative step use it.
- **Chunked with new hardware (D-025 as first proposed):** needs three additions:
  - a fused 3-phase `DNC` vector mode
  - a lane mask in the matrix unit for products shorter than one segment
  - a per-output scale vector in the matrix unit's output stage
- **Chunked "lite" (new):** the same maths from existing commands only.
  - **No lane mask:** operands are zero-padded to whole 256-input segments in SRAM. A zero group also has the smallest exponent, so it can't disturb the beat's emax.
  - **No scale vector:** the per-output scales are fp16 (the QKV command's format), and the vector unit applies them after each product. The low weight part's 2⁻⁸ rides in the existing per-command tscale.
  - The decay sums are a 6-step scan of ADD commands. The triangular solve is a hardware loop of 63 FMA commands.
  - About 95 vector and 12 matrix commands per head per 64 tokens, instead of 3 + 4.

### 1. Time, part by part (32K prompt; chunk at 16K)

- **Attention layers:** mixer 4.75 ms and FFN 2.39 ms per chunk in all three versions. **No change.**
- **The DeltaNet layers' FFN:** 7.18 ms in all three. **No change.**
- **Matrix unit:** attention 2.65 ms and FFN 9.44 ms busy in all three. The chunked versions add 0.9 ms of DeltaNet products.
- **Only the DeltaNet mixer changes:**
  - sequential 12.65 ms
  - new hardware 6.87 ms
  - lite 8.01 ms
- **Whole prompt:**
  - **32K:** sequential 6.85 s, new hardware 5.37 s, **lite 5.66 s (−17%)**
  - **20K:** 4.07 s, 3.14 s, 3.32 s
- **Decode and the speculative step don't change.** Their program doesn't use any of this. In the lite version the matrix unit and the vector unit are also physically unchanged (91.7 and 253.0 tok/s at 32K).

### 2. The numbers inside the model, layer by layer (experiment set 10)

*2,048 tokens each of prose and code, the chip's full numerics. Each version is compared with the sequential one at every layer.*

- **How to read it:** changing *anything* sets a floor. The yardstick is the chip's own decode order of the sequential rule, a pure fp32 reordering known to be harmless (note 18). It already changes 54–62% of the int8 values entering later matrices and moves the residual stream by 4.5–4.8%.
  - Once one rounding decision flips, the next layers round differently, and two runs settle at about the int8 rounding noise apart. Neither is wrong; both are as far from the true value.
  - So the question is whether a version stays at that floor or goes past it.
- **Prose** (mean over layers; the residual at the last layer):
  - **yardstick:**
    - DeltaNet output 5.0%, attention output 6.4%, FFN output 6.1%, residual 4.5%
    - int8 values changed: attention inputs 54%, FFN inputs 58%
  - **chunked, one int8 pass:** 8.1%, 9.9%, 9.7%, 7.0%; inputs changed 66%, 71%. **Past the floor.**
  - **chunked with new hardware:** 4.4%, 5.3%, 5.3%, 3.9%; 57%, 61%. At the floor.
  - **chunked lite:** 5.4%, 6.9%, 6.5%, 4.8%; 57%, 61%. At the floor.
- **Code:**
  - yardstick: 4.6%, 4.7%, 5.0%, 4.8%; 62%, 63%
  - one int8 pass: 7.7%, 7.0%, 7.9%, 7.7%; 76%, 79%. Past the floor.
  - new hardware: 4.9%, 5.0%, 5.4%, 5.1%; 66%, 67%
  - lite: 4.9%, 5.2%, 5.5%, 5.2%; 66%, 67%
- **What this says for the matrix calculations, attention and the FFN:**
  - The 3-pass versions disturb the inputs of every later matrix exactly as much as a harmless reordering does.
  - One int8 pass disturbs them about 1.6 times more.
  - Nothing grows from layer to layer: the residual's worst layer is within 1.1× of its last.
- **Quality of lite** (KL, paired against sequential):
  - prose +0.0020 ± 0.0019, code +0.0001 ± 0.0005
  - 32K 0.0461 vs 0.0448 (stderr 0.008), top-1 93.3% vs 93.3%
  - **Within noise,** like the hardware version (experiment set 9).

### 3. What each version costs the hardware and the system

- **New-hardware version:**
  - **Matrix-unit lane mask, synthesized on ASAP7 (beat reducer):**
    - area +1.6% (area-mapped) / +2.7% (delay-mapped)
    - critical path +2.6% / +1.4%, both measured within one run
    - About 0.004 mm² for 64 rows, but it sits on the reducer path that every decode cycle uses.
  - **The per-output scale vector:** a 64-entry fp32 register and its load path, in the matrix unit's output stage.
  - **The DNC sequencer:** a new fused control mode with its own tests.
- **Lite version:**
  - **No hardware change at all.**
  - **Program:** 5,129 → 6,497 bundles, 79% of the 8,192-bundle instruction memory.
  - **Descriptors:** 159 → 218 of 1,024.
  - **Semaphores:** 36 → 40 of 63.
  - **SRAM:** the same 6.27 MB.
  - **It needs all 4 hardware-loop levels inside a DeltaNet layer:** key heads, value heads, 64-token halves, solve steps.
    - So the outer loop over prompt chunks must be a counted backward branch (BNE), which the ISA already has.
    - The assembler needs labels for it (a small software task).
    - The timing runs unroll that loop, which costs nothing in time.
- **Standard practice points the same way:**
  - **The matrix unit is the largest, most replicated and most timing-critical block, and decode, which sets the score, uses it every cycle.** Don't add logic to its datapath for a prefill-only gain.
  - **Keep algorithms in software when the speed cost is small.** The lite DeltaNet can be fixed or changed after tape-out; a fused sequencer freezes it. Here software costs 0.29 s on a 32K prompt.
  - **Verification grows with every hardware mode.** Lite reuses commands that are already verified. Its new risk is in the program, which the functional simulator (step 6) checks against golden v2 like every other program.
  - **The score is decode tokens/s per dollar.** Prefill already met its target (D-008), so prefill gains are a bonus, not the goal. That's one more reason not to spend matrix-unit timing on them.

### Verdict

- **The effect on attention, the FFN, decode and the matrix unit is nil for the lite version:**
  - same timing outside the DeltaNet mixer
  - the same disturbance to later layers as a harmless reordering
  - quality within noise
  - no hardware change
- **So the chunked DeltaNet stays, in its lite form.** D-025 is revised to it.
- **The fused DNC mode, the lane mask and the per-output scale vector are dropped.** They would buy 0.29 s on a 32K prompt for a touch on the matrix datapath and one more mode to verify.
- **One int8 pass is ruled out:** past the floor inside the model, and worse on code (experiment set 9).

