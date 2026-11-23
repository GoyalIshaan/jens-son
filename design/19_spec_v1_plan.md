# 19. Plan for spec v1

*Written 2026-11-14. Status: plan, for review. It follows the flow in D-016: (1) golden model and quality decisions, (2) architecture and performance model in a loop, (3) freeze spec v1. Step 1 is now finished (`15_measurements.md`, experiment sets 1–4). This note says what spec v1 contains, what's already settled, what's still open and how each open item gets closed, and when we call it frozen.*

## What spec v1 is

Spec v1 is **the contract** that three things will be checked against:
- **the golden model.** It has to compute exactly what the spec says, bit for bit.
- **the instruction-level simulator** (D-016 step 4).
- **the RTL** (step 5).

If any two of them disagree, the spec says which one is wrong.

**What it contains:**
1. **Targets and scope:** model, workload, metric, context, card sizes.
2. **Numerics, defined bit-exactly:** every format, every rounding rule, and the order of every sum that matters.
3. **Memory formats:** how weights, scales, tables, KV, state and the conv history are laid out in DRAM.
4. **Blocks:** each block's job, its throughput per cycle, its buffer sizes and its interfaces.
5. **ISA v1:** the bundle format, the instructions, the registers, and the timing rules a compiler can rely on.
6. **The Qwen3.5-9B program,** in outline: prefill, decode, and the speculative draft-and-verify step.
7. **Performance:** predicted tok/s, prefill time and utilization from the performance model, with the assumptions behind each.
8. **Verification plan:** test vectors, bit-exact checks and formal checks.

**What it does not contain:**
- RTL, a floorplan, or vendor IP choices beyond interface widths
- prices (those stay in notes 07 and 08)

## Already settled (inputs to the spec)

**Model and workload:**
- Qwen3.5-9B, text only, batch 1, local use (D-002, D-007)
- context up to 1,010,000 tokens (D-011)
- programmable through a public VLIW ISA (D-009)

**Weights (D-005, decided 2026-11-12):**
- Each weight is a 4-bit code that looks up a learned 16-entry table of **int8** values, one table per matrix.
- Each group of 16 weights has an **E4M3** scale; each matrix has one fp32 scale. That's 4.5 bits per weight.
- Rounding is GPTQ, done offline.
- The embedding is int8 with a scale per 32.
- Evidence: GSM8K −1.9 points against bf16, no detectable change on HumanEval or MBPP. Plain int4 loses 7.5 points on GSM8K.

**Activations, KV cache and state:**
- Activations are int8 with a **power-of-two** scale per 32 (D-019).
- KV cache: int8 with one scale per 256-value head, plus int4 with a scale per 32 as an opt-in long-context mode.
- DeltaNet state: fp32.

**Accumulation:**
- An exact integer sum within each weight group.
- fp32 across groups.

**Speculative decoding (D-013):**
- Qwen's own MTP head as the drafter.
- The KV end pointer can move back; the DeltaNet state uses lazy commit.
- **Measured** (note 15 §15): with the chip format and a 32K draft vocabulary, drafts 1–3 are accepted 90.0%, 84.1% and 78.2% of the time, 3.26 tokens per step at k = 3. That's k = 3 at short and medium context, k = 1–2 at long context (D-013, note 21).

**Matrix unit (D-015):**
- About 1,024 weight lanes, each with 4 token rows.

**Memory sizes (D-012):**
- about 4 MB of on-chip SRAM
- an 8 GB base card, with a 16 or 24 GB long-context card

## What the measurements change in earlier decisions

**D-017 (lane arithmetic) has to move from groups of 32 to groups of 16:**
- **Adder tree:** 16 products per group. The worst case is 16 × 128 × 127 = 260,096, which fits a **19-bit** signed sum (limit 262,143).
- **Scale unit:** 1,024 lanes ÷ 16 = **64 group sums per token row per cycle**, so 256 per cycle with 4 rows. Each one is multiplied by the weight scale (E4M3 converted to fp32) times the activation scale, then added into an fp32 total.
- **Activation scales:** one activation scale (per 32) covers two weight groups, so the scale register file only needs to hold the current 32 activation scales.
- **Bytes per lane:** the weight block is 8 bytes of codes plus 1 E4M3 byte per 16 weights, 9 bytes in all. A full 1,024-lane beat is 576 bytes.
- **What the memory actually delivers:** at 512 GB/s and 92% utilization, about 471 bytes per cycle arrive at 1 GHz, which is about 837 weights. So the 1,024 lanes have about 20% headroom.

**The table:**
- 16 int8 entries, loaded per matrix.
- About 250 matrices per token × 16 bytes ≈ 4 KB per token, which is negligible.

**The KV scale:**
- One fp16 scale per 256 values instead of per 64.
- Scale storage drops 4×. Each token has 16,384 KV values, so its scales take 512 bytes per token with a scale per 64, and 128 bytes with one per head. At 1M context that's 0.51 GB down to 0.13 GB.

**The MTP head:**
- 243.3M matrix weights, which matches the size note 12 assumed.
- 137 MB at 4.5 bits.

## Open items spec v1 must close, and how

Each item gives the question, the options, how we decide, and who decides.

1. **Memory type (D-004). Decided 2026-11-18: GDDR7 in 2 GB chips, 512 GB/s (note 22).**
   - Options: 4 × GDDR7 x32 (512 GB/s, the current leaning) or LPDDR5X 256-bit (273 GB/s).
   - How: cited, dated prices per GB and per package, plus the performance model's tok/s for each.
   - **Decided by the user,** because it sets most of the unit cost (60–75%, D-010).
2. **Core clock: 800 MHz or 1 GHz.**
   - It sets how many weights arrive per cycle, so it sets the lane count.
   - How: the performance model, plus a rough timing check of the lane, the adder tree and the scale unit in open-PDK synthesis.
3. **Matrix unit size and the prefill floor (D-015). Target set 2026-11-18: 20K in about 7 s, 32K in about 12 s; 1,024 × 32 proposed (note 22).**
   - Problem: 4 s for a 2K prompt only works if each weight is held for 2 cycles across 8 tokens (3.97 s).
   - Options: loosen the floor, grow the unit about 25%, or raise the clock.
   - How: the performance model's prefill timeline. **Decided by the user.**
4. **How many speculative tokens to verify (k), and the token rows.**
   - How: the measured MTP acceptance (experiment set 4), combined with the performance model's real step cost. That cost includes the DeltaNet unit and attention for k + 1 tokens and the time spent drafting.
5. **DeltaNet unit width: 128, 256 or 512 fp32 lanes.**
   - Depends on item 4, because verifying k + 1 tokens multiplies the state work per load.
   - How: the performance model. The one-pass update order is checked in experiment set 4.
6. **On-chip SRAM: 4 MB, banking, and the conv history.**
   - The conv history is 2.4 MB if kept on chip; the other option is to stream it.
   - How: the performance model's buffer occupancy and DRAM read/write switching cost (note 18).
7. **Decode attention at long context.**
   - At 1M tokens with int8 KV, attention reads about 16.5 GB per token (16,384 bytes per past token), about 3.7× the weights. That's why decode drops to about 22 tok/s there (note 11). The question is how the attention unit streams that much data at full memory speed.
   - How: the performance model at 4K, 32K, 262K and 1M.
8. **LM head and sampling.**
   - Greedy and top-k over 248K logits computed as they stream, without storing them all.
   - The draft head uses a trimmed vocabulary list of the **top 32K tokens** (measured best: 277 tok/s at 4K k = 3, vs 227 with the full list).
   - How: write the program for it and count cycles and buffer space.
9. **Special functions** (exp, softplus, sigmoid, SiLU, rsqrt).
   - Which ones, at what accuracy, built how (table plus polynomial).
   - How: an accuracy target set from the golden model. Inject the planned approximation and require that KL stays within the measured noise.
10. **ISA v1.**
    - What to define: bundle slots, encoding, register files, DMA descriptors, loops, synchronization, and the "load table" step.
    - How: write the whole Qwen3.5-9B decode step in draft ISA. Count bundles and instruction bytes, and check that no slot is the bottleneck.
11. **Host interface. Decided 2026-11-18: PCIe Gen4 x4 on a physical x16 card (D-018, note 23).**
    - PCIe width and generation. The host only sends tokens and receives tokens and logits, but loading the 5 GB of weights at boot sets the startup time.
    - How: a short calculation, then **decided by the user.**
12. **Bit-exact numerics contract.**
    - Problem: the golden model currently imitates the chip's rounding in fp32 PyTorch, which isn't bit-exact.
    - Spec v1 must define the chip's arithmetic exactly:
      - the integer group sum
      - E4M3 decoding
      - how activations are rounded to int8 (round half to even, saturating)
      - the order of fp32 additions across groups and lanes
      - the special-function approximations
    - How: build **golden v2**, a chip-exact mode. Check that it gives the same quality as experiment set 3, within noise.

## Order of work

**Done before the progression below:**
- **Model-side evidence:** experiment sets 1–4, including MTP acceptance and the draft window (note 15). Done 2026-11-18.
- **Performance model v1:** `calc/perf_model.py`, note 21. It matches the hand numbers exactly. Done 2026-11-18.
- **Decisions on memory, metric and host link:** D-004, D-008, D-018 (notes 22–23). Done 2026-11-18.

**The progression (agreed with the user 2026-11-20). Each step starts only after the user has reviewed the previous one.**

1. **Build one matrix-unit group in RTL and measure it.** *Done 2026-11-20 (notes 24–26). Power-of-two activation scales tested and adopted (D-019); beat reducer built. Left for later: a power measurement (user: later).*
   - RTL for one group: 16 lanes × R token rows, with the table lookup, the int8 multipliers, the 16-input adder tree and the scale step. A testbench checks it bit-exact against a Python model.
   - Synthesize it with Yosys on the open SkyWater 130 nm library at R = 4, 8, 32 and 64.
   - Report area (in gate equivalents, then scaled to 6 nm-class), the critical path, and what the scale step costs.
   - **Decides:** 32 vs 64 rows (D-015), and the clock (open item 2).
2. **Budgets, mostly on paper** (open items 6, 8, 9). *Done 2026-11-20 (note 27), reviewed and approved by the user 2026-11-21. Decided: D-012 SRAM about 6.5 MB, D-020 special-function unit, D-021 attention on the int8 lanes (measured at 8K, 32K and with full chip numerics).*
   - the on-chip SRAM list for the chosen row count
   - the special-function unit's width and accuracy, with a golden-model test of the eˣ approximation
   - the LM head and on-chip sampling
   - ~~a GPU check of draft acceptance under sampling~~ not needed: the user runs greedy (2026-11-20)
3. **ISA v1 and the Qwen program** (open item 10). *Decode and speculative step done 2026-11-21 (note 28), approved by the user 2026-11-21: D-022 ISA v1, D-023 weight FIFO 192 KB and 512 vector lanes, D-024 golden v2 part 1. Prefill program done overnight (note 28 part 2): 32K prompt in 6.85 s within 6.27 MB SRAM.*
   - bundle format, registers, DMA, loops, sync, "load table"
   - the full Qwen3.5-9B decode and speculative step written in it, with counts of bundles, bytes and slot use
4. **Golden v2: bit-exact chip numerics** (open item 12). *Started overnight 2026-11-21 (note 29): matrix contract, special-function algorithms, test vectors in `golden/vectors/`; experiment set 8 done: no measurable quality change on the chip's numerics. Finished 2026-11-21 (note 31): vectors for CONV4, ROPE and the R8 solve; the layer composites are covered by the functional simulator, which matches the golden model at the rounding floor.*
   - integer group sums, E4M3 decoding, int8 rounding (ties to even, saturating), the order of fp32 adds, the special-function approximations
   - a GPU check that quality matches experiment set 3
   - test vectors for every block
5. **Write `design/spec/spec_v1.md`, run the freeze checklist below, and get the user's sign-off.** *User 2026-11-21: cost numbers (wafer, PHY, board) can come later; the priority is a working chip in RTL with the performance numbers. The spec is written once the functional simulator has also run prefill and the speculative step, since those runs still change the ISA.*
6. **Build the chip in simulation** (D-016 steps 4–5).
   - the instruction-level simulator, bit-exact against golden v2 *(started 2026-11-21, note 31: `isa/fsim_v1.py` runs the 0.8B decode end to end)*
   - the RTL for every block, each checked against its step-4 test vectors
   - the whole chip in Verilator running Qwen3.5-0.8B end to end
   - stretch goals (D-014): layout of the main blocks with open tools, an FPGA prototype

## When spec v1 counts as frozen

- **Every number in the spec** links to a decision, and that decision links to evidence: a measurement, the performance model, or a cited source.
- **The performance model predicts** the D-008 score at or above the D-010 bar (0.273 tok/s per $) on the base card, using cited prices.
- **Golden v2 (chip-exact)** matches experiment set 3's quality within noise.
- **The whole Qwen3.5-9B decode and speculative step is written in ISA v1** and runs in the performance model.
- **The user has signed off** on items 1, 3 and 11.

## For RTL simulation: the stand-in model

- A full 9B token is about 31M cycles, too slow for Verilator. Qwen3.5-0.8B has the same layer pattern:
  - 24 layers, 3 DeltaNet to 1 full attention
  - head dim 256, DeltaNet head size 128
  - the same 248K vocabulary and an MTP head
- **Two differences** the program has to be parameterized for:
  - its DeltaNet query/key heads equal its value heads (16:16, where the 9B has 16:32)
  - its LM head is tied to the embedding
- Hidden size is 1,024, so one weight lane beat covers one row of the hidden dimension.
