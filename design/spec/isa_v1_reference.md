# ISA v1 reference

*Status: proposed (D-022), 2026-11-21. Source of truth: `isa/isa_v1.py` (encodings and field checks) and `isa/sim_v1.py` (timing). The design reasons are in note 28; the bit-exact arithmetic of each command is in golden v2 (note 29, `golden/bitexact.py`, `golden/bitexact_vec.py`).*

## Machine state

- **32 scalar registers** x0–x31, 48 bits wide. x0 is always 0.
- **4 nested hardware loops** (start, end, count).
- **63 counting semaphores**, numbered 1–63. Number 0 in a field means "none".
- **Instruction memory:** 8,192 bundles × 32 bytes (256 KB). Programs are loaded from DRAM; switching between the prefill and decode programs takes under 1 µs.
- **Descriptor memory:** 1,024 descriptors × 64 bytes (64 KB), loaded with the program.
- **SRAM:**
  - 6.5 MB, byte addresses with 23 bits
  - 64 banks × 64-byte lines, low-order interleaved
  - the DeltaNet state buffers sit in their own banks (D-023)
- **Weight FIFO:** 192 KB (D-023). Written only by DMA W, read only by the matrix unit, in program order.
- **Per-unit command queues:** 16 entries each, for W, R, X, L (DMA), M (matrix), V (vector).

## The bundle

- **256 bits = four 64-bit slots:** S (scalar/control), D (DMA), M (matrix), V (vector). An empty slot is all zeros (NOP).
- **One bundle issues per cycle, in order.** All slots read registers before the S slot writes any.
- **Issue stalls when:**
  - a D/M/V command's queue is full, or
  - a WAITS finds its semaphore at 0
- **S slot layout** (52 of 64 bits):
  - op: 5 bits
  - rd, rs1, rs2: 5 bits each
  - imm: 32 bits, signed
- **D/M/V slot layout** (64 bits):
  - op: 5 bits
  - desc: 10 bits
  - r0–r4: 5 bits each (25 bits)
  - wait1, wait2, sig1, sig2: 6 bits each (24 bits)

## S slot: scalar and control

- **Arithmetic:**
  - `ADD SUB MUL AND OR XOR MIN MAX rd, rs1, rs2`
  - `ADDI ANDI SLLI SRLI rd, rs1, imm`
  - `LI rd, imm`
- **Memory:** `LW rd, imm(rs1)` and `SW rs2, imm(rs1)`, 32-bit words in SRAM.
  - Used for token ids, the accepted-draft count, and host buffers.
  - A load of a value a unit produces must follow a WAITS on that unit's semaphore.
- **Branches:** `BEQ BNE BLT rs1, rs2, imm` and `J imm`, relative to the next bundle. The Qwen programs don't use them.
- **Loops:** `LOOP rs1, n`.
  - Runs the next n bundles x[rs1] times; the count is read once, at the LOOP.
  - A count of 0 skips the body.
  - Loops nest 4 deep. Several loops may end on the same bundle.
- **Sync:**
  - `WAITS imm`: stall issue until semaphore imm > 0, then decrement it
  - `SIGS imm`: increment it
- **`HALT`**

## Commands (D, M, V slots)

- **A command** = an opcode + a descriptor (its shape) + up to 5 registers + up to 2 semaphores to wait on + up to 2 to signal.
- **Registers** are read at issue and travel with the command.
- **Each register field has a meaning per opcode:** an offset added to one of the descriptor's base addresses, or an override for one count (0 = use the descriptor's count).
- **Waits:** the command starts when every semaphore it waits on is > 0, then decrements them.
- **Signals:** it increments them when its results are visible to every unit.
- **Every unit runs its queue in order.** The `NOP` form of each unit's opcode waits and/or signals without doing work: a fence.

### D slot: DMA

- **Descriptor `Dma`:** a 3-D copy.
  - n0 contiguous bytes, repeated n1 times (strides s1_src, s1_dst), repeated n2 times (strides s2_src, s2_dst)
  - bases: src 36 bits, dst 36 bits; counts: 24 bits; strides: 32 bits, signed
- **Registers:** r0 = source offset, r1 = destination offset, r2 = n1 override, r3 = n2 override.
- **Opcodes:**
  - **`W`:** DRAM → weight FIFO. Waits for FIFO space; needs no semaphore.
  - **`R`:** DRAM → SRAM. Signals when the last byte has landed.
  - **`X`:** SRAM → DRAM. The controller batches writes; it signals when they're accepted.
  - **`L`:** SRAM → SRAM, any byte granularity (transposes, scatters). Does not touch DRAM.
  - **`H2S`, `S2H`:** host ↔ SRAM over PCIe.
  - **`NOP_W NOP_R NOP_X NOP_L`:** fences in each DMA queue.
- **Ordering:** DRAM accesses from different queues are not ordered against each other. A program that reads what it just wrote (the new K/V) must put a fence that waits on the write's semaphore in the reading queue.

### M slot: matrix unit

- **Descriptor `Mm`:**
  - **Sizes:** n_in and n_out (20 bits); rows (10 bits; more than 64 rows hold each beat ceil(rows/64) cycles).
  - **Activation:** base, row stride, exponent base, exponent stride.
  - **Destination:** base, row stride.
  - **wfmt:**
    - `lut4`: 4-bit codes + an E4M3 scale per 16; table loaded by TAB
    - `int8`: raw int8 values, no table, no scale (K and V)
  - **seg:** 1 (one output per 1,024-input beat) or 4 (four outputs per beat, 256 inputs each).
  - **epi:**
    - `f32`: write the fp32 outputs
    - `acc`: add into the fp32 destination
    - `argmax`: a running best value and index per row; ties go to the lowest index; writes the index + id_base
  - **tscale:** the per-tensor fp32 multiply applied to each output, gmax/127 (note 29).
  - **wsrc:** for MMS only, the SRAM address of the weights.
  - **override:** which count r4 replaces, n_in or n_out.
- **Registers:** r0 = activation offset, r1 = exponent offset, r2 = destination offset, r3 = weight-source offset (MMS), r4 = count override.
- **Opcodes:**
  - **`TAB`:** pop the matrix's 32-byte header from the FIFO: 16 int8 table entries into every group's table, then the fp32 tscale (gmax/127) applied to every output of the next lut4 matrix, then 12 bytes of padding.
  - **`MM`:** weights from the FIFO.
  - **`MMS`:** weights from SRAM.
  - **`NOP`:** a fence.
- **Order of work:** for each tile of 64 outputs (256 in seg-4 mode with rows ≤ 16), for each 1,024-input chunk, for each output of the tile, one beat.
- **The arithmetic per beat** is golden v2's contract (`golden/bitexact.py`): exact group sums, 16 fraction bits rounded down, one fp32 rounding per beat, fp32 accumulation in input order, then × tscale.

### V slot: vector unit

- **Descriptor `Vec`:**
  - a box of n × m × p elements (16 bits each)
  - up to 4 operands (dst, a, b, c), each with a base (23 bits), byte strides sn, sm, sp (20 bits, signed) and a type (f32, f16, i8, u32, e8)
  - a stride of 0 broadcasts
  - an `e8` operand read by an elementwise op (MUL, FMA, ...) gives 2^e, one exponent byte per 32 elements along n: element (p, m, n) reads the byte at base + p·sp + m·sm + (n ÷ 32)·sn. Added 2026-11-21: the lite DeltaNet's residual x − q·2^e needs it (one FMA with `neg`).
  - op-specific parameters
  - override: which dimension r4 replaces
- **Registers:** r0–r3 = offsets for dst, a, b, c; r4 = the dimension override (or a count for DN and CAUSAL, below).
- **Elementwise, f32:**
  - **`ADD SUB MUL MAX`:** dst = a op b.
  - **`FMA`:** dst = fma(a, b, c).
  - **Parameter `neg`** (ADD, SUB, MUL, FMA and the special functions): negates a before the operation, so MUL gives −(a·b) and FMA gives fma(−a, b, c). It costs one sign flip per lane. Added 2026-11-21: the lite DeltaNet needs v − γ·Sᵀk and −A, and the first draft of its program had both signs wrong (found while writing its exact reference).
  - **`COPY`:** dst = a, converting types.
- **Special functions:** `EXP SIGMOID SILU SOFTPLUS RSQRT RECIP`: dst = f(a · b + c), with b and c optional (D-020 widths; `golden/chip_sfu.py`).
- **Reductions over n:** `RSUM RSUMSQ RMAX`. dst gets one value per (m, p). Order rule R1 (`golden/bitexact_vec.py`).
- **Formats:**
  - **`QUANT`:** int8(a × b) with one power-of-two exponent per 32 (D-019); c = the exponent destination; parameter `exp_bias` adds to every exponent (exact).
  - **`QKV`:** int8 with one fp16 scale per row (KV, D-005); c = the scale destination.
  - **`DEQ`:** int8 × fp16 scale per 32 → f32 (the embedding).
- **Model-specific:**
  - **`ROPE`:** rotates the first `rot` dims of each n-row by the cos/sin in b.
  - **`CAUSAL`:** sets −inf where position base (r2) + n > limit (r3) + p × step.
  - **`CONV4`:** the kernel-4 causal conv. b = weights; c = history (3 rows); also writes the new window.
  - **`DN`** executes R4 for one to four consecutive value heads. The optional descriptor parameter `heads` defaults to one. Its two-bit decoded representation is `heads - 1`. r1 selects the first value head. State slot h starts at c.base + r3 + h × 128 × c.sm. Each value head uses its matching query/key head and absolute output column.
    - First replay r4 tokens from the log in place, then run T tokens. T=0 emits no output. A one-head descriptor may select a private working copy with `commit_spec=0`.
    - Grouped updates use `commit_spec=1`. Speculation issues a replay-only descriptor, checkpoints all replayed states to DRAM, waits for its DMA completion, then issues the tentative tokens in the same SRAM banks. The next state load waits for the last local consumer. The speculative state is discarded on bank reuse.
  - **`MATCH`:** counts leading equal u32 values of a and b.
  - **`SMTILE`:** the fused online-softmax step over one tile of scores (order rule R7): × K scale, max, exp, sum, α, l, int8(p × V scale). Operands, fixed 2026-11-21 (the first version had no position or limit operand and no register offset for its running max and sum, note 31 bug 6):
    - a = the fp32 scores (n positions × m rows); r1 offsets it
    - b = the fp16 K scale per position; the V scales sit `vs` bytes after it (parameter); r2 offsets both
    - c = the running max m, one fp32 per row (stride c.sm); α is written at c + 4·`mlb` and l is read and written at c + 5·`mlb` (parameter `mlb`, the size of one of the six row-vector slots m, mn, mt, lt, α, l); r3 offsets all three
    - dst = P, int8; its exponents (one per 32 positions) at dst + `e`[0] bytes, `e`[1] bytes per row (parameter); r0 offsets both
    - r4 = the position count (override n)
    - **No mask:** a tile that needs the causal mask is preceded by `CAUSAL`, which writes −inf into the scores. SMTILE keeps an input of −inf at −inf whatever the K scale (a select, not a multiply), so positions never written (scale 0 or stale) can't turn into NaN. Only the last tile needs it: in decode the last 1,024-position tile, in prefill the 256-position block holding the chunk's own 128 positions (both programs peel that tile out of their loop). Cost: one CAUSAL, 16 cycles per decode tile and 72 per prefill block.
    - Parameter bits: vs (12), e (13 + 6), mlb (10): 41 of the 48.

## Throughput (for the timing model)

- **Matrix unit:**
  - 1,024 lanes × 64 rows
  - lut4: 576 bytes per beat; int8: 1,024 bytes per beat
  - one beat per cycle while rows ≤ 64
  - results 24 cycles after the last beat
- **Vector unit:**
  - 512 fp32 lanes
  - eˣ 128 per cycle; SiLU and sigmoid 64 per cycle (D-023); softplus, 1/√x, 1/x 2 per cycle
  - Grouped DN uses the eight-cycle FMA recurrence and round-robin grants in `isa/sim_v1.py::dn_group_cycles`. For 512 lanes and ready SRAM, a full four-head token costs 1,443 clocks including operand fetch and output transfers. Replay costs 1,043 clocks per four heads.
  - 8 cycles of overhead per command
- **DMA:**
  - DRAM 512 GB/s × 0.92
  - reads arrive about 150 ns after they're served
  - writes cost 1.15× and are batched
  - L copies 64 bytes per cycle

## Conventions the programs rely on

- **Weights** are stored in the order the matrix unit reads them, each matrix with its 32-byte header in front (table, tscale, padding). lut4 data: for each 64-output tile, for each 1,024-input chunk, for each output, 512 bytes of codes (input 2i in the low nibble) then 64 E4M3 scales. int8 data (K, V^T, SRAM operands): [256-input segment][output][256 inputs]. The packer is `isa/pack_v1.py`.
- **K** is stored as natural rows. **V** is stored transposed in 256-position blocks ([block][dim][position]). New V goes through a 64-position SRAM ring and is written as whole 32-byte pieces (decode), or as whole 128-byte rows (prefill).
- **Buffers shared between phases** (prefill) may only be written by the M or V unit, never by a prefetching DMA.
