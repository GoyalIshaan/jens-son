# One forward pass of Qwen3.5-9B, op by op, and which unit runs each op

Source of truth: `isa/qwen_v1.py` (the program the chip runs), `isa/fsim_v1.py` (what each command does),
`golden/bitexact_vec.py` (the exact math). This file follows one **decode pass**: one new token (T = 1)
goes all the way from token id to next-token id. Prefill and speculative passes differ in a few places,
listed at the end.

## The units

- **Scalar core (S slot)**
  - Address arithmetic (layer offsets, position, buffer parity), hardware loops, semaphore waits.
  - It never touches model data. It only works out where things are and when they may start.
- **DMA engines (D slot), four queues**
  - W: DRAM → weight FIFO. Streams every weight matrix, in program order, straight to the matrix unit.
  - R: DRAM → SRAM. Embedding rows, layer parameters (norm weights etc.), DeltaNet state, KV scales, RoPE tables.
  - X: SRAM → DRAM. Writes back DeltaNet state, conv windows, new K and V.
  - L: SRAM → SRAM. Reshapes in place, for example transposing a new V row into the V^T layout.
- **Matrix unit (M slot)**, `rtl/chip/chip_mxu.sv` + `mxu_row_pipe.sv`
  - 1,024 weight lanes (64 groups of 16).
  - `TAB`: pops a matrix's 32-byte header (16-entry 4-bit decode table, fp32 tscale) from the FIFO.
  - `MM`: int8 activations × weights from the FIFO. Weights are 4-bit codes with an E4M3 scale per 16
    (`lut4`), or plain int8 for K and V (attention).
  - Epilogues: `f32` (write fp32), `acc` (add into the destination, used for every residual add),
    `argmax` (keep a running best and its index, used by the LM head).
- **Vector unit (V slot)**, `rtl/vec/`. One command queue, several engines inside:
  - FMA bank (`fp32_fma_pipe8.v`): ADD, SUB, MUL, FMA, ROPE, CONV4, DEQ.
  - Pass path: COPY, MAX, CAUSAL (no FMA, raw bits kept).
  - Reducer (`vec_reduce_rows.sv`): RSUM, RSUMSQ, RMAX.
  - SFU (`sfu.v`): EXP, SIGMOID, SILU, SOFTPLUS, RSQRT, RECIP. Each computes f(a × b + c); b and c are optional.
  - Quantizer (`act_quant32_pipe.v`): QUANT (int8 + 2^e per 32), QKV (int8 + fp16 per row).
  - DeltaNet head engine (`vec_dn_head.sv`, `vec_dn_group.sv`): DN, one whole head's state update.
  - Softmax tile (`vec_smtile*.sv`): SMTILE, the fused online-softmax step (when `fused=True`).

## Model shape, for the numbers below

- Hidden size 4,096. 32 layers in 8 groups of [DeltaNet, DeltaNet, DeltaNet, Attention]: 24 DeltaNet layers, 8 attention layers.
- Every layer = a mixer (DeltaNet or attention) + an FFN, each wrapped in "norm → work → add to residual".
- DeltaNet: 16 key heads and 32 value heads, all of size 128. Each value head has a 128 × 128 fp32 state.
- Attention: 16 query heads and 4 KV heads (4 query heads share a KV head), head size 256, RoPE on the first 64 dims, plus an output gate.
- FFN: 4,096 → 12,288 (gate and up) → 4,096 (down).
- Vocabulary 248,320.
- The residual stream `H` (4,096 fp32 values) lives in SRAM the whole pass.

---

## Step 0. Before the first layer

1. **Look up the token id**, multiply by the row size → DRAM address of the embedding row.
   - Unit: scalar core.
2. **Fetch the embedding row**: 4,096 int8 values + one fp16 scale per 32.
   - Unit: DMA R.
3. **DEQ**: int8 × its scale → 4,096 fp32 values. This is the starting residual `H`.
   - Unit: vector, FMA bank.
4. **Fetch the RoPE angle tables** for this position: a coarse row (position / 1,024) and a fine row (position mod 1,024).
   - Unit: DMA R.
5. **Build cos and sin for this position** from the two rows by angle addition:
   - cos(A+B) = cos A · cos B − sin A · sin B
   - sin(A+B) = sin A · cos B + cos A · sin B
   - That's MUL, FMA, MUL, FMA over 32 frequencies.
   - Unit: vector, FMA bank.
   - Why two tables: one table for every position up to 40K would be huge. Two 1,024-row tables cover 1M positions.

---

## Step 1. A DeltaNet layer (runs 24 times)

### 1a. Layer parameters

1. **Load this layer's small parameters** (norm weights, conv taps, dt_bias, −exp(A_log), gated-norm weight) into one of two parameter slots.
   - Unit: DMA R. The other slot is still in use by the previous layer: double buffering.

### 1b. Input RMSNorm (the same 4 ops before every mixer and every FFN)

1. **RSUMSQ**: ss = x0² + x1² + ... + x4095², in the fixed golden order.
   - Unit: vector, reducer.
2. **RSQRT**: r = 1 / sqrt(ss × (1/4096) + eps). The SFU's f(a·b + c) form gives "mean of squares + eps" for free.
   - Unit: vector, SFU.
3. **MUL**: x × r for all 4,096 values.
   - Unit: vector, FMA bank.
4. **QUANT**: (x·r) × norm weight, then to int8 with one shared 2^e per 32 values.
   - The multiply by the norm weight is folded into QUANT's b input.
   - The output is what the matrix unit reads: 4,096 int8 values + 128 exponent bytes.
   - Unit: vector, quantizer.

### 1c. In-projection

1. **Stream the weights**: 4,096 × 12,352 at 4 bits.
   - Unit: DMA W, then `TAB` on the matrix unit.
2. **MM** 4,096 → 12,352 outputs, written as fp32. The four DeltaNet projections are stored as one matrix:
   - qkv: 8,192 (q 2,048 = 16 heads × 128, k 2,048, v 4,096 = 32 heads × 128)
   - z: 4,096 (the output gate)
   - b: 32 (one per value head, becomes beta)
   - a: 32 (one per value head, becomes the decay)
   - Unit: matrix unit.

### 1d. Short convolution over q, k, v

1. **Fetch the conv history**: the previous 3 tokens' 8,192 qkv values.
   - Unit: DMA R.
2. **CONV4**: for each of the 8,192 channels, y = w0·x(t−3) + w1·x(t−2) + w2·x(t−1) + w3·x(t). This also writes the new 4-token window.
   - Unit: vector, FMA bank.
3. **Write the window back** to the conv log in DRAM, for the next token.
   - Unit: DMA X.
4. **SILU** on all 8,192 conv outputs: y = x · sigmoid(x).
   - Unit: vector, SFU.

### 1e. Normalize q and k (L2 norm per head)

1. **RSUMSQ** over each 128-value head: 16 q heads + 16 k heads = 32 sums. These are short rows, so they pack 4 per beat.
   - Unit: vector, reducer.
2. **RSQRT**: 1 / sqrt(sum + 1e−6), 32 values.
   - Unit: vector, SFU.
3. **MUL**: each head's 128 values × its own r.
   - Unit: vector, FMA bank.
4. **MUL**: q × 1/sqrt(128) (the attention-style scale, applied once here).
   - Unit: vector, FMA bank.

### 1f. Beta and decay (32 values each, one per value head)

1. **SIGMOID**(b) → beta, how strongly this token writes into the state.
   - Unit: vector, SFU.
2. **ADD**: a + dt_bias.
   - Unit: vector, FMA bank.
3. **SOFTPLUS**: log(1 + e^x).
   - Unit: vector, SFU.
4. **MUL** × (−exp(A_log)) → g, a negative number. It's stored already negated.
   - Unit: vector, FMA bank.
5. **EXP**(g) → decay d, between 0 and 1: how much of the old state survives.
   - Unit: vector, SFU.

### 1g. The DeltaNet state update (the heart of the layer)

1. **Load the resident heads' states**: 4 heads at a time, each 128 × 128 fp32 = 64 KB.
   - Unit: DMA R.
2. **DN**: for each value head h (key head = h / 2), with state S (128 × 128), k, q (128), v (128), beta, d:
   - sk = Sᵀ·k: what the state currently predicts for this key (128 values)
   - delta = (v − d·sk) × beta: the correction, scaled by beta
   - sq = Sᵀ·q
   - o = (k·q) × delta + d × sq: this head's output (128 values)
   - S_new = d × S + k ⊗ delta: the outer-product write, 128 × 128 FMAs
   - Small example with 2-dim heads: S = [[1,0],[0,1]], k = [1,0], v = [3,5], beta = 0.5, d = 1
     - sk = [1, 0]
     - delta = ([3,5] − [1,0]) × 0.5 = [1, 2.5]
     - S_new = S + [1,0] ⊗ [1, 2.5] = [[2, 2.5],[0, 1]]
   - Unit: vector, DeltaNet head engine. It's all FMAs, about 4 × 128 × 128 per head per token.
     It runs on the vector unit, not the matrix unit, because the state is fp32 and changes every token.
3. **Store the updated states** back to DRAM.
   - Unit: DMA X.
4. Repeat for 32 / 4 = 8 head groups (a hardware loop).
   - Unit: scalar core.

### 1h. Gated RMSNorm on the 4,096 outputs

1. **RSUMSQ** per head (32 rows of 128).
   - Unit: vector, reducer.
2. **RSQRT**: 1 / sqrt(sum / 128 + eps).
   - Unit: vector, SFU.
3. **MUL** × r, then **MUL** × the gated-norm weight.
   - Unit: vector, FMA bank.
4. **SILU**(z): the gate, 4,096 values.
   - Unit: vector, SFU.
5. **QUANT**: (normed output × silu(z)) → int8 + 2^e per 32.
   - Unit: vector, quantizer.

### 1i. Out-projection and residual add

1. **Stream weights** 4,096 × 4,096.
   - Unit: DMA W, matrix unit `TAB`.
2. **MM** with the `acc` epilogue: H = H + W_out · x. The residual add happens inside the matrix unit's drain.
   - Unit: matrix unit.

### 1j. FFN (the same in every layer)

1. **RMSNorm**: RSUMSQ, RSQRT, MUL, QUANT, as in 1b.
   - Units: reducer, SFU, FMA bank, quantizer.
2. Then 12 chunks of 1,024 (a hardware loop), each:
   - **MM** gate chunk: 4,096 → 1,024. Unit: matrix unit (weights via DMA W).
   - **MM** up chunk: 4,096 → 1,024. Unit: matrix unit.
   - **SILU**(gate). Unit: vector, SFU.
   - **QUANT**: silu(gate) × up → int8. The multiply is folded into QUANT. Unit: vector, quantizer.
   - Chunking lets the vector unit work on chunk i while the matrix unit computes chunk i+1.
3. **MM** down: 12,288 → 4,096 with `acc`, so H = H + W_down · x.
   - Unit: matrix unit.

---

## Step 2. An attention layer (runs 8 times, every 4th layer)

### 2a. Parameters, norm, projections

1. **Load layer parameters.** Unit: DMA R.
2. **Input RMSNorm**: RSUMSQ, RSQRT, MUL, QUANT. Units: reducer, SFU, FMA bank, quantizer.
3. **MM** q + gate: 4,096 → 8,192 (16 query heads × 256, then their 16 gates × 256). Unit: matrix unit.
4. **MM** k: 4,096 → 1,024 (4 KV heads × 256). Unit: matrix unit.
5. **MM** v: 4,096 → 1,024. Unit: matrix unit.

### 2b. q-norm, k-norm and RoPE (done for the 16 q heads, then the 4 k heads)

1. **RSUMSQ** per 256-value head. Unit: vector, reducer.
2. **RSQRT**: 1 / sqrt(sum / 256 + eps). Unit: vector, SFU.
3. **MUL** × r, **MUL** × the q-norm or k-norm weight. Unit: vector, FMA bank.
4. **ROPE** on the first 64 dims of each head: rotate pairs by this position's angles.
   - Small example, one pair (x, y) at angle θ: x' = x·cos θ − y·sin θ, y' = x·sin θ + y·cos θ.
   - Unit: vector, FMA bank.

### 2c. Quantize for the matrix unit and the KV cache

1. **QUANT** q → int8 + 2^e per 32, with exponent bias −4. That bias is an exact × 1/16 = 1/sqrt(256), the attention scale, for free.
   - Unit: vector, quantizer.
2. **QKV** on k: int8 + one fp16 scale per row (max |k| / 127). Unit: vector, quantizer.
3. **QKV** on v: the same. Unit: vector, quantizer.

### 2d. Append the new K and V to the cache

1. **V → V^T staging**: transpose the new V row into a 64-position ring, so V^T can later stream like a weight matrix.
   - Unit: DMA L.
2. **K and V scales → staging.** Unit: DMA L.
3. **K row out** to the KV cache in DRAM. Unit: DMA X.
4. **V^T 32-byte chunks out**, and **scale chunks out.** Unit: DMA X.

### 2e. Attention core, per KV head (4 times), in tiles of 1,024 positions

The 4 query heads that share this KV head are treated as 4 rows (GQA).

1. **Start**: running max m = −∞, running sum l = 0, output O = 0 (COPY of constants).
   - Unit: vector, pass path.
2. For each tile of up to 1,024 cached positions:
   - **Stream the K tile** (int8) from DRAM. Unit: DMA W. K goes to the matrix unit like a weight matrix.
   - **MM q · K**: 4 rows × 256 → up to 1,024 scores. Unit: matrix unit (int8 weight mode, 4 outputs per beat).
   - **Fetch this tile's K and V scales.** Unit: DMA R.
   - Then either the fused path or the unfused path:
     - Fused: **CAUSAL** (last tile only), then **SMTILE**, which does everything below in one command.
       Units: vector pass path, then vector softmax tile.
     - Unfused, the same math as separate commands:
       - **MUL** scores × K scale. Unit: FMA bank.
       - **CAUSAL**: set positions after this token to −∞. Unit: pass path.
       - **RMAX**: the tile's max per row. Unit: reducer.
       - **MAX**: new m = max(old m, tile max). Unit: pass path.
       - **SUB**: score − new m. Unit: FMA bank.
       - **EXP** → p. Unit: SFU.
       - **RSUM**: tile sum of p. Unit: reducer.
       - **SUB** old m − new m, then **EXP** → alpha. Units: FMA bank, SFU.
       - **FMA**: l = l × alpha + tile sum. Unit: FMA bank.
       - **COPY**: old m = new m. Unit: pass path.
       - **QUANT**: p × V scale → int8 + 2^e per 32. Unit: quantizer.
   - **Stream the V^T tile.** Unit: DMA W.
   - **MM P · V**: 4 rows × up to 1,024 → 256. Unit: matrix unit.
   - **FMA**: O = O × alpha + P·V. This rescales the old output to the new max. Unit: vector, FMA bank.
   - The program is software-pipelined: tile i's q·K runs on the matrix unit while tile i−1's softmax runs on the vector unit.
3. **RECIP**: 1 / l. Unit: vector, SFU.
4. **MUL**: O × (1 / l), written into the output layout. Unit: vector, FMA bank.

### 2f. Output gate, o-projection, FFN

1. **SIGMOID**(gate), 4,096 values. Unit: vector, SFU.
2. **QUANT**: attention output × sigmoid(gate) → int8. Unit: vector, quantizer.
3. **MM** o-projection 4,096 → 4,096 with `acc`: H = H + W_o · x. Unit: matrix unit.
4. **FFN**, exactly as in 1j. Units: matrix unit, SFU, quantizer, reducer, FMA bank.

---

## Step 3. After the last layer

1. **Final RMSNorm**: RSUMSQ, RSQRT, MUL, QUANT. Units: reducer, SFU, FMA bank, quantizer.
2. **Stream the LM head**: 4,096 × 248,320 at 4 bits, about 0.5 GB, the biggest single matrix.
   - Unit: DMA W.
3. **MM** with the `argmax` epilogue: 248,320 logits, keeping only the best one and its index. The logits never get written out.
   - Unit: matrix unit (16-way max tree + running compare, from 3c of the report).
4. **The next token id** lands in SRAM and is read for the next pass.
   - Unit: scalar core.

---

## Which unit does what, summed up

- **Matrix unit**: every weight multiply (in-proj, out-proj, q/k/v/o, FFN gate/up/down, LM head),
  both attention matmuls (q·K and P·V), every residual add (`acc` epilogue), and the final argmax.
- **Vector unit, FMA bank**: elementwise math (MUL, ADD, SUB, FMA), DEQ, CONV4, ROPE, building cos/sin,
  and the O = O·alpha + P·V rescale.
- **Vector unit, reducer**: every RMSNorm and L2-norm sum of squares, and softmax max and sum (unfused).
- **Vector unit, SFU**: RSQRT (norms), SILU (conv, FFN, DeltaNet gate), SIGMOID (beta, attention gate),
  SOFTPLUS and EXP (DeltaNet decay), EXP (softmax), RECIP (attention 1/l).
- **Vector unit, quantizer**: every QUANT (before every matrix multiply) and QKV (K and V for the cache).
- **Vector unit, pass path**: COPY, MAX, CAUSAL.
- **Vector unit, DeltaNet head engine**: the DN state update, 32 heads per DeltaNet layer.
- **Vector unit, softmax tile**: SMTILE, when the fused path is used.
- **DMA**: all weights (W), all small loads (R), all write-backs (X), the V transpose (L).
- **Scalar core**: addresses, loops, semaphores. No model math.

## Rough counts for one decode pass

- Matrix multiplies: 24 DeltaNet layers × (1 in-proj + 1 out-proj + 25 FFN) + 8 attention layers ×
  (3 q/k/v + 1 o-proj + 25 FFN) + 1 LM head. The 25 = 12 gate chunks + 12 up chunks + 1 down.
- RMSNorms over 4,096: 2 per layer × 32 + 1 final = 65.
- DeltaNet head updates: 24 × 32 = 768, each touching a 64 KB state.
- Attention: 8 layers × 4 KV heads × (context length / 1,024) tiles.

## How the other passes differ

- **Prefill** (`isa/qwen_prefill_v1.py`): many tokens at once (a chunk of rows).
  - Each matrix multiply processes the whole chunk of rows per weight stream, which is why prefill is mostly matrix-bound.
  - DeltaNet uses the chunked form: the vector part is a `DNC` command, and the chunk-wise matmuls go to the
    matrix unit as `MMS` (weights read from SRAM instead of the FIFO).
  - The per-head short rows (128 and 256 values) are where the packing from the report pays off.
- **Speculative step** (`mode="spec"`): the main pass runs on k + 1 tokens, then each draft runs the MTP
  layer (2 norms, an fc 8,192 → 4,096, one attention layer + FFN over a 4K-position ring, a norm,
  a 32K-vocabulary LM head with argmax). DeltaNet also replays logged tokens (DN "committed replay")
  so rejected drafts never corrupt the state.
