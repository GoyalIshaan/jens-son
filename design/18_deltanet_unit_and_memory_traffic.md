# 18. The DeltaNet unit, and keeping memory busy

*Written 2026-11-11. Status: design sketch and design history. Nothing here is decided. Numbers use the 512 GB/s card from note 11 and the fp32 state from D-005.*

## Part 1: the DeltaNet unit

### What it computes

24 of the 32 layers are Gated DeltaNet. Each layer has 32 heads, and each head keeps a memory S of 128 × 128 numbers. For every token, each head:
1. **Fades** S by a decay g between 0 and 1.
2. **Reads with the key:** kv_mem = Sᵀk.
3. **Writes the correction:** delta = β·(v − kv_mem), then S ← S + k·deltaᵀ.
4. **Reads the output with the query:** o = Sᵀq.

**A tiny example (2 × 2):**
- **Start:** S = [[0.2, 0.4], [0, 0]], with k = [1, 0], v = [0.5, 0.2], β = 0.5, g = 0.9.
- **Fade:** S = [[0.18, 0.36], [0, 0]].
- **Read with the key:** kv_mem = [0.18, 0.36].
- **Correction:** delta = 0.5 × ([0.5, 0.2] − [0.18, 0.36]) = [0.16, −0.08]. Row 1 becomes [0.34, 0.28], halfway to v.
- **Output** with q = [1, 0]: [0.34, 0.28].

### Sizes per token

- **The memory:** 32 × 128 × 128 = 524,288 numbers per layer, 12.6 M in all. That's 2.1 MB per layer and 50 MB in all at fp32.
- **Traffic:** read plus write is 4.2 MB per layer, about 100 MB per token. That's about 2% of the 4.6 GB per token.
- **Arithmetic:** about 4 multiply-adds per memory number, about 2.1 M per layer and 50 M per token. That's tiny next to the 8 B for the weight matrices.
- **What makes it hard** is that each update depends on the previous token's memory, and the memory has to stream in and out every layer. The arithmetic itself is small.

### The one-pass trick

new S = g·S + k·deltaᵀ, so the output Sᵀq can be computed from the *old* memory:

> o = g·(S_oldᵀq) + (kᵀq)·delta

So a single pass over each head's memory gives both reads, Sᵀk and Sᵀq. Then delta is computed, and the new rows are written from the on-chip copy. The output only needs a correction by one number, k·q. **One DRAM read and one write per head, instead of three passes.** We still need to check this rewrite is accurate enough in fp32, using the golden model.

### Data flow per head

1. Load the head's 64 KB from DRAM into an on-chip buffer. The next head loads at the same time (double buffering).
2. **Pass 1:** for each of the 128 rows, add that row into both Sᵀk and Sᵀq.
3. Compute delta (128 values).
4. **Pass 2:** for each row, new row = g·old row + k[i]·delta. Write it back.
5. o = g·(Sᵀq) + (k·q)·delta, then send it to the gated RMSNorm.

### Sizing

- **Memory time per layer:** streaming the 4.2 MB in and out at 512 GB/s takes about 8 µs.
- **Compute:** about 3 row-operations × 128 rows × 32 heads ≈ 12,000 row-operations per layer.
  - 128 fp32 lanes: about 12 µs. Longer than memory, so the unit would hold things up.
  - **256 fp32 lanes: about 6 µs. Fits inside the 8 µs, so it's hidden.**
  - Verifying 4 speculative tokens per load: about 24 µs per layer. This is the case for 512 lanes.
- **Total:** about 24 × 8 µs ≈ 0.2 ms per token, out of about 9.8 ms.

### The smaller pieces

- **Causal conv, kernel 4, over 8,192 channels:** 32 K multiply-adds per token. It needs the last 3 inputs per channel: 98 KB per layer, 2.4 MB in all, either on chip or streamed.
- **A special-function unit** (exp, softplus, sigmoid, rsqrt) for the gates, the L2 norms and the gated RMSNorm.
- **The lazy-commit log for speculation:** k, v, β and decay per draft token. About 50 KB per layer for 4 tokens.

### Open questions

- 128, 256 or 512 lanes, which depends on how many speculative tokens we verify.
- Conv history on chip (2.4 MB) or streamed.
- The accuracy of the one-pass trick and of the special functions.

## Part 2: ideas for memory traffic, and what we concluded

### Idea: dedicate memory lanes (a whole GDDR7 chip) to DeltaNet. Rejected.

- One chip out of four is 128 GB/s. DeltaNet needs about 100 MB per token × 100 tok/s = 10 GB/s, so that chip would sit about 92% idle.
- Weights would drop from 512 to 384 GB/s, about 25% slower overall.
- **The lesson:** when memory sets the speed, reserving bandwidth wastes it. All lanes should be shared.

### Idea: place the DeltaNet unit next to the memory interface to cut latency. No real speed effect.

- Crossing the chip takes about 1 to 2 ns. A DRAM access takes about 100 ns. The DeltaNet work takes about 6 to 8 µs per layer.
- Placement matters for wiring and power (the floorplan), not for speed.
- **What decides speed is overlap: memory must never be idle.**
  - The DeltaNet memory loads while the input projection weights are finishing.
  - The next head loads while the current one computes.
  - The output projection weights start streaming while DeltaNet computes.

### Smart point 1: on-chip DeltaNet memory is the extreme version of "close to the unit"

- 50 MB of SRAM, about 25 mm² (roughly +30% die). It removes the 100 MB per token of traffic: about 2% more speed.
- **Not worth it at today's prices.** But it's the right choice if SRAM becomes cheaper, or if several users share a chip, since each user brings their own 50 MB of memory to stream.

### Smart point 2: switching between reads and writes costs time. A real job for the controller.

- DRAM loses time every time it switches from reading to writing.
- Weights are pure reads. The DeltaNet memory (2.1 MB per layer) and the KV cache are writes that interrupt the weight stream.
- **Fixes:**
  - the controller collects writes and sends them in large batches
  - writes are spread across all 4 chips
- The performance model has to measure how much time this costs.

### Smart point 3: a read-only weight region in memory

DRAM isn't faster for data that's never written. But a controller that *knows* a region is read-only after load can:
1. **Partition the banks.** Reserve a few banks in every chip for written data (DeltaNet memory, KV), and give the rest to weights. Writes then never close a weight row, and no chip's bandwidth is reserved. This is the good version of "dedicated lanes".
2. **Lay the weights out in the order they're streamed,** once at load time, spread evenly across the 4 chips. Nearly every read then hits an already-open row.
3. **Simplify and protect.** No write-ordering checks are needed for the region, and it can be write-protected after loading.

It doesn't remove the read/write switching on each chip's shared data wires, so write batching (point 2) is still needed.

## The principle

> When memory sets the speed, the goal is not to make any single operation fast. It's to keep memory busy 100% of the time with useful bytes. Overlap work, batch the writes, keep writes away from the weight stream, and never reserve bandwidth that might sit idle.
