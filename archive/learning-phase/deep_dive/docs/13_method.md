# Method: from goal to architecture

Here is the procedure, with three worked examples. Every number comes from `model/` or is a stated assumption.

::viz workbench

## The procedure

### Step 1. Write the workload down

List model sizes, layer shapes, precision, context, target token rate and cost metric. Decide whether you control the software.

### Step 2. List every matmul shape

Write M, N, K for each operation. Note the smallest K and smallest M you must run well. These set your array width.

### Step 3. Find intensity at your batch

Compute FLOPs and bytes per step. Compare to candidate ridge points. This tells you whether you are buying compute, bandwidth or capacity.

### Step 4. Budget bandwidth and capacity first

Decode is usually limited by these. Fill in:

`required bandwidth = (weights + batch × KV per user) ÷ step time`

`required capacity = weights + batch × KV per user`

### Step 5. Choose numerics

Pick the narrowest format your models tolerate, then compute total bytes again.

### Step 6. Choose array width and count

Width from the smallest K you care about. Count from required throughput ÷ (clock × width²). Check area and power.

### Step 7. Size the on-chip memory

From tile size, double buffering, fusion and staging. Check bank bandwidth at each array edge.

### Step 8. Size the feeders

Vector throughput, DMA queue depth (bandwidth × latency), scalar control. Check cycle tyranny.

### Step 9. Choose parallelism, then network

Decide S and P from capacity and latency. Compute network bandwidth from all-reduce rates. Pick a topology from your traffic.

### Step 10. Close the physical budget

Area, perimeter, power, cooling, yield. Identify what binds. Iterate.

### Step 11. Search around your answer

Change several things at once. Bloat protects bloat, and savings compound through 1/x: two changes that each make something 2× cheaper can make it 100× cheaper together if costs are nearly gone. [R] A model will be wrong, but it will show you options and sensitivities. [R]

## Example 1: a datacenter decode system

**Goal.** Serve a dense 70B model in FP8. 80 layers, 8 KV groups, width 128, 8K context. 50 tokens per second per user, with 256 users per system.

**Step time.** 1 ÷ 50 = 20 ms.

**Weights.** 70 GB. Reading them each step needs 3.5 TB/s.

**KV.** Per user: 2 × 80 × 8192 × 8 × 128 × 1 = 1.34 GB. For 256 users: 344 GB. Reading it each step: 17.2 TB/s.

**So KV traffic is five times weight traffic.** That is the main finding. Batching makes weights cheap per token and has no effect on KV.

**Compute.** 2 × 70B × 256 × 50 = 1.8 PFLOP/s. A modern accelerator's peak exceeds this.

**Intensity.** 2 × 70e9 × 256 FLOPs per step, over 70e9 + 344e9 bytes: about 87 FLOP/byte. That sits well under any modern ridge point, so this system is memory-bound. The array is not the problem.

**Sizing, with Ironwood-class parts.** Bandwidth: 20.7 TB/s ÷ 7.3 TB/s ≈ 3 chips. Capacity: 414 GB ÷ 206 GB (192 GiB) ≈ 2 chips, before any headroom. Use 4 for margin. Bandwidth binds first and capacity is close behind, which suggests this kind of part suits this workload fairly well.

**Network.** X = 8192 × 256 × 2 bytes = 4.2 MB per all-reduce, with 2 per layer, so 160 per step. At P=4 that is about 1 GB per step, or 50 GB/s per chip. Ironwood has 600 GB/s. Easy.

**Now change one thing.** Double the context to 16K. KV doubles to 688 GB, and needs 34 TB/s. You need about 6 chips, and compute is unchanged. Long context is a memory purchase.

**What would a better design change?** Sparse attention would cut that KV traffic. GQA with 4 groups would halve it. 4-bit KV would halve it again. Each of those is worth more here than any array improvement.

## Example 2: an edge device

**Goal.** A 3B model at 4 bits on a phone-class chip with 68 GB/s LPDDR, one user.

**Weights.** 1.5 GB. Bandwidth limit: 68 ÷ 1.5 = 45 tokens per second at batch 1.

**Compute.** 2 × 3e9 × 45 = 270 GFLOP/s. A 32×32 array at 500 MHz gives 1 TOP/s, which is plenty.

**Finding.** This chip's job is mostly memory interface. The array can be tiny. Speculative decoding gives more reuse for one user, so more of the bandwidth works. [R] Even batch 4 is much more efficient than batch 1, and speculation and parallel agents can supply it. [R]

**What not to build.** A big array. Fancy sparsity. A wide NoC. Spend effort on memory efficiency and compression.

## Example 3: a tiny embedded model

See *Your chip*. It gives the opposite conclusion: all weights fit on chip, so bandwidth is cheap, and the shape of the problem drives everything.

## Decision tables

### Which side of the roofline are you on?

| If | You are | Spend on |
| --- | --- | --- |
| Intensity ≪ ridge | Memory-bound | Bandwidth, compression, KV reduction |
| Intensity ≈ ridge | Balanced | Stop tuning this ratio |
| Intensity ≫ ridge | Compute-bound | Arrays, numerics, clock |

### Which limit binds first?

| Symptom | Binding limit | Moves that help |
| --- | --- | --- |
| Watts exceed cooling | Power | Wider and slower, lower voltage, narrower numerics |
| Area exceeds reticle | Area | Faster clock, multi-die, smaller SRAM |
| Perimeter exceeds PHY needs | Beachfront | Multi-die, stacked memory, chiplets |
| Array idle waiting on control | Feeding | 2D registers, queues, DMA feed |
| Array idle waiting on memory | Bandwidth | Larger tiles, fusion, more HBM |
| Array idle on small shapes | Shape | Smaller arrays, block-diagonal packing, heterogeneous cores |
| Latency too high at required batch | Batch | Faster network, more within-layer split, more weight bandwidth |

## The design review checklist

Before you trust a design, answer these.

1. What is the target metric, and what is its denominator?
2. What is the utilization on the smallest shape you must run?
3. Is every queue sized by bandwidth × latency?
4. What is the worst-case power pattern, and how fast can you throttle?
5. How does yield survive one defect in the array?
6. How do you test it at 2×2?
7. What happens when software is worse than you assumed?
8. Which number here is unmeasured?
