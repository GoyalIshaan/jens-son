# Control and feeding

Someone has to tell the array what to do on every cycle. How you answer that question decides how much of your chip is control logic, how hard your compiler is, and how predictable your machine is.

::viz pipeline

## The spectrum of control

The core tradeoff: hardware that figures things out at runtime costs area and power. Software that figures things out in advance costs engineers and flexibility.

| Style | Who schedules | Hardware cost | Compiler burden | Example |
| --- | --- | --- | --- | --- |
| Out-of-order superscalar | Hardware, per instruction | Very high | Low | CPUs |
| SIMT with warps | Hardware, per warp | High | Medium | GPUs |
| Modest VLIW with DMA | Compiler, per bundle | Low | High | TPUs |
| Very wide static VLIW | Compiler, cycle-exact, whole system | Very low | Extreme | Groq |
| Fixed function | Nobody, burned into silicon | Lowest | None | Taalas, in the extreme |

Fancy cores give great performance boosts, at a cost in area, power and parallelism. That is the main reason CPUs are poor at AI. [IA]

## Why VLIW fits AI

A VLIW bundle packs several operations for different units into one instruction. The compiler must guarantee they can run together. Hardware becomes simple, and the compiler burden is heavy. [IA]

This suits AI because AI code has few branches, and the branches it has rarely depend on data. Everything is very regular. [R] Program size is small, so wider instructions are cheap, and CPU-style backward binary compatibility is unnecessary. [J]

TPU v2's bundle is 322 bits: 2 scalar slots, 4 vector slots (2 for load/store), 2 matrix slots (push and pop), 1 miscellaneous slot and 6 immediates. [IA, J] As hardware grew, Google simply widened the bundle. Ironwood's is over 50% wider. [J]

The sequence runs like this: scalar unit fetches the bundle, runs scalar slots, forwards decoded instructions to the vector and matrix units, which run decoupled from scalar execution. [J]

## SIMT and what it buys

GPUs run warps of 32 threads (AMD uses waves of 64 in the datacenter) sharing one instruction. [IA] When a warp stalls on memory, the hardware switches to another warp. This is **warp scheduling**, and Roune calls it the largest part of what makes a GPU a GPU. [R]

His argument is that warp scheduling is hardware-implemented pipelining. The same result can come from software pipelining at less hardware cost, since AI memory access is predictable. It is an artifact of graphics heritage, where shaders do have unpredictable access. [R]

The other side of this argument matters too. Warp scheduling makes high performance reachable for programmers who have never heard of software pipelining. If you pipeline by hand, CUDA is far more productive. [R] That is a large part of why CUDA won.

Watch how GPUs have moved toward TPU-like structures: Blackwell's SM has a Tensor Memory Accelerator for bulk asynchronous copies, and dedicated Tensor Memory beside each tensor core. [IA] Warp specialization, where some warps only move data, is awkwardly recreating CPU-style threads inside a GPU. [R]

## Cycle tyranny

An array consumes operands continuously. Suppose it takes one LHS vector and one RHS vector, and emits one result per cycle. Reading, writing and post-processing could mean about 10 vector operations per cycle, each 256 wide, with no gaps allowed. Roune calls this the tyranny of the cycles. [R]

One non-TPU chip project failed exactly here. The control core could not keep up, so the array could never reach full rate, regardless of how it was programmed. [R]

Ways to reduce this pressure:

| Technique | How it helps | Cost |
| --- | --- | --- |
| 2D vector registers | One 8×128 register carries 8 rows, so 8× fewer instructions | Tiling complications |
| Queues around the array | Burst rows in, then do other work | Queue area |
| Slower array clock | More control cycles per array cycle | More arrays needed |
| Loop unrolling | Addresses become constant offsets | Code size |
| DMA straight into the array | Control acts every few thousand cycles | Custom transforms still need the vector path |

An instructive story: TPU v2 engineers saw inner loops with heavy address arithmetic, and added a dedicated address unit. Nobody used it, since unrolling made those addresses constants. The sample loops were not unrolled when they measured. [R] This is a clear case of what the *Judgement* chapter calls derived consequences.

Systolic array DMAs are Roune's idea: a DMA delivers thousands of rows directly into the array from SRAM, HBM, the network or another array. It is close to a dataflow design. It breaks down when you need to transform data in transit, or when op fusion puts vector work between load and multiply. [R]

## Software pipelining

This is the most important software concept in AI hardware. A loop has three stages, A, B and C, on different units. Run in order, only one unit is active at a time, so utilization is 33%. Overlap iterations, and all three run at once. [R]

| Step | Unit A | Unit B | Unit C |
| --- | --- | --- | --- |
| 1 | x₁ | | |
| 2 | x₂ | a₁ | |
| 3 | x₃ | a₂ | b₁ |
| 4 | x₄ | a₃ | b₂ |
| 5 | | a₄ | b₃ |
| 6 | | | b₄ |

Twelve sequential steps become six. For S stages and N iterations, total steps are N + S − 1, and utilization in the prologue and epilogue is lower. Key rules:

- The slowest stage sets the pace, so balance stages.
- Prefer fewer stages. Each adds a prologue and epilogue, and needs buffers.
- Each live operation needs resources, such as a second buffer.

The universal AI loop has three stages: load tile, compute tile, store tile. Every kernel does this. Roune's recommendation is a tiled pipelining library with declared stages and tile coordinates, so nobody writes this by hand. [R]

Hardware needs very little for this: a queue for memory requests. All CPUs have one. TPUs have one for DMAs. [R]

## Synchronization

With everything asynchronous, you need to know when data has arrived.

- **Sync flags.** TPU DMAs set a flag on completion, and the program stalls until it is set. [J]
- **Semaphores and barriers.** General coordination between engines. [E]
- **Time.** Groq's chips know the cycle on which everything happens. SYNC parks instruction queues, and NOTIFY wakes them, so all 144 queues proceed in lockstep. [IA]

## Static scheduling at the extreme

Groq's instruction set includes operations that exist only for time: `NOP n` waits, `DESKEW` pauses until a hardware counter overflows, and a `Runtime Deskew` delays a chip by a small amount. [IA] The whole machine is plesiochronous: separate clocks at nearly the same rate, aligned through hardware and software counters. Links are paced by software, and forward error correction replaces retransmission, because a retry would break the schedule. [IA]

What you get: fully deterministic timing, simple hardware and high utilization. What it costs:

- The compiler must schedule every operation in the system to the cycle.
- If skew varies, execution stalls by design.
- Drift, jitter and memory refresh become software problems. Groq's patents cover scheduling around DRAM refresh, yet no Groq chip has used DRAM. [IA]

Irrational Analysis's reversal is worth understanding. After years of calling the design an abomination, the author concluded that if the compiler works, real clock forwarding could remove the worst weakness. Nvidia's ISSCC 2026 optical link holds forwarded-clock phase variation to about 16 ps. That is an inference from an investor with disclosed positions, in an unfinished draft. It remains an interesting way to think: when one key limit gets removed, a discarded architecture can return.

Roune cautions against overrating static scheduling, since TPUs also allow careful timing, and the gain over other chips is exaggerated by marketing. [R]

## Do you need a host?

GPUs and TPUs rely on a host CPU. Roune argues this adds cost, an extra failure domain and extra data hops, and that a chip with its own CPU cores could run Linux itself. [R] To stop OS interrupts from disturbing arrays, keep OS work on cores with no attached array.

Ironwood has 2,304 hosts for 9,216 chips, and hosts are its main availability challenge. [J]

## Hardware limits are software interfaces

Every hardware limit is a question: how many DMAs in flight? How deep is this queue? How many descriptors? Early TPU teams guessed at many of these, and software teams did not know a decision had been made. [R] Roune's rule: hardware should never design a software-facing interface without software. And almost everything is software-facing.

## Tuning reference

| Knob | Effect | Cost |
| --- | --- | --- |
| VLIW width | More parallel units controlled | Instruction memory |
| Register tile depth | Less control per array vector | Tiling constraints |
| Queue depth | Decoupling, fewer stalls | Area, latency |
| Outstanding DMAs | Sustained bandwidth | Tracking tables |
| Simple CPU cores | Runs collectives, OS work | Area |
| Static scheduling | Removes arbitration | Compiler effort, brittleness |
