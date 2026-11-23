# The core insights

Everything in the four PDFs, boiled down. If you only read one page, read this one. Each insight links to the chapter that proves it.

Source tags used throughout: **[J]** Jouppi, Lakshmanamurthy, Young and Patterson, the Ironwood paper. **[R]** Bjarke Roune, *Designing AI Chip Hardware and Software*. **[IA]** Irrational Analysis, *It's the dataflow, stupid* (an unfinished draft). **[E]** the Component Encyclopedia. **[F]** your own factbook in `research/`. **[D]** derived here and tested in `deep_dive/model/`. **[O]** outside the four PDFs: general engineering knowledge you should re-check against a real PDK.

## What a chip actually is

**1. An AI chip is a machine for keeping a systolic array busy.** The array sets the ceiling on speed and efficiency. Everything else on the die exists to feed it, drain it, or do the small amount of work it cannot do. Brand names like TensorCore, MXU, Matrix Core and NeuronCore all refer to the same idea. [R] → *Systolic arrays*

**2. Any computer can be compared with four questions.** How is memory arranged? How does compute reach memory? How fancy and how wide is the compute? How do chips talk to each other? Use these before you read any spec sheet. [IA] → *The frame*

**3. The goal is balance, measured in useful work per dollar or per watt.** Peak FLOPS is one input. A real design balances systolic compute, vector compute, scalar compute, memory bandwidth, memory capacity, network bandwidth, thermals and latency all at once. Adding compute to a chip that is not compute-bound only lowers utilization. [R] → *The frame*

**4. Data movement is the cost you are really managing.** An HBM byte costs on the order of 100 times the energy of an 8-bit multiply-accumulate. Every architecture decision is about how many times you can reuse a byte before you have to fetch another one. [O] → *Area, clock, power*

## Why arrays have the size and shape they do

**5. Doubling an array's side divides scalar work by 4 and vector work by 2.** A 2N×2N array does four times the math of an N×N array with vectors only twice as wide. Going from Volta's 4×4 to a TPU's 128×128 is five doublings: 1024× less scalar work and 32× less vector work per unit of matmul. This is why a 30-person TPU team could compete with Nvidia. [R] → *Systolic arrays*

**6. The reduction depth K is the hard limit on array size.** You can usually find more rows or more columns of work by batching. You cannot invent more terms in a dot product. Attention has K around 16 to 128 while feed-forward has K in the thousands, so attention is what stops arrays from being 1024×1024. [R] → *Matmul shape*

**7. Every factor of two of mismatch costs you a factor of two or four.** A 64-wide problem on a 128-wide array runs at 25%, or 50% if software packs it block-diagonally. A 257 dimension can run at half the speed of a 256 dimension. Padding is the tax on shape mismatch. [R] → *Matmul shape*

**8. The row count needed for full utilization equals the array's width, divided by how much more bandwidth weights get than activations.** On a 128×128 array with equal bandwidth, you need 128 rows of activations per weight tile. One row gets 0.8% utilization. This is the single most important equation in inference hardware, because those rows are your batch. [R, D] → *Matmul shape*

**9. Bigger arrays force bigger batches, which force more KV cache and more latency, which force faster networks.** Array size is not a local decision. It propagates through batch, HBM capacity, token latency and link bandwidth. [R] → *Scale-out*

**10. A larger array clocked down is still more efficient than a smaller array at the same throughput, up to a limit.** Keep this option in your search even when you do not need more FLOPS. Lower clocks allow lower voltages, and dynamic energy falls with voltage squared. [R, O] → *Area, clock, power*

**11. Numerics are the cheapest speedup you will ever get, and they are not free.** Going from 16 to 8 bits halves bandwidth and more than halves multiplier cost. Below 8 bits, the accumulators start to dominate and returns diminish. Every format you support adds area inside the most replicated cell on the chip. Eight bits is always enough for inference if quantization is done well. [R] → *Systolic arrays*

## How the pieces around the array work

**12. A systolic array is a reduction chain laid out in space, and a tensor core is a reduction tree.** Both compute dot products. The chain uses only neighbor wires but has latency proportional to K. The tree has logarithmic latency but needs fan-in wiring. Real designs mix them: small trees inside a processing element, chains between elements. [D, O] → *Reduction trees*

**13. Crossbar cost grows with the square of the port count, so nobody uses one flat crossbar at scale.** You pay in wires, and the die is a 2D plane. Designs use small crossbars inside clusters, then meshes or rings between clusters. [E, IA] → *Crossbars and NoC*

**14. Memory is just another node on the network.** Where you attach SRAM and HBM to the fabric determines hop counts, contention and energy. How the network on chip is set up matters as much as how much memory you have. [IA] → *Crossbars and NoC*

**15. Who decides what lives in SRAM changes the whole machine.** If hardware decides, it is a cache. If software decides, it is a scratchpad. A scratchpad removes tag arrays, replacement logic and timing uncertainty, and it moves that complexity into your compiler. For high-performance AI, manual control ends up being the more convenient option. [IA, R, J] → *Memory*

**16. Complexity is conserved. You choose where it lives.** Groq removed arbiters, caches and replay, and its design is simple enough for a small team to reproduce. The complexity went into the hardest compiler ever written. Warp scheduling puts the same pipelining in hardware at a large area cost. Software pipelining puts it in a library. [IA, R] → *Control and feeding*

**17. Queues buy decoupling, and they do not buy bandwidth.** A deep FIFO absorbs a temporary mismatch between producer and consumer. It does nothing for a permanent mismatch. Size queues from latency times bandwidth, because one undersized DMA queue can stall a whole core. [R, D] → *Control and feeding*

**18. Cycle tyranny sneaks up on people.** An array consumes vectors every cycle, and whatever drives it must never miss a beat. One real chip project could never reach full array rate because its control core could not keep up. Two-dimensional registers, queues, slower array clocks and DMA-fed arrays all reduce this pressure. [R] → *Control and feeding*

**19. Non-matmul work is small until it is your bottleneck.** Softmax, normalization, quantization and activation functions run on the vector unit. Amdahl's law applies at the worst moment: if the array gets 10× faster and the vector unit does not, the vector unit becomes the chip. Ironwood went from two restricted ALUs per lane to four general ones. [J] → *Vector units*

## Workloads and systems

**20. Prefill and training want compute, decode wants memory, and the difference is arithmetic intensity.** A prefill step reuses each byte across thousands of tokens. A naive decode step reuses it once. The same chip therefore has idle arrays during decode and idle memory during prefill. [R] → *Workloads*

**21. Decode attention is hurt by different things than decode feed-forward.** Batch fixes feed-forward because users share weights. Batch does nothing for attention because users do not share keys. Grouped-query attention and speculative decoding fix attention instead, by a combined factor like 64×. [R] → *Workloads*

**22. Most provisioning answers swing by 10× to 100× depending on software.** How much bandwidth, how much HBM and how many chips all depend on how well software uses them. Big HBM and big networks partly exist to protect customers from their own inefficient software. [R] → *Workloads*

**23. The balance of the TPU line has drifted hard toward compute.** Across five generations, peak BF16 per unit of HBM bandwidth rose about 5×. VMEM per unit of compute fell about 13×, and interconnect per unit of compute fell about 20×. My reading is that this works because software got better at reuse and overlap. Each generation asks for more batch. [D] → *Generations*

**24. Pipelining across layers helps weights but not KV cache. Splitting within layers helps both, and costs network bandwidth.** This surprises almost everyone, including the people who sized TPU v2's network. Doubling within-layer parallelism a bit more than doubles per-chip bandwidth. [R] → *Scale-out*

**25. A router hop wastes at least half your bandwidth, and a torus can waste none, until mixture-of-experts shows up.** The classic collectives need only neighbor traffic. MoE traffic is all-to-all, which favors small tori and routed fabrics. This is why topology is still contested between Google's torus and Nvidia's NVLink. [R, IA] → *Scale-out*

**26. At scale, reliability is an architecture feature.** Google's optical switches exist mainly so a failed rack does not take down a pod. Ironwood added a spare array row, functional self-test and hardware replay to catch silent errors. Parallelism multiplies failure domains, and only duplication reduces them. [J, R] → *Scale-out*

## How great designers think

**27. Stability beats novelty.** TPU v2's block diagram still describes Ironwood eight years later. Generations scaled the components and kept the architecture, so compilers and models carried over. Five generations of progress came mostly from more of the same blocks, running faster. [J] → *Generations*

**28. Six decisions turned out to matter most, in this order.** Systolic arrays. Narrow floating point chosen for range. HBM. Custom chip-to-chip links. Software-managed scratchpads with DMA. Vector units for everything else. [J] → *Generations*

**29. Parametrize everything by array width.** Write `VECTOR_WIDTH` as a constant in hardware and software. You can then test on 2×2 tiles where a human can check each number, fit the design on an FPGA, and change the array size between generations without a rewrite. Your RTL already does this, which is the right instinct. [R] → *Your chip*

**30. Numbers mean nothing without context, and derived consequences change the answer.** Turning off op fusion in a compiler built around fusion measures nothing useful. Every hardware experiment needs the question: how would software change in response? Design-space search beats hill climbing, because savings combine superlinearly and bloat protects bloat. [R] → *Judgement*

## The one-paragraph version

Start from a workload and a cost metric. Compute its arithmetic intensity at the batch you can afford. That tells you whether you are buying compute, bandwidth or capacity. Choose array width from the smallest K you must run efficiently, and array count from your area and power budget. Choose numerics as narrow as your models tolerate. Size SRAM so tiles reach the machine's ridge point and double-buffer cleanly. Size DMA queues from bandwidth times latency. Give the vector unit enough throughput that softmax and quantization hide behind the array. Pick parallelism before you pick network bandwidth, and pick topology from your real traffic pattern. Then check every number against what software will really do, because software decides how much of all this you needed.
