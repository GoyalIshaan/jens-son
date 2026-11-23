# Component atlas

The encyclopedia catalogs about 995 entries in 40 categories. [E] This atlas organizes them by why an architect cares. It does not repeat definitions: the original PDF is in `claude_research/`. Use this to find which categories matter for which decision, then read the entries there.

A reminder from the source: "helps" describes an intended benefit, not a free gain. Every block costs area, energy, wires, latency and design effort.

## Tier 1: the datapath you are designing

| Category | What to learn | Key entries |
| --- | --- | --- |
| 05 Small datapath | How arithmetic is built | Carry-save, parallel-prefix, accumulator, barrel shifter, leading-zero counter, saturation |
| 06 Larger arithmetic | Inside a MAC | Booth recoder, compressor tree, Wallace and Dadda trees, FMA, aligner, normalizer, polynomial evaluator, lookup table |
| 07 Compute organizations | The array choices | SIMD, SIMT, VLIW, dot-product and outer-product engines, systolic array, reduction tree, CGRA |
| 08 AI blocks | Transformer pieces | GEMM engine, softmax, LayerNorm, RMSNorm, GELU, RoPE, Top-k, MoE router, KV-cache controller, KV allocation table, speculative-state buffer |
| 09 Numerics and compression | Format hardware | Quantizer, requantizer, block-floating exponent, stochastic rounding, bit packer, sparse-index decoder, decompression engine |

## Tier 2: what feeds the array

| Category | What to learn | Key entries |
| --- | --- | --- |
| 10 Memory types | Which storage to use | SRAM port types, register file, scratchpad, CAM |
| 11 Inside a macro | Why SRAM costs what it does | Bitcell, sense amplifier, spare row, repair mux, ECC, read and write assist |
| 12 Caches | The automatic alternative | Tag array, MSHR, prefetcher, coherence controller, directory |
| 13 Queues | Decoupling | Skid buffer, ping-pong buffer, credit tracker, async FIFO, reorder buffer |
| 14 Data movement | The DMA path | Scatter/gather DMA, multidimensional stride generator, outstanding-request table, transfer-completion tracker |
| 15 Rearrangement | Layout hardware | Crossbar, permutation network, transpose engine, bank-address remapper, broadcast and reduction networks |
| 16 Control | Sequencing | FSM, microsequencer, scoreboard, arbiters, barrier controller, semaphore |
| 19 On-chip communication | NoC parts | Router, virtual channel, allocators, credit flow control, deadlock avoidance |

## Tier 3: what connects and powers it

| Category | What to learn | Key entries |
| --- | --- | --- |
| 21 External memory | HBM path | HBM controller, memory PHY, bank scheduler, refresh controller, training engine |
| 23 High-speed links | SerDes | CDR, DFE, FFE, CTLE, FEC, link training, eye monitor |
| 24 Clocks | Timing distribution | PLL, DLL, clock tree, clock mesh, clock gating cell |
| 25 Synchronization | Domain crossing | Two-flop synchronizer, gray pointers, async FIFO, reset synchronizer |
| 26 Power management | Keeping the chip alive | Power switches, isolation, level shifter, DVFS, droop detector, throttling controller |
| 39 Wires and power delivery | Physical limits | Repeater, power mesh, backside power, ESD clamp |
| 40 Die-to-die and package | Multi-chip | Silicon bridge, TSV, microbump, hybrid-bonding pad, RDL, heat spreader |

## Tier 4: what keeps it trustworthy

| Category | What to learn | Key entries |
| --- | --- | --- |
| 31 Test and debug | Bring-up | Scan, BIST, trace buffers |
| 32 Reliability | Scale | ECC, memory scrubber, replay controller, lockstep comparator, spare functional unit, aging monitor |
| 30 Security | Multi-tenant trust | Roots of trust, encryption engines |
| 20 Address translation | Isolation | MMU, IOMMU, TLB, protection |

## Tier 5: know they exist

| Category | Why it may matter |
| --- | --- |
| 01 Semiconductor devices | Threshold voltage options, FinFET vs nanosheet, fuses for repair |
| 02 Passives | Decoupling |
| 03 Combinational logic | Compound gates and muxes |
| 04 State elements | Flop types, scan flop, multibit flop, retention flop |
| 17 CPU structures | Needed if you follow Roune's AI-CPU idea |
| 18 GPU structures | Warp scheduler, operand collector, coalescer: what you avoid or replicate |
| 22 Peripherals | PCIe, Ethernet, JTAG, I²C for bring-up |
| 27–29 Analog, converters, RF | Mostly in PHYs |
| 33 FPGA | Prototyping |
| 34 Application engines | Video, crypto, compression engines |
| 35 Specialized compute | Analog MAC, compute-in-memory, near-memory, photonics |
| 36–37 Sensors, photonics | Thermal sensors, optical links |
| 38 Physical layout | Tap cells, keepouts, routing channels, seal ring |

## Component to chapter map

| If you wonder about | Read |
| --- | --- |
| Wallace tree, Dadda tree, compressor, carry-save | Reduction trees |
| Crossbar, router, virtual channel, credit | Crossbars and NoC |
| Scratchpad, MSHR, cache, banks | Memory |
| VLIW, scoreboard, warp scheduler, DMA | Control and feeding |
| Softmax unit, polynomial evaluator, exp | Vector units |
| Spare row, FBIST, replay, ECC | Generations, Scale-out |
| Clock tree, repeater, droop detector | Area, clock, power |
| TSV, hybrid bonding, interposer | Area, clock, power, Memory |

## The five questions, again

For any entry you meet, ask the encyclopedia's own questions. [E]

1. What work does it perform?
2. What supplies its data?
3. What limits its throughput?
4. How much state and routing does it require?
5. Does the target process or IP library support it?
