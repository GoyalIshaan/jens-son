# Overnight report, 2026-11-22

*What I did while you slept. Details are in note 33 (`design/33_whole_chip_rtl.md`), decisions D-026 to D-029 (`design/DECISIONS.md`), and the blog notes.*

## The headline

- **The whole chip is in RTL, and it runs the real programs bit-exact against FSim.**
  - Every block is RTL, linked in one top level (`rtl/chip/chip_top.sv`): the scalar core with 6 queues and 63 semaphores, 4 DMA engines, the weight FIFO, the matrix unit, the vector unit (note 32), and the SRAM.
  - The test compares every byte of SRAM and DRAM at the end.
- **What passes:**
  - the smoke program: every unit and command kind
  - the real decode program on a 4-layer Qwen3.5-shaped model (hidden 1,024; 3 DeltaNet layers and 1 attention layer), for 1 token and for 3 tokens, LM-head argmax included
  - the real prefill program: one 128-token chunk through the same model
  - 10 random matrix-unit programs (`mmfuzz`), 14 commands each: every format, epilogue and row count from 1 to 128
- **The biggest finding: the timing model assumed the matrix unit's bookkeeping was free, and for prefill it isn't.**
  - Built the obvious way, a 32K prefill on the 9B would take 32.6 s instead of 11.4 s.
  - I redesigned the matrix unit so loading, computing and the epilogue overlap (D-029). That brings it to 11.7 s, which meets D-008's 12 s. Decode stays at 90 tokens/s.
- **The RTL's speed checks out.** The tiny prefill's vector unit matches the RTL-rate model to 3%, so the 9B projections built on that model (90 tokens/s decode, 296 speculative, 11.7 s for a 32K prompt) rest on measured RTL behaviour.

## What I built

- **Binary descriptors** (`isa/desc_v1.py`): each descriptor is a 64-byte word; the RTL's field offsets are generated from the same file.
- **The scalar core** (`chip_core.sv`):
  - one 256-bit bundle per clock
  - 32 registers of 48 bits
  - hardware loops, 4 deep
  - six 16-deep command queues
  - semaphores granted in unit order, so two units can never both take the last count
- **4 DMA engines** (`chip_dma.sv`):
  - 3-D strided copies, 512 bytes per clock, 256 pieces in flight
  - the ISA's done points
  - a byte-transpose path in the SRAM-to-SRAM engine (a 64 × 64 tile, 32 bytes per clock), added after the prefill run showed transposes moving 1 byte per clock
- **The weight FIFO** (`chip_wfifo.sv`), 192 KB.
- **The matrix unit's controller** (`chip_mxu.sv`), around the datapath verified in notes 24–26. It does TAB, MM and MMS (lut4, and int8 seg 1 and seg 4), with the f32, acc and argmax epilogues. It's now the D-029 version.
- **The vector unit's command decoder** (`chip_vunit.sv`): every register exception FSim has.
- **The SRAM port model** (`chip_sram.sv`): it counts, every clock, what D-028's 64 banks would have to serve.
- **The test harness:**
  - `run_chip.py`: tests smoke, mmfuzz, tiny and tinyprefill
  - `tb_chip.sv`: DRAM timed like the model (one shared 471-byte-per-clock server, 150 clocks latency), and a per-command trace
  - `compare_timing.py`: lines the RTL's trace up against the timing model, command by command and by opcode; `MODEL=rtl` compares against the measured-rate model instead
- **Verilator** (`build_verilator.sh`), about 60× faster than Icarus. This Mac needed two workarounds: the 26.5 SDK, and Apple's `ar`.
- **Measurement scripts:**
  - `isa/alias_lint.py`: D-027
  - `isa/sram_traffic.py`: SRAM bandwidth per unit
  - `isa/mxu_overhead.py`: the matrix unit's hidden costs under each design

## Decisions I made (all marked "for your review" in DECISIONS.md)

- **D-026: R1 keeps 4 rows in flight (8 KiB).** You approved this ("4 is ok"). Evidence: 2 in flight 12.7 s, 4 give 11.4 s, 8 give 10.7 s.
- **D-027: a vector command's output equals an input exactly (in place) or doesn't touch it at all.** This is your rule. The assembler and FSim reject partial overlaps, and the lint finds none in the 9B decode, speculative and prefill programs.
- **D-028: the SRAM stays 64 banks × 64-byte lines (4 KB per clock).**
  - Capping the vector unit at 4 KB per clock changes nothing measurable, and even 3 KB costs 0.02 ms per prefill chunk.
  - 128 banks would double the bank periphery for no gain.
  - Measured conflicts: in the tiny decode, one line per bank per clock would add at most 1,071 clocks to 57,443 (1.9%).
- **DMA moves 512 bytes per clock per engine.** That's DRAM's whole bandwidth at 1 GHz. At 64 bytes per clock, the decode was 6× slower than the model.
- **D-029: the matrix unit overlaps everything.**
  - two activation buffers of 128 rows (the next chunk loads during this chunk's beats)
  - two accumulator halves of 8,192 words (a finished tile drains while the next fills)
  - an epilogue of 16 rows × 16 outputs per clock
  - seg-4 loads 4 rows per read
  - The ISA's ACC_WORDS of 4,096 was too small for 128-row prefill; it's now 8,192 per half.
  - Results are bit-identical, because the order of every addition and comparison is unchanged.

## Numbers

- **Tiny decode, 1 token: 57,314 RTL clocks vs 48,045 in the model (1.19×).** The gap:
  - about 6,500 clocks of vector pipeline latency: SFU commands and reductions take 110–145 clocks each beyond their beats, where the model charges 8
  - about 1,000 clocks because simulation runs 1 row per clock instead of 64
  - the lut4 weight stream running at about 87% of the model's rate (found and fixed, see below)
  - about 900 clocks of TAB waiting on the FIFO
  - Against the measured-rate model (`MODEL=rtl`), it's 57,314 vs 51,881 (+10.5%).
    - Vector commands match to 0.2% (11,848 vs 11,869).
    - What's left is the matrix unit waiting on the weight stream (+4,457) and TAB (+1,781). So the weight stream's 87% is the real decode gap.
- **Why the weight stream ran at 87%, and the fix.**
  - The timing model lets the next W command start requesting as soon as the previous one has issued its last request.
  - The RTL's DMA stayed busy until the last byte arrived. So each of the 48 W commands left DRAM idle for one 150-clock latency, about 6,200 clocks in all.
  - **Fixed:** a DRAM-reading engine (W, R) now takes its next command once the last request is issued. Each piece in flight carries a "last of its command" bit, so each command's done (and its semaphores) still comes exactly when its last byte lands, in order. The core keeps up to 4 pending signal sets for W and R (`chip_core.sv`, `chip_dma.sv`).
  - **Tiny decode, 1 token, after the fix: 50,705 clocks, bit-exact** (was 57,314). That's 1.055× the plain model and 2.3% under the measured-rate model. Matrix-unit busy time now matches the model (35,337 vs 36,241).
- **Tiny decode, 3 tokens: 172,283 vs 144,094 (1.20×)** before the fix; **152,069 after it, bit-exact (1.055×)**.
- Smoke (now with a transpose) and mmfuzz seeds 1–3 pass on the new DMA.
- **Tiny prefill chunk (128 tokens), all bit-exact:**
  - first RTL at 8 rows per clock: 2,232,728 clocks
  - at the chip's 64 rows per clock, before D-029: 1,688,640 clocks, bit-exact. D-029 cuts this tiny chunk 1.95×; the tiny model is so vector-heavy that the vector unit hides part of the gain the 9B sees (2.8×).
  - with D-029: 867,985 clocks
- **Why 867,985 and not the model's 238,399: it's the vector unit, and that was already known.**
  - The plain timing model charges vector commands their architectural rate. The RTL-rate model (`isa/bench_rtl_rates.py`, note 32) charges what the vector RTL measurably does.
  - Stacked with D-029's matrix-unit costs, the RTL-rate model predicts 789,565 clocks. The RTL is 9.9% above that.
  - Per command, the vector unit matches the RTL-rate model to 3% (716,340 vs 694,675 busy clocks); RSUMSQ alone matches to 0.2%.
  - So the RTL behaves as the rate model says. The tiny model is just more vector-heavy than the 9B.
- **What the remaining 9.9% was:**
  - About 32,000 clocks were one command: the V → V^T transpose on the L DMA engine. It ran at 1 byte per clock, because a transpose is made of 1-byte pieces and the DMA moved one piece per clock.
  - **Fixed:** the L engine now transposes through a 64 × 64-byte tile at 32 bytes per clock (`chip_dma.sv`). Smoke passes bit-exact with an edge-shaped 70 × 100 transpose (341 clocks; it was 7,000).
  - The timing model now charges transposes at 32 bytes per clock (+0.08% on the 9B prefill).
  - The rerun of the prefill with both DMA fixes (transpose and overlapping commands): **821,127 clocks, bit-exact**, 4.0% above the measured-rate model (was 9.9%).
  - The rest is smaller: the matrix unit (about 35,000 clocks over its model, mostly int8 MMS commands at 1 row per clock in this test) and about 6 clocks per FMA command.
- **9B, from the timing model with every measured RTL cost added:**
  - **Decode at 32K:** 90.0 tokens/s. The plain model says 90.6.
  - **Speculative decode (k = 3):** 296.1 tokens/s, vs 302.0.
  - **32K prefill: 11.7 s.** The plain model says 5.7 s; the vector unit's RTL rates make it 11.4 s; D-029's remaining matrix-unit cost adds 0.3 s. D-008's target (about 12 s) is met.
  - Without D-029 (the serial matrix unit) prefill would be 32.6 s, and speculative decode 253 tokens/s.

## What is not done (honest list)

- **The SRAM is still a behavioural port model.** The banked arbiter (D-028's plan in note 33 part 3) isn't written. Its cost is measured as at most 1.9% on decode. The prefill bank figure from this run: 64 one-line banks would need 1,506,678 clocks for the 583,884 clocks that had SRAM traffic, so in prefill the banks are oversubscribed 2.6× when every unit runs at once. That needs a real arbiter study before layout (see below).
- **The vector unit runs prefill at half its architectural rate** (11.4 s vs 5.7 s for 32K). The levers are measured in `bench_rtl_rates.py`: R1 at one row per clock, QUANT/DEQ 512 wide, and the SMTILE rate. This is the biggest remaining speedup on the chip, and it's a design task rather than a decision.
- **The 2.9 ns activation quantizer** still needs pipelining to 1 GHz (note 32).
- **The matrix unit accumulates in one combinational step.** The chip needs a pipelined adder; the order stays the same, so it's safe.
- **H2S (host-to-SRAM DMA) faults.** There's no host-input model in the RTL or in FSim.
- **No whole-unit synthesis yet.** The area figures are estimates from per-cell numbers in notes 24 and 32.

## Decisions for you (only these three)

- **D-029's area: about 1.2 mm² of ASAP7 cells.**
  - The serial matrix unit is the alternative, and it would make 32K prefill 32.6 s instead of 11.7 s.
  - The split: 0.63 mm² for the activation buffers as flip-flops, 0.15 for the accumulators, 0.45 for 512 epilogue FMAs.
  - The cheaper fallback is 8 rows per drain clock: half the epilogue, about 0.3 s more per 32K prefill.
  - I chose the full width. Tell me if area matters more.
- **Review D-028 (64 SRAM banks), now with the prefill evidence.**
  - Decode conflicts cost at most 1.9%.
  - In prefill, the units together ask for 2.6× what 64 one-line banks serve when every request lands in its own clock. Most of that is the matrix unit's 16-line drain and the vector unit's 512-address beats hitting the same banks.
  - My plan is to write the arbiter and measure the real stall, rather than add banks now. Say if you'd rather I first try 128 banks in the port model, which is a one-line change.
- **Review the descriptor encoding** (`isa/desc_v1.py`, 64 bytes each). It's cheap to change now and expensive after layout.

## How to run things

- `SIM=verilator python3 rtl/chip/run_chip.py smoke`: 6 s plus a 5-minute first build
- `SIM=verilator RP=8 python3 rtl/chip/run_chip.py mmfuzz 3`: about 1 minute
- `SIM=verilator python3 rtl/chip/run_chip.py tiny 1`: about 6 minutes
- `SIM=verilator RP=64 python3 rtl/chip/run_chip.py tinyprefill`: about 75 minutes
- `python3 rtl/chip/compare_timing.py tiny`: RTL vs model, per command
- `MODEL=rtl TAG=_d029 python3 rtl/chip/compare_timing.py tinyprefill`: the prefill run against the measured-rate model
- `python3 isa/mxu_overhead.py`: the matrix-unit design comparison
