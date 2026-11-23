# Open questions

Things we don't know yet, and how we plan to find out. When one is answered, move the answer into the right note and record the date here.

## About the model

- **Does Qwen3.5-9B keep its quality at int4 weights and int8 activations?** Find out by building the golden model and measuring perplexity plus a few benchmarks against the floating-point model.
- **Can the DeltaNet state be bf16 instead of fp32?** The config says `mamba_ssm_dtype` float32. Test in the golden model.
- **What are the exact DeltaNet update equations?** Read the reference implementation in Hugging Face transformers and write them down step by step with small numbers.
- **Does Qwen3.5-0.8B use the same layer types and pattern?** Check its config.json.
- ~~**Is the MTP head worth using?**~~ **Answered 2026-11-18:** yes. 3.26 tokens per step at k = 3, which gives ×2.73 at 4K on the base card (note 15 §15, note 21).

## About the workload

- **Is 1 s for a 2K-token prompt the right prefill target?** It sets the matrix unit's size. Look at what agents actually send: typical prompt lengths and how often they change.
- **How often would batch 2 to 8 actually happen** in a single-person agent setup?

## About the memory system

- **Real LPDDR5X timing values** from a cited source.
- **Which DRAM simulator** (DRAMsim3 or Ramulator 2) and how to connect it to Verilator.
- **Energy per byte** for LPDDR5X vs on-chip SRAM, to decide whether keeping state on chip is worth it for energy.

## About the architecture

- **One matrix unit for both prefill and decode, or two?**
- **How to pick the next token from 248K logits** without storing all of them.
- ~~**How many slots per VLIW bundle,** and what the instruction format looks like.~~ Answered by ISA v1 (note 28, D-022): 4 slots of coarse commands. Open: the prefill program, SRAM bank conflicts.
- **Does the old `tile_engine` handle back-to-back tiles of different sizes correctly?** Write a test for it.

## About cost (needed for the tokens-per-dollar score)

- **Memory prices:** LPDDR5X per package and per GB, GDDR7 per chip, both cited and dated. DRAM prices moved a lot in 2025 and 2026, so the date matters.
- **Silicon cost:** wafer price for a 5 nm or 6 nm class node, die yield, and the resulting cost per mm².
- **PHY cost:** area and power for an LPDDR5X vs a GDDR7 PHY.
- **Package and board cost** for each memory choice.
- **Competitors:** first pass done 2026-11-10 (`07_competitors_and_goal.md`). Still needed: measured Qwen3.5-9B tok/s on real hardware (at least your Mac, ideally an RTX 5060), and November prices.
- **The prefill floor:** what prompt lengths and waiting times does local use actually have?

## About manufacturing (from 08)

- **Which process nodes have a licensable GDDR7 PHY,** and how big is it?
- **Mask set cost by node,** from a cited source.
- **The RTX 5060's die size and node,** and an estimate of its bill of materials.
- **Real die area,** from synthesis plus IP vendor PHY sizes.

## About chip infrastructure (from 09)

- **Core clock:** 800 MHz or 1 GHz?
- **The GDDR7 controller interface width** and clock ratio for the PHY we'd license.
- **Is a simple bus enough** for control traffic?

## About the instruction set

- **Which general features are free** (any shape, loops, strided DMA), and which cost real area (for example floating-point matrix math)?
- **How do we document and version the instruction set** so outside users and the RL model can target it?

## About claims for the blog

- **What do Etched and Taalas publicly claim** about single-architecture or single-model chips? Cite before using.
- **How fast does an M4 Pro Mac actually run Qwen3.5-9B at 4-bit?** Measure it, to use as our comparison point.

## From notes 10 to 14 (2026-11-10)

- **Formats:** quality of int4 with a scale per 32 vs per 128, MXFP4, FFN at 3 bits, and W4A8 vs W4A16, measured in the golden model. Does Qwen publish FP8 or GPTQ int4 checkpoints we could start from?
- **Long context:** does int4 KV or bf16 state hurt quality at 128K to 1M? How much does static YaRN hurt short-context quality (the model card warns it can)?
- **GDDR7:** does the PHY and controller IP we could license support clamshell (x16) mode? What do 3 GB chips cost?
- **SRAM:** the real density on our node. The 0.5 mm² per MB we used is a rough guess.
- ~~**MTP:** the real acceptance rate for k = 1, 2, 3; the exact MTP module structure; the cost of a trimmed 32K vocabulary.~~ **Answered 2026-11-18** (note 15 §15): 90.0 / 84.1 / 78.2% conditional acceptance. The structure was read from the checkpoint and its wiring measured. 32K costs 0.12 tokens per step and is the fastest overall.
- **Rollback:** how vLLM and SGLang roll back DeltaNet state during speculation. Is lazy commit already known?
- **llama.cpp:** does it support Qwen3.5 MTP speculation? This affects the fair comparison.
- **Prefill:** do we keep the 4 s floor for 2K tokens, given the 1,024 × 4 sketch only just meets it?
- **Tapeout services:** current prices and process options for Tiny Tapeout, wafer.space, IHP shuttles and ChipFoundry.
- **FPGA:** which board (at least 8 GB DDR) we'd use for a slow, real-hardware demo, and its price.

## From the measurements (2026-11-11, `15_measurements.md`)

- **Calibration-based quantizers:** do GPTQ or AWQ, run on our lookup-table formats, push code KL below 0.10 and bring code perplexity under +13%? Could they make 3-bit FFN usable?
- **Task benchmarks** (code generation, maths, chat), not just perplexity and KL.
- **KV and state error far beyond 8K:** 32K, 128K and 262K tokens, especially for int4 KV.
- **A real RTX 5060 measurement,** and an A10G run (Beam put the A10G job on an RTX 4090).
- **llama.cpp's MTP speculation for Qwen3.5,** for the fair comparison with speculation on.

## From note 18 (DeltaNet unit and memory traffic, 2026-11-11)

- **DeltaNet unit width:** 128, 256 or 512 fp32 lanes, depending on how many speculative tokens are verified.
- **One-pass output trick:** check its fp32 accuracy in the golden model.
- **Read/write switching:** how much time do DeltaNet and KV writes cost the weight stream, and how much do write batching and spreading writes across chips recover? Measure in the performance model.
- **Read-only weight region and bank partitioning:** how many banks per chip to reserve for written data, and the resulting row-hit rate for the weight stream.
- **Conv history:** keep it on chip (2.4 MB) or stream it.

## From experiment set 2 (2026-11-11)

- ~~**Table entries at 8-bit precision**~~: answered 2026-11-12. int8 is free (note 15, section 12).
- **Mixed precision by depth:** later FFN layers tolerated 3 bits best (layers 24–31: +3.6% on their own). Would GPTQ on 3-bit FFN in the last 8 layers only be worth about 3% more speed?
- ~~**Task benchmarks** for the final format~~: answered 2026-11-12. GSM8K −1.9 points; no change on HumanEval or MBPP (note 15, section 13).
- **KV and state at 128K to 262K.** 128K answered 2026-11-18: int8 KV is free (KL 0.0002), int4 holds (KL 0.016, falling with distance). 262K is still open.

## From experiment set 3 (2026-11-12)

- **The 1.9-point GSM8K gap:** would a short quantization-aware finetune into the same format close it, at no byte cost? Or would 5-bit codes for DeltaNet alone (about +2% bytes)?
- **Chat and multilingual quality** for the chip format (not yet tested).
- **Table stats for E4M3 and int5 tables:** the stats code skipped them (whole-number first entry); fix it if the numbers are ever needed.

## From experiment set 4 and the performance model (2026-11-18)

- ~~**Windowed draft head**~~ **Answered 2026-11-18** (note 15 §19): a 4K window loses 1.8% of tokens per step at 32K. Adopted in D-013. Still open: the loss at 1M itself.
- **Acceptance under sampling** (temperature 0.7–1.0) and on text people write, not the model's own greedy output.
- **Acceptance for drafts 4 and beyond:** is k = 4 better than 3 at short context?
- **8 token rows instead of 4:** the area and power cost from open-PDK synthesis. It meets the prefill floor (2.31 s for 2K) and makes long-context verify work.
- **Special-function unit width:** at long context, the softmax exponentials for k + 1 verify tokens make attention compute-bound with 64 per cycle.
- ~~**1M context on the 24 GB card**~~ **Answered 2026-11-18:** with the 4K draft window it needs 22.4 GB with int8 KV, so it fits.

## From progression step 1 and experiment set 5 (2026-11-20)

- **Power-of-two activation scales on a real task:** KL shows no change with the chip's weights (+0.0002 ±0.0006, note 25). A paired GSM8K run would confirm it before it becomes a decision (D-005, D-017).
- **Power of the matrix unit at 32 and 64 rows:** needs place-and-route and a power tool (OpenROAD/OpenSTA). It decides 64 rows (D-015). **User, 2026-11-20: later.** 64 rows stays proposed until then.
- **The beat reducer** (64 group results → 1 per row) and the accumulators: not built yet. Its size depends on the activation-scale decision.

## Answered by the user, 2026-11-20

- **Decoding mode:** greedy is the main use. The draft-acceptance numbers (note 15 §15, §19) were measured with greedy, so they apply directly; no sampling check is needed for step 2. On-chip sampling (top-k, temperature) stays supported but is not the design point.
- **Power tools (Docker + OpenROAD):** later.
- **Blog:** not publishing yet. Notes keep being written blog-ready.

## Waiting on the user (2026-11-21, after the overnight run)

- ~~**Approve or change D-022, D-023 and D-024.**~~ Approved by the user 2026-11-21. 1,024 vector lanes: on hold (user).
- ~~**Prefill's DeltaNet:** sequential or chunked?~~ The user said to try the chunked delta rule (2026-11-21). Done, note 30. After the three-way comparison (part 2), the no-new-hardware version: quality within noise, attention, the FFN and decode unchanged, a 32K prompt in 5.66 s. **Approve the revised D-025?**
- **1,024 vector lanes** (32K prefill 5.45 s instead of 6.85 s, about +0.2 mm²): only if prefill speed matters beyond the D-008 target.
