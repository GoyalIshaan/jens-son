# 12. Speculative decoding: do we design for it?

*Started 2026-11-10. Status: under consideration. The leaning is to design for it from day one (D-013, proposed). The numbers come from `spec_decode()` in `calc/qwen35_budget.py`. Acceptance rates are assumptions until we measure them.*

## The idea, with small numbers

At batch 1, our chip spends a step reading 4.63 GB (weights, KV and state) to produce **one** token. The multipliers sit mostly idle, waiting for weights.

Speculative decoding uses that idle compute:
1. **Draft:** a cheap predictor guesses the next few tokens, say 3.
2. **Verify:** run the real model once on all of them. Each weight read from DRAM gets multiplied by 4 token vectors instead of 1. The weights are read once, so this step costs about the same as a normal one.
3. **Accept** the guesses up to the first wrong one, plus one token the real model produces itself. So every step yields at least one correct token, and the output is **exactly the same** as without speculation.

If guesses are right 70% of the time and we draft 3, a step yields about 2.5 tokens on average instead of 1.

## Qwen3.5-9B already has a drafter

From the config and model card:
- **`mtp_num_hidden_layers: 1`:** the model ships with one extra multi-token-prediction (MTP) layer, trained alongside it.
- **The card says** the MTP module was "trained with multi-steps", so it can draft several tokens in a row.
- **The card's serving examples** use 2 speculative tokens (vLLM) or 3 speculative steps (SGLang).
- **No acceptance rates are published.** Ours below are assumptions.

So we don't need a separate draft model. That matters, because a separate model would need its own weights in memory and its own reads.

## What a draft costs

We model the MTP module as one full-attention decoder layer plus a projection from 2 × 4096 to 4096, as in Qwen3-Next. That's an assumption to check against the actual weights. At int4:
- **The MTP layer:** 137 MB per draft.
- **But each draft also needs the LM head,** to turn the hidden state into a token: 248,320 × 4096 weights, **572 MB**.
- **Together:** 15.3% of a main pass, for every drafted token.

**The trick:** drafts don't need the full 248K vocabulary. Use only the 32K most common tokens for drafting. That's 75 MB, and a draft costs only **4.6%** of a main pass. If the right token is outside that 32K, the guess is wrong and verification catches it. The output is still exact. Similar trimming is used in published drafting work. Exactly which 32K tokens to use, and how much acceptance it costs, is something we'd measure.

## Speedup, from the script

Expected tokens per step with acceptance rate α and k drafts is (1 − α^(k+1)) ÷ (1 − α). The speedup is that divided by the step cost, 1 + k × (draft cost).

With the **trimmed vocabulary** (each draft 4.6%):
- **α = 0.6:** k=1 gives 1.53×, k=2 gives 1.80×, k=3 gives 1.91×
- **α = 0.7:** k=1 gives 1.63×, k=2 gives **2.01×**, k=3 gives 2.23×
- **α = 0.8:** k=1 gives 1.72×, k=2 gives 2.24×, k=3 gives 2.60×

With the **full vocabulary** (each draft 15.3%), the same α = 0.7 row is 1.47×, 1.68× and 1.74×.

Caveats:
- **The formula assumes every guess is equally likely to be right.** In practice later guesses are worse, so k=3 is optimistic.
- **We ignore the MTP layer's own small KV cache** (about 8 MB at 4K context) and the time to draft. Drafts run one after another, so they add latency.

Even so, **about 2× is plausible.** That's the biggest single lever we've found. It's bigger than any memory or node choice.

## What our hardware gets almost free

- **Verify compute.** At 512 GB/s and 92%, 837 weights arrive per cycle at 1 GHz. Verifying 4 tokens means 4 MACs per weight, 3,350 MACs per cycle. Our prefill floor already needs about 4,096 MACs (D-007). **The prefill array is exactly the verify array.** No extra silicon.
- **The KV cache.** Write entries for all candidate positions, then move the end pointer back if guesses are rejected. Nothing to undo.

## The hard part: rolling back the DeltaNet state

DeltaNet layers keep a running state that each token changes. If we verify 4 tokens and only 2 are accepted, the state must end up as it was after token 2, not after token 4.

Options we considered:
- **Write a checkpoint after every candidate token.** Three extra fp32 states per step is 151 MB, about 3% extra bytes per step. That works, but it's wasteful.
- **Run verification twice.** Doubles the cost. Rejected.
- **Lazy commit (our leaning).**
  - Don't write the new state at all during verification. Keep the old state in DRAM.
  - Save only the small inputs each token would have used to update it: k, v, beta and decay per head. That's 6,208 values per layer per token, about 1.2 MB for 4 tokens across 24 layers.
  - At the start of the next step, when the state is loaded anyway, apply the updates for the tokens that were accepted, then continue.
  - Cost: almost no extra bytes, a little extra compute, which we have spare.
  - The short convolution's last 3 inputs roll back the same way.

The lazy-commit idea works because the DeltaNet update is cheap to replay and its inputs are small, while the state itself is big. **We should check this against how vLLM or SGLang handle Qwen3.5's MTP with linear attention.** That's a fair question for the blog: did we reinvent something?

## Where the accept/reject decision happens

- **On the host:** each step needs a PCIe round trip, a few microseconds. A step takes about 5 to 10 ms, so that's negligible.
- **On the chip:** comparing a few token IDs is trivial. That keeps the chip running on its own, and makes the public ISA a complete loop.

**Leaning:** on chip, with the host told only about accepted tokens.

## Does this make the comparison unfair?

GPUs can do MTP speculation too: vLLM and SGLang support it for Qwen3.5. But llama.cpp, the usual tool for a single user, may not support it for this model yet. Not checked. So we'll report:
- **Score without speculation:** the main D-008 number, directly comparable to the RTX 5060 tables in `07_competitors_and_goal.md`.
- **Score with speculation,** compared against a GPU also running speculation where software supports it.

## What this means for the design now

Things that are cheap to plan for now and expensive to add later:
- **The matrix unit takes up to 4 (maybe 8) token rows** per weight. This is the same as prefill, so it's free.
- **The DeltaNet unit supports lazy commit:** a log buffer, plus "replay n updates" when it loads the state.
- **The KV cache uses an end pointer** that can move back.
- **The ISA has instructions** for compare-and-accept, and for a trimmed LM head (a vocabulary list).

## What to measure

- **The real MTP acceptance rate** for Qwen3.5-9B at int4, for k = 1, 2, 3, on chat-like text and on code.
- **How much the trimmed vocabulary lowers acceptance.**
- **The exact MTP module structure** from the checkpoint's weight names.
