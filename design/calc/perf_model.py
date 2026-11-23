"""Performance model v1 for the Qwen3.5-9B chip (cycle-approximate).  Run:  python3 design/calc/perf_model.py

What it does
- Writes one decode step (or one speculative verify step, or one prefill chunk) as a list of
  operations, in program order. Each op says how many bytes it reads and writes in DRAM and how
  many cycles its compute block needs.
- Runs them on a simple two-resource schedule: one DRAM stream and one compute chain.
  - Ops compute one after another (each layer's output feeds the next).
  - DRAM may fetch ahead of compute by at most `prefetch` bytes (the on-chip buffer). Weights,
    KV and state addresses are all known in advance, so fetch-ahead is always possible.
  - A streaming op (a matrix, attention) finishes when both its bytes have arrived and its
    compute is done. Writes (state, KV append) are batched by the controller: they take DRAM
    time, scaled by `write_penalty` for read/write switching, but compute never waits for them.
- Reports time per token, what limits it, and how busy DRAM is.

Everything not listed is not modelled yet: norms, RoPE, gates and other vector work (small),
instruction issue, DRAM refresh and bank conflicts beyond the flat `util`, and the host link.

Every input is either from qwen35_budget.py (model shapes), a decision (D-xxx), a measurement
(note 15), or an assumption marked ASSUMED.
"""
import math
import os
import sys
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen35_budget as B  # noqa: E402

GB, MB, KB = 1e9, 1e6, 1e3
H, FFN, V = B.HIDDEN, B.FFN, B.VOCAB

# ---------------------------------------------------------------- formats (D-005)
W_BYTES = 9 / 16                    # 4-bit code + one E4M3 byte per 16 weights = 4.5 bits
EMB_BYTES = 1 + 2 / 32              # int8 embedding, fp16 scale per 32 (ASSUMED fp16 scale)
KV_BYTES = 1 + 2 / 256              # int8 KV, one fp16 scale per 256-value head
STATE_BYTES = 4                     # DeltaNet state fp32
CONV_BYTES = 4                      # conv history fp32 (ASSUMED; bf16 would halve it)
MTP_WEIGHTS = 243_269_632           # measured from the checkpoint (note 15, experiment set 4)


@dataclass(frozen=True)
class Chip:
    bw: float = 512 * GB            # 4 x GDDR7 x32 at 32 Gb/s (D-004 leaning)
    util: float = 0.92              # DRAM efficiency on long streams (note 08 estimate)
    clk: float = 1.0e9
    lanes: int = 1024               # weight lanes (D-015)
    rows: int = 4                   # token rows per lane (D-015)
    dn_lanes: int = 256             # DeltaNet fp32 lanes (note 18; D-023 proposes 512)
    replay: bool = False            # count the DeltaNet replay of accepted tokens (note 28)
    exp_per_cycle: int = 64         # softmax exp throughput (ASSUMED)
    prefetch: float = 1 * MB        # bytes DRAM may run ahead of compute (ASSUMED, part of 4 MB SRAM)
    write_penalty: float = 1.15     # DRAM time per written byte vs read byte (ASSUMED, note 18 pt 2)
    state_on_chip: bool = False
    conv_on_chip: bool = True       # 2.4 MB (open item 6)
    kv_bytes: float = KV_BYTES
    ideal: bool = False             # compute free, no writes penalty: the hand-calculation case

    @property
    def bw_eff(self):
        return self.bw * self.util


@dataclass
class Op:
    name: str
    cat: str                        # weights / attention / deltanet / mtp / other
    rd: float = 0.0                 # DRAM bytes read
    wr: float = 0.0                 # DRAM bytes written
    cyc: float = 0.0                # compute cycles on its block


# ---------------------------------------------------------------- the program
def mm(name, n_in, n_out, T, chip, cat="weights"):
    """A weight matrix streamed through the lanes for T token rows. A beat of `lanes` weights is
    held for ceil(T / rows) cycles when T exceeds the rows."""
    w = n_in * n_out
    return Op(name, cat, rd=w * W_BYTES, cyc=math.ceil(w / chip.lanes) * math.ceil(T / chip.rows))


def attention(name, L, T, chip, kv_heads=B.KV_HEADS, q_heads=B.Q_HEADS, cat="attention"):
    """Decode attention over L past positions on the weight lanes: K and V stream like weights
    (int8 straight to the lanes, no table) and the 4 query heads that share each KV head (GQA)
    are the 4 token rows. T tokens need 4*T rows."""
    vals = L * kv_heads * B.HEAD_DIM * 2                       # K and V values
    rep = q_heads // kv_heads
    cyc = math.ceil(vals / chip.lanes) * math.ceil(rep * T / chip.rows)
    cyc += q_heads * T * L / chip.exp_per_cycle               # softmax exponentials
    append = T * kv_heads * B.HEAD_DIM * 2 * chip.kv_bytes
    return Op(name, cat, rd=vals * chip.kv_bytes, wr=append, cyc=cyc)


def deltanet(name, T, chip, replay=0.0):
    """One Gated DeltaNet layer, one-pass order (note 18). The state is read once and written once
    per step, whatever T is (lazy commit for speculation, D-013). Per token: 3 row-operations over
    128 rows x 32 heads, each 128 fp32 elements wide."""
    state = B.LIN_V_HEADS * B.LIN_DIM * B.LIN_DIM * STATE_BYTES
    # replay: tokens accepted last step, re-applied from the log before the new ones (2 passes each; D-013,
    # found by the ISA v1 simulator, note 28)
    elems = (3 * T + 2 * replay) * B.LIN_DIM * B.LIN_DIM * B.LIN_V_HEADS
    io = 0 if chip.state_on_chip else state
    return Op(name, "deltanet", rd=io, wr=io, cyc=elems / chip.dn_lanes)


def conv(name, T, chip):
    ch = 2 * B.LIN_QK_HEADS * B.LIN_DIM + B.LIN_V_HEADS * B.LIN_DIM     # 8192
    hist = 0 if chip.conv_on_chip else ch * (B.CONV_K - 1) * CONV_BYTES
    w = ch * B.CONV_K * W_BYTES                                       # counted as weights, as in the budget
    return Op(name, "other", rd=w + hist, wr=hist, cyc=ch * B.CONV_K * T / chip.dn_lanes)


def decode_ops(L, T, chip, replay=0.0):
    """The main model for T tokens at once (T = 1: plain decode; T = k + 1: speculative verify)."""
    lin_qkvz = 2 * B.LIN_QK_HEADS * B.LIN_DIM + 2 * B.LIN_V_HEADS * B.LIN_DIM
    ops = [Op("embedding rows", "other", rd=T * H * EMB_BYTES)]
    for layer in range(B.LAYERS):
        if (layer + 1) % B.FULL_EVERY == 0:
            ops += [mm(f"L{layer} q+gate", H, 2 * B.Q_HEADS * B.HEAD_DIM, T, chip),
                    mm(f"L{layer} k", H, B.KV_HEADS * B.HEAD_DIM, T, chip),
                    mm(f"L{layer} v", H, B.KV_HEADS * B.HEAD_DIM, T, chip),
                    attention(f"L{layer} attention", L, T, chip),
                    mm(f"L{layer} o", B.Q_HEADS * B.HEAD_DIM, H, T, chip)]
        else:
            ops += [mm(f"L{layer} qkvz", H, lin_qkvz, T, chip),
                    mm(f"L{layer} beta,a", H, 2 * B.LIN_V_HEADS, T, chip),
                    conv(f"L{layer} conv", T, chip),
                    deltanet(f"L{layer} deltanet", T, chip, replay),
                    mm(f"L{layer} out", B.LIN_V_HEADS * B.LIN_DIM, H, T, chip)]
        ops += [mm(f"L{layer} ffn gate+up", H, 2 * FFN, T, chip),
                mm(f"L{layer} ffn down", FFN, H, T, chip)]
    ops.append(mm("LM head", H, V, T, chip))
    return ops


def draft_ops(L, chip, vocab=32_768, mtp_attention=True, window=None):
    """One MTP draft token (T = 1): fc, one gated full-attention layer over its own KV cache of
    length L, SwiGLU, and the LM head over a trimmed vocabulary. 243.3M weights + vocab x 4096."""
    ops = [Op("mtp embedding row", "mtp", rd=H * EMB_BYTES),
           mm("mtp fc", 2 * H, H, 1, chip, "mtp"),
           mm("mtp q+gate", H, 2 * B.Q_HEADS * B.HEAD_DIM, 1, chip, "mtp"),
           mm("mtp k", H, B.KV_HEADS * B.HEAD_DIM, 1, chip, "mtp"),
           mm("mtp v", H, B.KV_HEADS * B.HEAD_DIM, 1, chip, "mtp")]
    if mtp_attention:
        ops.append(attention("mtp attention", L if window is None else min(L, window), 1, chip, cat="mtp"))
    ops += [mm("mtp o", B.Q_HEADS * B.HEAD_DIM, H, 1, chip, "mtp"),
            mm("mtp ffn gate+up", H, 2 * FFN, 1, chip, "mtp"),
            mm("mtp ffn down", FFN, H, 1, chip, "mtp"),
            mm("draft LM head", H, vocab, 1, chip, "mtp")]
    return ops


# ---------------------------------------------------------------- the schedule
def run(ops, chip):
    """Two-resource schedule. Returns seconds and a per-category breakdown."""
    bw = chip.bw_eff
    me = ce = 0.0                        # time DRAM / compute finished so far
    dram_busy = 0.0
    by_cat, bound = {}, {"memory": 0.0, "compute": 0.0}
    for op in ops:
        rd_t = op.rd / bw
        wr_t = op.wr * (1.0 if chip.ideal else chip.write_penalty) / bw
        c = 0.0 if chip.ideal else op.cyc / chip.clk
        ms = max(me, ce - chip.prefetch / bw)        # cannot run further ahead than the buffer
        me = ms + rd_t + wr_t
        cs = max(ce, ms)
        new_ce = max(cs + c, ms + rd_t)              # compute waits for reads, never for writes
        step = new_ce - ce
        ce = new_ce
        dram_busy += rd_t + wr_t
        by_cat[op.cat] = by_cat.get(op.cat, 0.0) + step
        bound["compute" if c > rd_t else "memory"] += step
    total = max(me, ce)
    return dict(secs=total, by_cat=by_cat, bound=bound, dram_busy=dram_busy / total,
                bytes=sum(o.rd + o.wr for o in ops))


def decode(L, chip):
    r = run(decode_ops(L, 1, chip), chip)
    r["tok_s"] = 1 / r["secs"]
    return r


def expected_tokens(acc):
    """acc[j] = P(draft j+1 accepted | drafts 1..j accepted). Tokens per step, including the bonus."""
    e, p = 1.0, 1.0
    for a in acc:
        p *= a
        e += p
    return e


def spec(L, k, chip, acc, vocab=32_768, mtp_attention=True, window=None):
    if k == 0:
        return decode(L, chip)
    ops = decode_ops(L, k + 1, chip, replay=expected_tokens(acc[:k]) if chip.replay else 0.0)
    for _ in range(k):
        ops += draft_ops(L, chip, vocab, mtp_attention, window)
    r = run(ops, chip)
    r["tok_per_step"] = expected_tokens(acc[:k])
    r["tok_s"] = r["tok_per_step"] / r["secs"]
    return r


def prefill(n, chip, chunk=64):
    """Prompt of n tokens in chunks. Each weight beat is held ceil(chunk/rows) cycles. DeltaNet runs
    token by token on its unit (the chunked-parallel form is not modelled). Attention for a chunk
    covers all earlier positions."""
    tot, parts = 0.0, {}
    for start in range(0, n, chunk):
        T = min(chunk, n - start)
        r = run(decode_ops(start + T, T, chip), chip)
        tot += r["secs"]
        for k, v in r["by_cat"].items():
            parts[k] = parts.get(k, 0.0) + v
    return tot, parts


def capacity(L, chip, draft_window=None):
    layers_head = B.P["read_per_token"] * W_BYTES
    emb = B.P["emb"] * EMB_BYTES
    mtp = MTP_WEIGHTS * W_BYTES
    draft_L = L if draft_window is None else min(L, draft_window)
    kv = L * B.KV_VALUES_PER_TOKEN * chip.kv_bytes + draft_L * 2 * B.KV_HEADS * B.HEAD_DIM * chip.kv_bytes  # + MTP KV
    state = B.STATE_VALUES * STATE_BYTES + B.CONV_STATE_VALUES * CONV_BYTES
    return layers_head + emb + mtp + kv + state


# ---------------------------------------------------------------- report
CTXS = (4096, 32_768, 262_144, 1_010_000)
ASSUMED_ACC = (0.900, 0.841, 0.782)  # MEASURED: chip format, 32K draft vocab, mean of 4 sets (note 15, exp set 4)
# MEASURED (note 15 §19): draft-head acceptance with its attention limited to the last W positions,
# divided by full-window acceptance, per draft depth (32K-token text, bf16). Applied to ASSUMED_ACC.
WINDOW_RATIO = {16384: (0.860 / 0.866, 0.806 / 0.808, 0.697 / 0.700),
                4096: (0.855 / 0.866, 0.796 / 0.808, 0.682 / 0.700),
                1024: (0.844 / 0.866, 0.776 / 0.808, 0.660 / 0.700)}
DRAFT_WINDOW = 4096


def window_acc(w):
    return ASSUMED_ACC if w is None else tuple(a * r for a, r in zip(ASSUMED_ACC, WINDOW_RATIO[w]))


def fmt_cat(r):
    t = r["secs"]
    return ", ".join(f"{k} {100 * v / t:.0f}%" for k, v in sorted(r["by_cat"].items(), key=lambda x: -x[1]))


def check_against_hand_numbers():
    print("== 1. Check: ideal case vs qwen35_budget.py hand numbers ==")
    ideal = Chip(ideal=True, kv_bytes=1 + 2 / 64, conv_on_chip=True)
    # same accounting as the budget: no embedding row, no conv history, KV scale per 64, no MTP KV
    wsum = sum(o.rd for o in decode_ops(0, 1, ideal) if o.cat == "weights") / W_BYTES
    wsum += sum(o.rd for o in decode_ops(0, 1, ideal) if o.name.endswith("conv")) / W_BYTES
    print(f"  weights per token in the op list {wsum / 1e9:.4f} B vs budget {B.P['read_per_token'] / 1e9:.4f} B")
    worst = 0.0
    for L in CTXS:
        hand_bytes = (B.P["read_per_token"] * B.BYTES_PER_W["int4"] + L * B.KV_VALUES_PER_TOKEN * (1 + 2 / 64)
                      + 2 * B.STATE_VALUES * 4)
        hand = B.CARD_UTIL * B.CARD_BW / hand_bytes
        ops = [o for o in decode_ops(L, 1, ideal) if o.name != "embedding rows"]
        for o in ops:
            if o.cat == "attention":
                o.wr = 0.0                                   # budget ignores the KV append
        r = run(ops, ideal)
        err = 100 * (1 / r["secs"] - hand) / hand
        worst = max(worst, abs(err))
        print(f"  ctx {L:>9,d}: model {1 / r['secs']:6.2f} tok/s, hand {hand:6.2f} tok/s, error {err:+.3f}%")
    # speculation, 4K, MTP attention off as in the hand calc
    main = B.P["read_per_token"] * B.BYTES_PER_W["int4"] + 4096 * B.KV_VALUES_PER_TOKEN + 2 * B.STATE_VALUES * 4
    draft = (B.mtp_weights() + 32_768 * H) * B.BYTES_PER_W["int4"] / main
    for k in (1, 3):
        e = expected_tokens((0.7,) * k)
        hand = e / (1 + k * draft)
        ideal8 = replace(ideal, kv_bytes=1.0)
        base_ops = [o for o in decode_ops(4096, 1, ideal8) if o.name != "embedding rows"]
        ops = [o for o in decode_ops(4096, k + 1, ideal8) if o.name != "embedding rows"]
        for _ in range(k):
            ops += [o for o in draft_ops(4096, ideal8, mtp_attention=False) if o.name != "mtp embedding row"]
        for o in base_ops + ops:
            if o.cat in ("attention", "mtp"):
                o.wr = 0.0                                   # drop KV appends only, keep state writes
        base = 1 / run(base_ops, ideal8)["secs"]
        sp = e / run(ops, ideal8)["secs"] / base
        worst = max(worst, abs(100 * (sp - hand) / hand))
        print(f"  speculation k={k}, accept 0.7: model x{sp:.3f}, hand x{hand:.3f}")
    pf = Chip()
    w_only = 2048 * B.P["read_per_token"] / (pf.lanes * pf.rows) / pf.clk
    print(f"  prefill 2K, weights only on 1024x4 lanes: {w_only:.2f} s (note 19 quotes 3.97 s)")
    print(f"  worst error {worst:.3f}% -> {'PASS' if worst < 2 else 'FAIL'} (target < 2%)")


def report():
    chip = Chip()
    check_against_hand_numbers()

    print("\n== 2. Base card, plain decode (512 GB/s, 92%, 1 GHz, 1024x4 lanes, 256 DeltaNet lanes) ==")
    for L in CTXS:
        r = decode(L, chip)
        print(f"  ctx {L:>9,d}: {r['tok_s']:6.1f} tok/s  ({1e3 * r['secs']:5.2f} ms, {r['bytes'] / GB:5.2f} GB, "
              f"DRAM busy {100 * r['dram_busy']:.0f}%, compute-bound {100 * r['bound']['compute'] / r['secs']:.0f}%)")
        print(f"      time: {fmt_cat(r)}")

    print(f"\n== 3. Speculative decoding, measured acceptance {ASSUMED_ACC} (chip format, 32K draft vocab) ==")
    for L in CTXS:
        base = decode(L, chip)["tok_s"]
        row = [f"k={k}: {spec(L, k, chip, ASSUMED_ACC)['tok_s']:6.1f} (x{spec(L, k, chip, ASSUMED_ACC)['tok_s'] / base:.2f})"
               for k in (1, 2, 3)]
        print(f"  ctx {L:>9,d}: k=0: {base:6.1f}   " + "   ".join(row))
    r = spec(4096, 3, chip, ASSUMED_ACC)
    print(f"  k=3 at 4K, where the step time goes: {fmt_cat(r)}; compute-bound "
          f"{100 * r['bound']['compute'] / r['secs']:.0f}%")
    for L in (4096, 1_010_000):
        a = draft_ops(L, chip)
        att = [o for o in a if o.name == "mtp attention"][0]
        print(f"  one draft at ctx {L:>9,d}: {sum(o.rd for o in a) / MB:7.0f} MB read "
              f"(of which its own KV {att.rd / MB:6.0f} MB)")

    print("  long context: limiting the draft head's attention to recent positions (acceptance held fixed;"
          " its effect on acceptance is not measured yet), and 8 token rows:")
    for L in (262_144, 1_010_000):
        for nm, c, w in (("full window, 4 rows", chip, None), ("4K window, 4 rows", chip, 4096),
                         ("4K window, 8 rows", replace(chip, rows=8), 4096)):
            rs = [spec(L, k, c, ASSUMED_ACC, window=w) for k in (1, 2, 3)]
            print(f"    ctx {L:>9,d} {nm:20s}: " + "  ".join(
                f"k={k} {r['tok_s']:5.1f} (compute-bound {100 * r['bound']['compute'] / r['secs']:.0f}%)"
                for k, r in zip((1, 2, 3), rs)))

    print(f"  with the draft head windowed to its last {DRAFT_WINDOW} positions (MEASURED acceptance {tuple(round(x, 3) for x in window_acc(DRAFT_WINDOW))}):")
    for L in CTXS:
        rows = []
        for nm, c in (("4 rows", chip), ("8 rows", replace(chip, rows=8))):
            rs = [spec(L, k, c, window_acc(DRAFT_WINDOW), window=DRAFT_WINDOW)["tok_s"] for k in (1, 2, 3)]
            best = max(range(3), key=lambda i: rs[i])
            rows.append(f"{nm}: " + " ".join(f"k={k} {r:5.1f}" for k, r in zip((1, 2, 3), rs)) + f" (best k={best + 1})")
        print(f"    ctx {L:>9,d}: k=0 {decode(L, chip)['tok_s']:5.1f} | " + " | ".join(rows))

    print("\n== 4. Sweeps (tok/s; 'k=3' uses the measured acceptance) ==")
    print("  DeltaNet lanes:")
    for n in (128, 256, 512):
        c = replace(chip, dn_lanes=n)
        print(f"    {n:3d} lanes: 4K k=0 {decode(4096, c)['tok_s']:6.1f}, k=1 {spec(4096, 1, c, ASSUMED_ACC)['tok_s']:6.1f}, "
              f"k=3 {spec(4096, 3, c, ASSUMED_ACC)['tok_s']:6.1f}")
    print("  token rows per lane (verify width):")
    for n in (2, 4, 8):
        c = replace(chip, rows=n)
        print(f"    {n} rows: 4K k=1 {spec(4096, 1, c, ASSUMED_ACC)['tok_s']:6.1f}, "
              f"k=3 {spec(4096, 3, c, ASSUMED_ACC)['tok_s']:6.1f}, prefill 2K {prefill(2048, c)[0]:.2f} s")
    print("  clock:")
    for f in (0.8e9, 1.0e9, 1.2e9):
        c = replace(chip, clk=f)
        print(f"    {f / 1e9:.1f} GHz: 4K k=0 {decode(4096, c)['tok_s']:6.1f}, k=3 {spec(4096, 3, c, ASSUMED_ACC)['tok_s']:6.1f}, "
              f"prefill 2K {prefill(2048, c)[0]:.2f} s, weights/cycle delivered {c.bw_eff / W_BYTES / f:5.0f}")
    print("  memory:")
    for nm, bw in (("LPDDR5X 256-bit 273 GB/s", 273 * GB), ("GDDR7 4 x32 512 GB/s", 512 * GB)):
        c = replace(chip, bw=bw)
        print(f"    {nm:26s}: 4K {decode(4096, c)['tok_s']:6.1f}, 32K {decode(32768, c)['tok_s']:6.1f}, "
              f"1M {decode(1_010_000, c)['tok_s']:5.1f}, 4K k=3 {spec(4096, 3, c, ASSUMED_ACC)['tok_s']:6.1f}")
    print("  prefetch buffer (how far DRAM may run ahead of compute):")
    for pb in (0.0, 0.25 * MB, 1 * MB, 4 * MB):
        c = replace(chip, prefetch=pb)
        print(f"    {pb / MB:4.2f} MB: 4K k=0 {decode(4096, c)['tok_s']:6.1f}, k=3 {spec(4096, 3, c, ASSUMED_ACC)['tok_s']:6.1f}")
    print("  write penalty (DRAM read/write switching):")
    for wp in (1.0, 1.15, 1.5, 2.0):
        c = replace(chip, write_penalty=wp)
        print(f"    x{wp:4.2f}: 4K {decode(4096, c)['tok_s']:6.1f}, 1M {decode(1_010_000, c)['tok_s']:5.1f}")
    print("  where the DeltaNet state and conv history live:")
    for nm, c in (("state DRAM, conv on chip", chip), ("state DRAM, conv DRAM", replace(chip, conv_on_chip=False)),
                  ("state on chip (50 MB SRAM)", replace(chip, state_on_chip=True))):
        print(f"    {nm:28s}: 4K {decode(4096, c)['tok_s']:6.1f}")
    print("  draft vocabulary (4K, k=3):")
    for vs in (16_384, 32_768, 65_536, V):
        print(f"    {vs:7,d}: {spec(4096, 3, chip, ASSUMED_ACC, vocab=vs)['tok_s']:6.1f} tok/s "
              f"(acceptance held fixed; trimming lowers it, measured in experiment set 4)")
    print("  long-context KV format (int4 g32 opt-in):")
    c4 = replace(chip, kv_bytes=0.5 + 2 / 32)
    for L in (262_144, 1_010_000):
        print(f"    ctx {L:>9,d}: int8 {decode(L, chip)['tok_s']:5.1f}, int4 {decode(L, c4)['tok_s']:5.1f} tok/s")

    print("\n== 5. Prefill (chunks of 64 tokens) ==")
    for n in (2048, 8192, 32_768):
        t, parts = prefill(n, chip)
        print(f"  {n:6d} tokens: {t:7.2f} s   " + ", ".join(f"{k} {v:.2f} s" for k, v in
                                                       sorted(parts.items(), key=lambda x: -x[1])))
    for nm, c in (("1.25x lanes (1280x4)", replace(chip, lanes=1280)), ("8 rows (1024x8)", replace(chip, rows=8)),
                  ("512 DeltaNet lanes", replace(chip, dn_lanes=512)),
                  ("8 rows + 512 DeltaNet lanes", replace(chip, rows=8, dn_lanes=512))):
        print(f"  2K with {nm:28s}: {prefill(2048, c)[0]:.2f} s")

    print("\n== 6. Capacity (weights incl. embedding + MTP head, KV int8 incl. MTP KV, state) ==")
    for L in CTXS:
        need = capacity(L, chip) / GB
        need4 = capacity(L, replace(chip, kv_bytes=0.5 + 2 / 32)) / GB
        needw = capacity(L, chip, DRAFT_WINDOW) / GB
        fit = lambda x: next((f"{gb} GB" for gb in (8, 16, 24) if x + 0.5 <= gb), "none of 8/16/24 GB")
        print(f"  ctx {L:>9,d}: {need:6.2f} GB with int8 KV ({need4:5.2f} GB int4); {needw:6.2f} GB with int8 KV and a "
              f"{DRAFT_WINDOW}-position draft window  fits: {fit(need)} / {fit(needw)} (with 0.5 GB working space)")


if __name__ == "__main__":
    report()
