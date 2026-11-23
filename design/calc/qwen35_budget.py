"""Budget calculator for a Qwen3.5-9B inference chip.

Every number quoted in the design notes comes from this file, so a reader
can rerun it and check our arithmetic. Run:  python3 design/calc/qwen35_budget.py

Config values are copied from Qwen/Qwen3.5-9B config.json (text model only).
"""

GB = 1e9
MB = 1e6
KB = 1e3

# ---------------------------------------------------------------- the model
HIDDEN = 4096
LAYERS = 32
FULL_EVERY = 4                     # full_attention_interval
N_FULL = LAYERS // FULL_EVERY      # 8 gated full-attention layers
N_LIN = LAYERS - N_FULL            # 24 Gated DeltaNet layers
FFN = 12288                        # intermediate_size, SwiGLU (3 matrices)
VOCAB = 248_320

# full attention
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 256
# linear attention (Gated DeltaNet)
LIN_QK_HEADS, LIN_V_HEADS, LIN_DIM, CONV_K = 16, 32, 128, 4


def params():
    """Weight counts per block (biases and norms ignored: they are tiny)."""
    q = HIDDEN * Q_HEADS * HEAD_DIM * 2          # x2: query plus output gate
    kv = HIDDEN * KV_HEADS * HEAD_DIM * 2        # K and V
    o = Q_HEADS * HEAD_DIM * HIDDEN
    full = q + kv + o

    qk = HIDDEN * LIN_QK_HEADS * LIN_DIM * 2     # q and k
    v = HIDDEN * LIN_V_HEADS * LIN_DIM
    z = HIDDEN * LIN_V_HEADS * LIN_DIM           # output gate
    ab = HIDDEN * LIN_V_HEADS * 2                # beta and decay (a) per head
    out = LIN_V_HEADS * LIN_DIM * HIDDEN
    conv = (2 * LIN_QK_HEADS + LIN_V_HEADS) * LIN_DIM * CONV_K
    lin = qk + v + z + ab + out + conv

    ffn = 3 * HIDDEN * FFN
    layers = N_FULL * (full + ffn) + N_LIN * (lin + ffn)
    emb = VOCAB * HIDDEN
    head = VOCAB * HIDDEN                        # untied LM head
    return dict(full=full, lin=lin, ffn=ffn, layers=layers, emb=emb, head=head,
                total=layers + emb + head,
                read_per_token=layers + head)    # embedding is one row lookup


P = params()

# bytes per weight, including per-group scales (int4: one fp16 scale per 32)
BYTES_PER_W = {"bf16": 2.0, "int8": 1.0 + 2 / 128, "int4": 0.5 + 2 / 32}

KV_VALUES_PER_TOKEN = N_FULL * KV_HEADS * HEAD_DIM * 2        # K and V
STATE_VALUES = N_LIN * LIN_V_HEADS * LIN_DIM * LIN_DIM         # DeltaNet S
CONV_STATE_VALUES = N_LIN * (2 * LIN_QK_HEADS + LIN_V_HEADS) * LIN_DIM * (CONV_K - 1)


def flops_per_token(ctx):
    """2 ops per MAC. Attention scores and weighted sum over ctx past tokens."""
    mm = 2 * P["read_per_token"]
    attn = N_FULL * Q_HEADS * HEAD_DIM * ctx * 2 * 2
    delta = N_LIN * LIN_V_HEADS * LIN_DIM * LIN_DIM * 2 * 4    # ~4 passes over S
    return mm + attn + delta


# -------------------------------------------------------------- memories
MEMORIES = [
    ("LPDDR5X-8533, 128-bit", 136.5 * GB),
    ("LPDDR5X-8533, 256-bit (M4 Pro class)", 273.0 * GB),
    ("LPDDR5X-8533, 512-bit (M4 Max class)", 546.0 * GB),
    ("GDDR7 32 Gb/s, 256-bit", 1024.0 * GB),
    ("HBM3E, one stack", 1229.0 * GB),
]


def decode_step(batch, ctx, bw, wfmt="int4", state_bytes=2, kv_bytes=1,
                state_on_chip=False):
    """One decode step serves `batch` users one token each.

    Weights are read once per step and shared by every user.  Each user also
    reads its own KV cache (ctx tokens) and reads + writes its DeltaNet state.
    Returns (bytes moved, seconds per step, tokens/s per user, tokens/s total).
    """
    w = P["read_per_token"] * BYTES_PER_W[wfmt]
    kv = ctx * KV_VALUES_PER_TOKEN * kv_bytes
    st = 0 if state_on_chip else 2 * STATE_VALUES * state_bytes
    total = w + batch * (kv + st)
    t = total / bw
    return total, t, 1 / t, batch / t


def main():
    print("== Qwen3.5-9B weights ==")
    for k in ("full", "lin", "ffn"):
        print(f"  {k:5s} block           {P[k]/1e6:8.1f} M")
    print(f"  all 32 layers        {P['layers']/1e9:8.3f} B")
    print(f"  embedding / LM head  {P['emb']/1e9:8.3f} B each")
    print(f"  total                {P['total']/1e9:8.3f} B")
    print(f"  read per token       {P['read_per_token']/1e9:8.3f} B")
    ffn_share = 32 * P["ffn"] / P["read_per_token"]
    print(f"  FFN share of reads   {ffn_share*100:8.1f} %")
    for f, b in BYTES_PER_W.items():
        print(f"  bytes/token at {f:5s} {P['read_per_token']*b/GB:8.2f} GB")

    print("\n== Per-user state ==")
    print(f"  KV per token         {KV_VALUES_PER_TOKEN} values = {KV_VALUES_PER_TOKEN/KB:.1f} KB int8")
    for ctx in (4096, 32768, 131072):
        print(f"  KV at ctx {ctx:6d}     {ctx*KV_VALUES_PER_TOKEN/MB:8.1f} MB int8")
    print(f"  DeltaNet state       {STATE_VALUES/1e6:.2f} M values = "
          f"{STATE_VALUES*4/MB:.1f} MB fp32, {STATE_VALUES*2/MB:.1f} MB bf16")
    print(f"  conv state           {CONV_STATE_VALUES/1e3:.0f} K values")

    print("\n== Compute per token ==")
    for ctx in (0, 4096, 32768):
        print(f"  ctx {ctx:6d}  {flops_per_token(ctx)/1e9:6.1f} GFLOP")

    print("\n== Batch-1 decode ceiling (int4 weights, ctx 4096, bf16 state) ==")
    for name, bw in MEMORIES:
        _, _, per_user, _ = decode_step(1, 4096, bw)
        print(f"  {name:40s} {bw/GB:7.1f} GB/s  {per_user:6.1f} tok/s")

    print("\n== Batching on LPDDR5X 256-bit (273 GB/s), int4, bf16 state ==")
    bw = 273.0 * GB
    for ctx in (4096, 32768):
        print(f"  context {ctx}")
        for b in (1, 2, 4, 8, 16, 32, 64):
            total, t, per_user, agg = decode_step(b, ctx, bw)
            tops = agg * flops_per_token(ctx) / 1e12
            macs = agg * flops_per_token(ctx) / 2 / 1e9     # MACs/cycle at 1 GHz
            print(f"    batch {b:3d}: {total/GB:6.2f} GB/step  "
                  f"{per_user:6.1f} tok/s/user  {agg:7.1f} tok/s total  "
                  f"{tops:5.2f} TOPS  {macs:7.0f} MAC/cycle@1GHz")

    print("\n== Keeping the DeltaNet state on chip (batch 1, ctx 4096, 273 GB/s) ==")
    for sb, name in ((4, "fp32 in DRAM"), (2, "bf16 in DRAM")):
        _, _, u, _ = decode_step(1, 4096, bw, state_bytes=sb)
        print(f"  {name:14s} {u:6.2f} tok/s")
    _, _, u, _ = decode_step(1, 4096, bw, state_on_chip=True)
    print(f"  on-chip SRAM   {u:6.2f} tok/s  (needs {STATE_VALUES*2/MB:.0f} MB bf16 SRAM)")

    print("\n== Prefill ==")
    for n, secs in ((2048, 1.0), (8192, 2.0), (32768, 10.0)):
        ops = n * flops_per_token(n // 2)        # average past length n/2
        macs = ops / 2 / secs / 1e9
        side = macs ** 0.5
        print(f"  {n:6d} tokens in {secs:4.1f} s: {ops/1e12:6.1f} TOP total, "
              f"{ops/secs/1e12:6.1f} TOPS, {macs:7.0f} MAC/cycle@1GHz (~{side:.0f}x{side:.0f})")

    print("\n== Little's law: bytes in flight to keep DRAM busy ==")
    for name, bw_ in MEMORIES[:2]:
        for lat_ns in (100, 150):
            inflight = bw_ * lat_ns * 1e-9
            print(f"  {name:40s} {lat_ns} ns: {inflight/KB:6.1f} KB = "
                  f"{inflight/64:5.0f} x 64 B requests")

    print("\n== Memory built from real parts (batch 1, ctx 32768 must fit) ==")
    # (name, parts, GB/s per part, GB per part options)
    # LPDDR5X-8533 package: x64 = 68.3 GB/s.  GDDR7 32 Gb/s chip: x32 = 128 GB/s.
    need = (P["total"] * BYTES_PER_W["int4"]      # all stored weights, incl. embedding
            + 32768 * KV_VALUES_PER_TOKEN + STATE_VALUES * 2) / GB
    print(f"  capacity needed at 32K context: {need:.2f} GB (+ working space)")
    parts = [("LPDDR5X x64 package", 68.27, (8, 16)),
             ("GDDR7 x32 chip, 32 Gb/s", 128.0, (2, 3))]
    for name, bw_part, caps in parts:
        for n in (1, 2, 4, 8):
            bw_ = n * bw_part * GB
            _, _, u, _ = decode_step(1, 4096, bw_)
            cap = ", ".join(f"{n*c} GB" for c in caps)
            fits = "fits" if n * max(caps) >= need + 0.5 else "too small"
            print(f"  {n} x {name:24s} {n*bw_part:7.1f} GB/s  {u:6.1f} tok/s  "
                  f"capacity {cap} ({fits})")

    competitors()
    die_costs()
    our_card()
    number_formats()
    long_context()
    sram_budget()
    spec_decode()
    shapes()

    print("\n== Simulation cost ==")
    cyc = P["read_per_token"] * BYTES_PER_W["int4"] / 136.5   # bytes per cycle at 1 GHz
    print(f"  cycles per 9B token at 136.5 B/cycle: {cyc/1e6:.1f} M")
    for rate in (1e3, 1e4, 1e5):
        print(f"  at {rate:8.0f} simulated cycles/s: {cyc/rate/3600:7.2f} hours per token")


# ------------------------------------------------------- competitors (2026)
# Measured: llama.cpp llama-bench tg128 on Llama 2 7B Q4_0 (file 3.56 GiB).
# CUDA numbers: github.com/ggml-org/llama.cpp/discussions/15013 (FA on).
# Apple numbers: github.com/ggml-org/llama.cpp/discussions/4167.
# B580: Vulkan scoreboard figure as relayed by a secondary write-up.
# Prices: latest street prices we could find (mostly Aug 2026), see 07 note.
LLAMA2_Q4_0_BYTES = 3.56 * 2**30          # we treat the whole file as read per token
QWEN_LLAMACPP_BYTES = (P["read_per_token"] * BYTES_PER_W["int4"]   # Q4_0 = 4.5 bits
                       + 2 * STATE_VALUES * 4                     # fp32 state, r+w
                       + 4096 * KV_VALUES_PER_TOKEN * 2)          # f16 KV, 4K ctx
COMPETITORS = [
    # name, price USD, GB/s, measured tg128 or None, kind, memory GB
    ("RTX 5060 8GB",            300, 448, 100.61, "card", 8),
    ("Arc B580 12GB",           310, 456, 72.0,   "card", 12),
    ("RTX 3060 12GB (used)",    300, 360, 76.92,  "card", 12),
    ("RTX 5060 Ti 16GB",        805, 448, 93.46,  "card", 16),
    ("Jetson Orin Nano Super",  399, 102, None,   "system", 8),
    ("Mac mini M4 16GB",        799, 120, 24.11,  "system", 16),
    ("Mac mini M4 Pro 24GB",   1599, 273, 49.64,  "system", 24),
]


def competitors():
    print("\n== Competitors at batch 1 ==")
    print(f"  Qwen3.5-9B bytes/token as llama.cpp runs it (Q4_0-like, 4K ctx): "
          f"{QWEN_LLAMACPP_BYTES/GB:.2f} GB")
    for name, price, bw, tg, kind, mem in COMPETITORS:
        if tg is None:
            util, note = 0.70, "assumed"
        else:
            util, note = tg * LLAMA2_Q4_0_BYTES / (bw * GB), "measured"
        qwen = util * bw * GB / QWEN_LLAMACPP_BYTES
        print(f"  {name:24s} ${price:5d} {bw:4d} GB/s  util {util*100:4.0f}% ({note})  "
              f"Qwen3.5-9B est {qwen:6.1f} tok/s  {qwen/price:6.3f} tok/s per $  [{kind}]")


# ------------------------------------------------------------- die cost
WAFER_PRICE = {   # USD per 300 mm wafer, rough 2024-2026 analyst estimates (see 08 note)
    "TSMC N16/N12": 5500, "TSMC N7/N6": 9500, "TSMC N5/N4": 18500, "TSMC N3": 20000,
}
D0 = {            # defects per cm^2, assumed for mature processes
    "TSMC N16/N12": 0.05, "TSMC N7/N6": 0.07, "TSMC N5/N4": 0.08, "TSMC N3": 0.10,
}


def dies_per_wafer(area_mm2, diameter=300.0):
    import math
    r = diameter / 2
    return int(math.pi * r * r / area_mm2 - math.pi * diameter / math.sqrt(2 * area_mm2))


def die_cost(area_mm2, node):
    import math
    dpw = dies_per_wafer(area_mm2)
    a_cm2 = area_mm2 / 100
    yld = ((1 - math.exp(-a_cm2 * D0[node])) / (a_cm2 * D0[node])) ** 2   # Murphy
    return WAFER_PRICE[node] / (dpw * yld), dpw, yld


def die_costs():
    print("\n== Cost per good die (wafer price / (dies per wafer x yield)) ==")
    for node in WAFER_PRICE:
        row = []
        for area in (40, 80, 150, 300):
            c, dpw, y = die_cost(area, node)
            row.append(f"{area:3d} mm2 ${c:6.2f} ({dpw} dies, {y*100:.0f}%)")
        print(f"  {node:13s} " + " | ".join(row))


def our_card():
    """First-pass unit cost of our card. Every input is an assumption (see 08 note)."""
    print("\n== Our card, first pass (assumptions, not quotes) ==")
    for mem_per_gb in (8.0, 12.0, 15.0):          # USD/GB; LPDDR5X contract ~12.2 in Q2 2026
        for util in (0.86, 0.92):
            gb, bw = 8, 4 * 128.0                  # 4 x 2 GB GDDR7 at 32 Gb/s
            die, _, _ = die_cost(80, "TSMC N7/N6")
            package, board = 15.0, 35.0
            unit = gb * mem_per_gb + die + package + board
            _, t, _, _ = decode_step(1, 4096, bw * GB)
            tok = util / t
            retail = 2.0 * unit                    # assumed retail = 2 x unit cost
            print(f"  mem ${mem_per_gb:4.1f}/GB util {util*100:3.0f}%: unit ${unit:6.1f} "
                  f"(memory ${gb*mem_per_gb:5.1f}, die ${die:4.1f})  {tok:6.1f} tok/s  "
                  f"retail-equiv ${retail:5.0f}  {tok/retail:5.3f} tok/s per $")


# ------------------------------------------------ 10: number formats
# bits per weight including the shared scale for each group
FORMATS = [
    ("int4, fp16 scale per 32 (our default)", 4 + 16 / 32),
    ("int4, fp16 scale per 128", 4 + 16 / 128),
    ("MXFP4 (E2M1, 8-bit power-of-2 scale per 32)", 4 + 8 / 32),
    ("NVFP4 (E2M1, FP8 scale per 16)", 4 + 8 / 16),
    ("mixed: FFN 3-bit + rest 4-bit, scale per 32", None),
    ("int8, fp16 scale per 128", 8 + 16 / 128),
    ("bf16", 16),
]
CARD_BW = 512 * GB          # 4 x GDDR7 x32 at 32 Gb/s
CARD_UTIL = 0.92


def ffn_weights():
    return (N_FULL + N_LIN) * P["ffn"]


def number_formats():
    print("\n== Weight formats: bytes per token and speed (512 GB/s, 92%, ctx 4096) ==")
    rest = 2 * STATE_VALUES * 4 + 4096 * KV_VALUES_PER_TOKEN * 1   # fp32 state r+w, int8 KV
    for name, bits in FORMATS:
        if bits is None:
            ffn = ffn_weights()
            wbytes = ffn * (3 + 16 / 32) / 8 + (P["read_per_token"] - ffn) * 4.5 / 8
            bits = wbytes * 8 / P["read_per_token"]
        wbytes = P["read_per_token"] * bits / 8
        tok = CARD_UTIL * CARD_BW / (wbytes + rest)
        print(f"  {name:46s} {bits:5.2f} bits  {wbytes/GB:5.2f} GB/token  {tok:6.1f} tok/s")

    print("\n== Accumulator widths (signed, worst case) ==")
    w_max, a_max = 8, 127                      # int4 is -8..7, int8 activations -127..127
    prod = w_max * a_max
    bits = lambda v: v.bit_length() + 1
    print(f"  int4 x int8 product   max |{prod}|        -> {bits(prod)} bits")
    print(f"  sum of 32 (one group) max |{32*prod}|      -> {bits(32*prod)} bits")
    for k in (HIDDEN, FFN):
        print(f"  sum of {k:5d} raw ints max |{k*prod}|  -> {bits(k*prod)} bits")
    print("  MXFP4 values x2 are integers in -12..12 -> product with int8 fits in "
          f"{bits(12*127)} bits; the shared scale is a shift, not a multiply")

    print("\n== DeltaNet state precision cost (batch 1, read + write every token) ==")
    w = P["read_per_token"] * BYTES_PER_W["int4"]
    for nm, b in (("bf16", 2), ("fp32", 4)):
        st = 2 * STATE_VALUES * b
        print(f"  {nm}: {st/MB:6.1f} MB per token = {100*st/w:4.1f}% of the weight bytes")


# ------------------------------------------------ 11: capacity and 1M context
MAX_CTX = 1_010_000          # model card: 262,144 native, up to 1,010,000 with YaRN
STORED_WEIGHTS = P["total"] * BYTES_PER_W["int4"]   # includes the embedding table
KV_BYTES = {"bf16": 2.0, "int8 (+scale/64)": 1 + 2 / 64, "int4 (+scale/32)": 0.5 + 2 / 32}
PREFILL_OPS = 2 * 4096 * 1e9  # 64x64 array, 1 GHz, 2 ops per MAC = 8.2 TOPS


def prefill_seconds(ctx, ops=PREFILL_OPS):
    """Sum of per-token work for positions 0..ctx-1, compute-bound, 100% busy."""
    linear = 2 * P["read_per_token"] * ctx
    attn = N_FULL * Q_HEADS * HEAD_DIM * 2 * 2 * ctx * ctx / 2
    return (linear + attn) / ops


def long_context():
    print("\n== Capacity and speed vs context (512 GB/s, 92%, batch 1, fp32 state) ==")
    print(f"  stored weights incl. embedding: {STORED_WEIGHTS/GB:.2f} GB "
          f"(embedding alone {P['emb']*BYTES_PER_W['int4']/GB:.2f} GB)")
    state = STATE_VALUES * 4 + CONV_STATE_VALUES * 4
    for ctx in (32_768, 131_072, 262_144, MAX_CTX):
        print(f"  ctx {ctx:9,d}:  prefill on 8.2 TOPS {prefill_seconds(ctx)/60:8.1f} min")
        for nm, b in KV_BYTES.items():
            kv = ctx * KV_VALUES_PER_TOKEN * b
            cap = STORED_WEIGHTS + kv + state
            per_tok = P["read_per_token"] * BYTES_PER_W["int4"] + kv + 2 * STATE_VALUES * 4
            tok = CARD_UTIL * CARD_BW / per_tok
            print(f"      KV {nm:17s} {kv/GB:6.2f} GB  needs {cap/GB:6.2f} GB  "
                  f"decode {tok:6.1f} tok/s")
    pcie = 8 * GB
    kv = MAX_CTX * KV_VALUES_PER_TOKEN * KV_BYTES["int8 (+scale/64)"]
    print(f"  streaming 1M int8 KV from host over PCIe Gen4 x4: {kv/pcie:.1f} s per token")

    print("\n== Board memory options (all 128-bit, 512 GB/s; $12/GB; die/package/board as our_card) ==")
    die, _, _ = die_cost(80, "TSMC N7/N6")
    base_tok = CARD_UTIL * CARD_BW / (P["read_per_token"] * BYTES_PER_W["int4"]
                                      + 4096 * KV_VALUES_PER_TOKEN + 2 * STATE_VALUES * 4)
    for nm, gb in (("4 x 2 GB", 8), ("4 x 3 GB", 12), ("8 x 2 GB clamshell", 16),
                   ("8 x 3 GB clamshell", 24)):
        unit = gb * 12 + die + 15 + 35
        print(f"  {nm:20s} {gb:2d} GB  unit ${unit:5.0f}  {base_tok:5.1f} tok/s at 4K  "
              f"{base_tok/(2*unit):5.3f} tok/s per $")


# ------------------------------------------------ 12: speculative decoding
def mtp_weights():
    """Assumed MTP module (as in Qwen3-Next): one full-attention decoder layer
    plus a 2H -> H projection that mixes the token embedding and hidden state."""
    return P["full"] + P["ffn"] + 2 * HIDDEN * HIDDEN


def spec_decode():
    print("\n== Speculative decoding with the built-in MTP head (batch 1, ctx 4096) ==")
    wb = BYTES_PER_W["int4"]
    main_bytes = (P["read_per_token"] * wb + 4096 * KV_VALUES_PER_TOKEN
                  + 2 * STATE_VALUES * 4)
    mtp = mtp_weights() * wb
    head_full = P["head"] * wb
    head_trim = 32_768 * HIDDEN * wb
    print(f"  main pass {main_bytes/GB:.2f} GB; MTP layer {mtp/MB:.0f} MB; "
          f"LM head {head_full/MB:.0f} MB full, {head_trim/MB:.0f} MB if drafts use top-32K vocab")
    for label, head in (("full vocab", head_full), ("trimmed vocab", head_trim)):
        draft = (mtp + head) / main_bytes
        print(f"  each draft costs {100*draft:4.1f}% of a main pass ({label})")
        for a in (0.6, 0.7, 0.8):
            row = []
            for k in (1, 2, 3):
                exp = (1 - a ** (k + 1)) / (1 - a)       # expected tokens per step
                row.append(f"k={k}: {exp:4.2f} tok/step, x{exp/(1+k*draft):4.2f}")
            print(f"    accept {a:.1f}:  " + "   ".join(row))
    wpc = CARD_UTIL * CARD_BW / wb / 1e9                  # weights per cycle at 1 GHz
    for k in (0, 1, 2, 3):
        print(f"  verify {k+1} tokens per weight pass: {wpc*(k+1):6.0f} MACs/cycle at 1 GHz")
    # to replay an update later you need k, v, beta and decay per head (not q)
    per_layer = LIN_QK_HEADS * LIN_DIM + LIN_V_HEADS * LIN_DIM + 2 * LIN_V_HEADS
    print(f"  lazy-commit log: {per_layer} values per DeltaNet layer per token = "
          f"{per_layer*N_LIN*2*4/KB:.0f} KB for 4 tokens (bf16), vs "
          f"{STATE_VALUES*4*3/MB:.0f} MB to write 3 extra fp32 state checkpoints")


# ------------------------------------------------ 13: matrix shapes
def shapes():
    print("\n== Weight matrices (in -> out), per layer type ==")
    rows = [
        ("full attn: q + gate", HIDDEN, 2 * Q_HEADS * HEAD_DIM, N_FULL),
        ("full attn: k", HIDDEN, KV_HEADS * HEAD_DIM, N_FULL),
        ("full attn: v", HIDDEN, KV_HEADS * HEAD_DIM, N_FULL),
        ("full attn: o", Q_HEADS * HEAD_DIM, HIDDEN, N_FULL),
        ("DeltaNet: q,k,v,z fused", HIDDEN, 2 * LIN_QK_HEADS * LIN_DIM + 2 * LIN_V_HEADS * LIN_DIM, N_LIN),
        ("DeltaNet: beta, a", HIDDEN, 2 * LIN_V_HEADS, N_LIN),
        ("DeltaNet: out", LIN_V_HEADS * LIN_DIM, HIDDEN, N_LIN),
        ("FFN gate", HIDDEN, FFN, LAYERS),
        ("FFN up", HIDDEN, FFN, LAYERS),
        ("FFN down", FFN, HIDDEN, LAYERS),
        ("LM head", HIDDEN, VOCAB, 1),
    ]
    tot = sum(i * o * n for _, i, o, n in rows)
    for nm, i, o, n in rows:
        print(f"  {nm:24s} {i:6d} -> {o:7d}  x{n:2d}  {i*o*n/1e6:8.1f} M  {100*i*o*n/tot:5.1f}%")
    print(f"  total {tot/1e9:.3f} B (script read_per_token {P['read_per_token']/1e9:.3f} B, "
          f"difference = conv weights)")
    print("  per-head non-weight math: attention q[4x256] . K[256xL] per KV head (GQA 4:1);"
          " DeltaNet S[128x128] per V head, 32 heads")


def sram_budget():
    """First guess at on-chip SRAM. Tile sizes are our choices, not facts."""
    print("\n== On-chip SRAM, first guess ==")
    T = 64                                            # prefill / verify tile of tokens
    items = [
        ("weight stream FIFO (Little's law, 100 ns, x2 headroom)", 2 * 512 * GB * 100e-9),
        ("DeltaNet state, one head fp32, double-buffered", 2 * LIN_DIM * LIN_DIM * 4),
        ("lazy-commit log for one layer, 4 tokens fp32", 4 * 6208 * 4),
        (f"activations in, {T} tokens x 4096 bf16", T * HIDDEN * 2),
        (f"FFN intermediate, {T} tokens x 12288 bf16", T * FFN * 2),
        (f"residual stream, {T} tokens x 4096 fp32", T * HIDDEN * 4),
        ("KV tile, 256 positions x 256 dim x K,V int8, double-buffered", 2 * 256 * HEAD_DIM * 2),
        ("instruction memory (guess)", 256 * 1024),
    ]
    tot = 0
    for nm, b in items:
        tot += b
        print(f"  {nm:62s} {b/KB:8.0f} KB")
    print(f"  total {tot/MB:.2f} MB; at a rough 0.5 mm2 per MB (N6-class, to verify) "
          f"about {tot/MB*0.5:.1f} mm2")
    print(f"  whole DeltaNet state on chip instead: {STATE_VALUES*4/MB:.1f} MB fp32, "
          f"about {STATE_VALUES*4/MB*0.5:.0f} mm2, saves 2.3% of bytes per token")


if __name__ == "__main__":
    main()
