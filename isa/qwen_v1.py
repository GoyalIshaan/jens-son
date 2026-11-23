"""The Qwen3.5-9B program written in ISA v1 (note 28).

build(mode, T, ...) returns a Prog for one of:
  - "decode": plain greedy decode, one token per pass (T = 1), N passes
  - "spec":   speculative steps: verify T = k + 1 tokens with the main model, accept, then k MTP drafts
The program is real ISA v1: hardware loops over layers, heads and attention tiles, registers for every
dynamic offset (position, layer, buffer parity), semaphores between units. It also lays out SRAM and DRAM
and checks that they fit.

Offline layout changes this program relies on (done once, when the weights are loaded):
  - every weight matrix is stored in the order the matrix unit reads it, with its 16-byte table in front
  - q_proj rows de-interleaved: all 16 query heads first, then the 16 gates
  - the FFN gate and up matrices split into 12 chunks of 1,024 rows, stored chunk by chunk, gate then up
  - in_proj_qkv, in_proj_z, in_proj_b and in_proj_a stored as one matrix (12,352 outputs)
  - norm weights stored as (1 + w) in fp32; exp(A_log) stored negated
  - V is kept transposed in 256-position blocks ([block][dim][position]) so P . V streams like a matrix
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "design", "calc"))
import isa_v1 as I  # noqa: E402
from isa_v1 import Dma, Mm, Opnd as O, Prog, Vec  # noqa: E402

from dataclasses import dataclass as _dc, replace


@_dc(frozen=True)
class Model:
    """The model the program is generated for. Defaults: Qwen3.5-9B. QWEN_08B is the RTL-simulation stand-in."""
    hid: int = 4096
    ffn: int = 12288           # padded to whole 1,024-input beats by the weight packer (zero rows)
    vocab: int = 248_320
    qh: int = 16               # query heads (attention), 4 per KV head
    kvh: int = 4
    hd: int = 256
    rot: int = 64
    lqk: int = 16              # DeltaNet key heads
    lv: int = 32               # DeltaNet value heads
    ld: int = 128              # DeltaNet head size
    layers: int = 32           # groups of 3 DeltaNet + 1 attention
    eps: float = 1e-6

    @property
    def ffn_p(self):
        return -(-self.ffn // 1024) * 1024


Q9B = Model()
QWEN_08B = Model(hid=1024, ffn=3584, qh=8, kvh=2, lqk=16, lv=16, layers=24)

HID, FFN, VOCAB = 4096, 12288, 248_320
QH, KVH, HD, ROT = 16, 4, 256, 64
LQK, LV, LD = 16, 32, 128
DRAFT_VOCAB, WINDOW = 32_768, 4096
WHDR = 32                                  # per-matrix header in the weight stream: 16-byte table, fp32 tscale, pad
EMB_ROW = HID + HID // 32 * 2              # int8 embedding row + fp16 scale per 32 (D-005)
MOW = 8192 + 4096 + 32 + 32                # DeltaNet projections: qkv | z | b | a
CYW = 8192 + 64                            # conv output: q | k | v | beta | decay
FF_CHUNK = 1024
N_CHUNK = FFN // FF_CHUNK
PRM_DN = (2 * HID + 8192 * 4 + 32 + 32 + LD) * 4
PRM_AT = (2 * HID + 2 * HD) * 4
PRM_MTP = (5 * HID + 2 * HD) * 4
PRMB = max(PRM_DN, PRM_AT, PRM_MTP)
STATE_HEAD = LD * LD * 4


def wbytes(n_in, n_out):
    return n_in * n_out * 9 // 16 + WHDR    # 4.5 bits per weight + the 32-byte header (table, tscale)


class Alloc:
    def __init__(self, name, limit=None, base=0):
        self.name, self.limit, self.top, self.items = name, limit, base, []

    def __call__(self, what, size, align=64):
        self.top = (self.top + align - 1) // align * align
        # BANK_STAGGER = S: buffer k starts k * S lines into the 64-bank stripe, so buffers one command uses
        # together don't line up on the same banks (note 34). 0 keeps the plain layout.
        st = int(os.environ.get("BANK_STAGGER", 0))
        if st and self.name != "DRAM":
            want = (len(self.items) * st) % 64
            self.top += ((want - (self.top // 64)) % 64) * 64
        a = self.top
        self.top += int(size)
        self.items.append((what, a, int(size)))
        return a

    def total(self):
        return self.top

    def report(self, top=None):
        lines = [f"  {w:52s} {s / 1024:9.1f} KB" for w, _, s in self.items]
        return "\n".join(lines)


class Sems:
    def __init__(self):
        self.ids = {}

    def __getattr__(self, k):
        if k.startswith("_") or k == "ids":
            raise AttributeError(k)
        if k not in self.ids:
            self.ids[k] = len(self.ids) + 1
            assert self.ids[k] <= I.N_SEMS, "out of semaphores"
        return self.ids[k]


# registers
X_L, X_A, X_NC, X_WP, X_KVO, X_DNL, X_PRMO, X_PRMP = 1, 2, 3, 4, 5, 6, 7, 8
X_T0, X_T1, X_T2, X_T3, X_CNT = 9, 10, 11, 12, 13
X_G, X_RK, X_RSC, X_RQA, X_RQE, X_ROG, X_RML = 14, 15, 16, 17, 18, 19, 20
X_POS, X_TC, X_PAR, X_POSP, X_TCP, X_PARP, X_LR = 21, 22, 23, 24, 25, 26, 27
X_H, X_SBP, X_K1, X_K2 = 28, 29, 30, 31


def build(mode="spec", k=3, steps=2, lmax=40_000, fused=False, cfg=Q9B, dn_heads=4):
    assert mode in ("decode", "spec")
    HID, VOCAB, QH, KVH, HD, ROT = cfg.hid, cfg.vocab, cfg.qh, cfg.kvh, cfg.hd, cfg.rot
    if not 1 <= dn_heads <= 4:
        raise ValueError("dn_heads must be between one and four")
    LQK, LV, LD, NL = cfg.lqk, cfg.lv, cfg.ld, cfg.layers
    FFN = cfg.ffn_p
    assert QH == 4 * KVH and HD == 256 and LD == 128 and NL % 4 == 0 and HID % 1024 == 0
    NDN, NAT = NL * 3 // 4, NL // 4
    QKV_CH = 2 * LQK * LD + LV * LD            # conv channels: q | k | v
    VD = LV * LD                               # DeltaNet value width (out-projection input)
    QD = QH * HD                               # attention output width (o-projection input)
    QPW = 2 * QD                               # q-projection outputs: all query heads, then their gates
    MOW = -(-(QKV_CH + VD + 2 * LV) // 64) * 64   # DeltaNet projections: qkv | z | b | a (one matrix, zero rows
    #                                            pad it to whole 64-output tiles: 12,352 for the 9B, 8,256 for the 0.8B)
    CYW = QKV_CH + 2 * LV                      # conv output: q | k | v | beta | decay
    N_CHUNK = FFN // FF_CHUNK
    EMB_ROW = HID + HID // 32 * 2
    PRM_DN = (2 * HID + QKV_CH * 4 + 2 * LV + LD) * 4
    PRM_AT = (2 * HID + 2 * HD) * 4
    PRM_MTP = (5 * HID + 2 * HD) * 4
    PRMB = max(PRM_DN, PRM_AT, PRM_MTP)
    WMAX = max(HID, VD, QD)
    assert MOW >= QPW + 2 * KVH * HD and MOW % 64 == 0
    T = 1 if mode == "decode" else k + 1
    p = Prog()
    sm = Sems()
    sa = Alloc("SRAM", I.SRAM_BYTES)
    da = Alloc("DRAM")
    R4 = 4 * T                                   # attention rows: T tokens x 4 query heads per KV head

    # ------------------------------------------------------------- SRAM
    CONST = sa("constants", 256)
    H = sa("residual stream (fp32)", T * HID * 4)
    HMTP = sa("MTP residual (fp32)", HID * 4) if mode == "spec" else 0
    EH = sa("MTP input [e ; h] (fp32)", 2 * HID * 4) if mode == "spec" else 0
    HPOST = sa("final-norm hidden, T rows (fp32, the drafts' input)", T * HID * 4) if mode == "spec" else 0
    HMTPP = sa("MTP final-norm hidden (fp32, the next draft's input)", HID * 4) if mode == "spec" else 0
    EMB = sa("embedding rows (int8 + scales)", T * EMB_ROW)
    XN = sa("normed matrix input (int8)", T * 2 * WMAX)
    XNE = sa("  its exponents", T * 2 * WMAX // 32)
    FF8 = sa("FFN intermediate (int8)", T * FFN)
    FF8E = sa("  its exponents", T * FFN // 32)
    MO = sa("projection outputs (fp32)", T * MOW * 4)
    CY = sa("conv output + beta + decay (fp32)", T * CYW * 4)
    CH = sa("conv history, one layer", 3 * QKV_CH * 4)
    CW = sa("conv window out, one layer", (3 + T) * QKV_CH * 4)
    LOGIN = sa("DeltaNet replay log, one layer", T * CYW * 4) if mode == "spec" else 0
    SB = sa("DeltaNet state, 4 heads", 4 * STATE_HEAD)
    DNO = sa("DeltaNet output (fp32)", T * VD * 4)
    GU = sa("FFN gate/up chunk x2 (fp32)", 2 * T * 2 * FF_CHUNK * 4)
    TMP = sa("vector temporary (fp32)", T * WMAX * 4)
    SS = sa("per-row scalars", 4096)
    PRM = sa("layer parameters x2", 2 * PRMB)
    QA = sa("q int8, per KV head", KVH * R4 * HD)
    QAE = sa("  its exponents", KVH * R4 * HD // 32)
    KQ = sa("new K int8", T * KVH * HD)
    VQ = sa("new V int8", T * KVH * HD)
    KVS = sa("new K,V scales (fp16)", T * 2 * KVH * 2)
    N_ATT = NAT + (1 if mode == "spec" else 0)
    VT = sa("V^T staging ring, 64 positions, all attn layers", N_ATT * KVH * HD * 64)
    SCL = sa("K/V scale staging ring", N_ATT * KVH * 2 * 64 * 2)
    RT = sa("rope table rows", T * 2 * 96 * 4)
    CS = sa("cos/sin for this pass", T * 2 * 32 * 4)
    slab = dict(S=R4 * 1024 * 4, P=R4 * 1024, PE=R4 * 32, OT=R4 * HD * 4, KS=2 * 1024 * 2)
    SLAB = sum((v + 63) // 64 * 64 for v in slab.values())
    TILE = sa("attention tile slabs x2 (scores, P, P.V, scales)", 2 * SLAB)
    off, a_ = {}, TILE
    for kname, v in slab.items():
        off[kname] = a_
        a_ += (v + 63) // 64 * 64
    S_T, P_T, PE_T, OT_T, KS_T = (off[x] for x in ("S", "P", "PE", "OT", "KS"))
    OG = sa("attention accumulators, 4 KV heads", KVH * R4 * HD * 4)
    MLB = 6 * R4 * 4
    ML = sa("softmax m, l, alpha", KVH * MLB)
    ATT = sa("attention output (fp32)", T * QD * 4)
    GT = sa("sigmoid(gate)", T * QD * 4)
    YB = sa("argmax ids, drafts, counts", 256)
    TK, ACC, DI = YB + 64, YB + 128, YB + 192

    # ------------------------------------------------------------- DRAM
    def layer_stream(kind):
        b = 0
        if kind == "dn":
            b += wbytes(HID, MOW) + wbytes(VD, HID)
        else:
            b += wbytes(HID, QPW) + 2 * wbytes(HID, KVH * HD) + wbytes(QD, HID)
        return b + 2 * N_CHUNK * wbytes(HID, FF_CHUNK) + wbytes(FFN, HID)
    main_stream = sum(layer_stream("at" if (l + 1) % 4 == 0 else "dn") for l in range(NL)) + wbytes(HID, VOCAB)
    mtp_stream = wbytes(2 * HID, HID) + layer_stream("at") + wbytes(HID, DRAFT_VOCAB)
    W_MAIN = da("main weights in stream order", main_stream)
    W_MTP = da("MTP draft weights", mtp_stream) if mode == "spec" else 0
    EMBT = da("embedding table (int8)", VOCAB * EMB_ROW)
    KH = lmax * HD                              # one KV head's K (or V^T) bytes
    KV_LAYER = 2 * KVH * KH + KVH * 2 * lmax * 2
    KVB = da("KV cache (int8 K, V^T, fp16 scales)", N_ATT * KV_LAYER)
    STB = da("DeltaNet state (fp32)", NDN * LV * STATE_HEAD)
    CLOG = da("conv window log", NDN * (3 + T) * QKV_CH * 4)
    DLOG = da("DeltaNet replay log", NDN * T * CYW * 4) if mode == "spec" else 0
    PRMD = da("norms and small per-layer parameters", (NL + 3) * PRMB)
    ROPE = da("rope angle tables (2 x 1,024 rows)", 2 * 1024 * 96 * 4)
    REMAP = da("draft-vocabulary id table", DRAFT_VOCAB * 4) if mode == "spec" else 0

    stream_base = [W_MAIN]                      # DRAM base of the weight stream being read (offsets in X_WP)

    def c(i):
        return O(CONST + 4 * i, 0, 0, 0)        # a broadcast constant
    C_1_4096, C_EPS, C_1E6, C_1_128, C_RS128, C_NINF, C_ZERO, C_1_256 = range(8)

    # ------------------------------------------------------------- building blocks
    def weights(n_in, n_out):
        """Stream one matrix (table + data) into the FIFO and pop its table."""
        n = wbytes(n_in, n_out)
        tile = 64 * 576                          # one 64-output tile of one 1,024-input chunk
        tiles = n_in * n_out // (1024 * 64)
        assert tiles * tile + WHDR == n
        wb = stream_base[0]
        p.D("W", Dma(wb, 0, WHDR + tile, 1), regs=(X_WP,), note="header + first weight tile")
        if tiles > 1:
            p.D("W", Dma(wb + WHDR + tile, 0, tile, tiles - 1, tile, 0), regs=(X_WP,), note=f"weights {n_in}x{n_out}")
        p.S("ADDI", X_WP, X_WP, imm=n)
        p.M("TAB")

    def mm(n_in, n_out, act, exps, dst, epi="f32", wait=(), sig=(), regs=(), id_base=0, note="",
           act_stride=None, exps_stride=None, dst_stride=None):
        weights(n_in, n_out)
        d = Mm(n_in, n_out, T, act, act_stride or n_in, exps, exps_stride or n_in // 32, dst,
               dst_stride if dst_stride is not None else n_out * 4, epi=epi, id_base=id_base)
        p.M("MM", d, regs=regs, wait=wait, sig=sig, note=note)

    def rmsnorm(src, w, wait=(), sig=(), n=HID, rows=None, dst=XN, dste=XNE):
        rows = rows or T
        p.V("RSUMSQ", Vec(n, 1, rows, dst=O(SS, 4, 0, 4), a=O(src, 4, 0, n * 4)), regs=(0, X_T0 if False else 0),
            wait=wait, note="rmsnorm sum of squares")
        p.V("RSQRT", Vec(1, 1, rows, dst=O(SS, 4, 0, 4), a=O(SS, 4, 0, 4), b=c(C_1_4096), c=c(C_EPS)))
        p.V("MUL", Vec(n, 1, rows, dst=O(TMP, 4, 0, n * 4), a=O(src, 4, 0, n * 4), b=O(SS, 0, 0, 4)))
        p.V("QUANT", Vec(n, 1, rows, dst=O(dst, 1, 0, n, "i8"), a=O(TMP, 4, 0, n * 4), b=w,
                         c=O(dste, 1, 0, n // 32, "e8")), regs=(0, 0, X_PRMP), sig=sig, note="rmsnorm -> int8")

    def prm_load(layer_slot):
        p.S("LI", X_PRMO, imm=layer_slot * PRMB)
        p.D("R", Dma(PRMD, PRM, PRMB), regs=(X_PRMO, X_PRMP), wait=(sm.PRM_free,), sig=(sm.PRM_ok,),
            note="layer parameters")
        p.fence("V", wait=(sm.PRM_ok,), note="parameters arrived")

    def prm_flip():
        p.S("LI", X_T3, imm=PRMB)
        p.S("SUB", X_PRMP, X_T3, X_PRMP)

    def ffn(post_w_off, src=H):
        rmsnorm(src, O(PRM + post_w_off, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready, sm.PRM_free))
        p.fence("M", wait=(sm.XN_ready,), note="FFN input ready")
        p.S("LI", X_CNT, imm=N_CHUNK)
        p.S("LI", X_T0, imm=0)                   # GU parity offset
        p.S("LI", X_T1, imm=0)                   # FF8 column offset
        p.S("LI", X_T2, imm=0)                   # FF8E column offset
        gsz = T * 2 * FF_CHUNK * 4            # one GU half
        with p.loop(X_CNT, note="FFN chunks"):
            weights(HID, FF_CHUNK)
            p.M("MM", Mm(HID, FF_CHUNK, T, XN, HID, XNE, HID // 32, GU, 2 * FF_CHUNK * 4), regs=(0, 0, X_T0),
                wait=(sm.GU_free,), note="FFN gate chunk")
            weights(HID, FF_CHUNK)
            p.M("MM", Mm(HID, FF_CHUNK, T, XN, HID, XNE, HID // 32, GU + FF_CHUNK * 4, 2 * FF_CHUNK * 4),
                regs=(0, 0, X_T0), sig=(sm.GU_ok,), note="FFN up chunk")
            p.V("SILU", Vec(FF_CHUNK, 1, T, dst=O(TMP, 4, 0, FF_CHUNK * 4), a=O(GU, 4, 0, 2 * FF_CHUNK * 4)),
                regs=(0, X_T0), wait=(sm.GU_ok,))
            p.V("QUANT", Vec(FF_CHUNK, 1, T, dst=O(FF8, 1, 0, FFN, "i8"), a=O(TMP, 4, 0, FF_CHUNK * 4),
                             b=O(GU + FF_CHUNK * 4, 4, 0, 2 * FF_CHUNK * 4), c=O(FF8E, 1, 0, FFN // 32, "e8")),
                regs=(X_T1, 0, X_T0, X_T2), sig=(sm.GU_free, sm.FF_part), note="silu(gate)*up -> int8")
            p.S("LI", X_T3, imm=gsz)
            p.S("SUB", X_T0, X_T3, X_T0)
            p.S("ADDI", X_T1, X_T1, imm=FF_CHUNK)
            p.S("ADDI", X_T2, X_T2, imm=FF_CHUNK // 32)
        # the down matrix needs all 12 chunks: wait for the last one (counting semaphore x12)
        p.S("LI", X_CNT, imm=N_CHUNK - 1)
        with p.loop(X_CNT, note="drain chunk signals"):
            p.fence("M", wait=(sm.FF_part,))
        mm(FFN, HID, FF8, FF8E, src, "acc", wait=(sm.FF_part,), sig=(sm.H_ready,), note="FFN down (+= residual)")

    # ------------------------------------------------------------- DeltaNet layer
    def deltanet(dn_index, spec):
        prm_load(dn_index_slot[dn_index])
        rmsnorm(H, O(PRM, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready,))
        mm(HID, MOW, XN, XNE, MO, wait=(sm.XN_ready,), sig=(sm.MO_ready,), note="DeltaNet in-projections")
        # conv history for this layer: the 3 rows that end at the last committed token (log row nc .. nc+2)
        p.S("LI", X_T0, imm=dn_index * (3 + T) * QKV_CH * 4)
        if (QKV_CH * 4) & (QKV_CH * 4 - 1) == 0:
            p.S("SLLI", X_T1, X_NC, imm=(QKV_CH * 4).bit_length() - 1)   # nc rows (rows nc .. nc+2 of last window)
        else:
            p.S("LI", X_T1, imm=QKV_CH * 4)
            p.S("MUL", X_T1, X_T1, X_NC)
        p.S("ADD", X_T0, X_T0, X_T1)
        p.D("R", Dma(CLOG, CH, 3 * QKV_CH * 4), regs=(X_T0,), wait=(sm.CH_free,), sig=(sm.CH_ok,), note="conv history")
        if spec:
            p.S("LI", X_T2, imm=dn_index * T * CYW * 4)
            p.D("R", Dma(DLOG, LOGIN, T * CYW * 4), regs=(X_T2,), wait=(sm.LOG_free,), sig=(sm.LOG_ok,),
                note="DeltaNet replay log")
        p.fence("V", wait=(sm.CW_free, sm.CYX_free), note="window and log buffers written out")
        p.V("CONV4", Vec(QKV_CH, 1, T, dst=O(CY, 4, 0, CYW * 4), a=O(MO, 4, 0, MOW * 4), b=O(PRM + 2 * HID * 4, 16, 0, 0),
                         c=O(CH, 4, 0, QKV_CH * 4), par=(("window", CW),)), regs=(0, 0, X_PRMP), wait=(sm.MO_ready, sm.CH_ok),
            sig=(sm.CH_free, sm.CW_ok), note="causal conv, kernel 4")
        p.S("LI", X_T0, imm=dn_index * (3 + T) * QKV_CH * 4)
        p.D("X", Dma(CW, CLOG, (3 + T) * QKV_CH * 4), regs=(0, X_T0), wait=(sm.CW_ok,), sig=(sm.CW_free,),
            note="conv window -> log")
        p.V("SILU", Vec(QKV_CH, 1, T, dst=O(CY, 4, 0, CYW * 4), a=O(CY, 4, 0, CYW * 4)))
        # L2 norm of q and k per head (32 rows of 128 per token), q also x 1/sqrt(128)
        p.V("RSUMSQ", Vec(LD, 2 * LQK, T, dst=O(SS, 4, 4, 128), a=O(CY, 4, LD * 4, CYW * 4)))
        p.V("RSQRT", Vec(2 * LQK, 1, T, dst=O(SS, 4, 0, 128), a=O(SS, 4, 0, 128), c=c(C_1E6)))
        p.V("MUL", Vec(LD, 2 * LQK, T, dst=O(CY, 4, LD * 4, CYW * 4), a=O(CY, 4, LD * 4, CYW * 4), b=O(SS, 0, 4, 128)))
        p.V("MUL", Vec(LQK * LD, 1, T, dst=O(CY, 4, 0, CYW * 4), a=O(CY, 4, 0, CYW * 4), b=c(C_RS128)))
        p.V("SIGMOID", Vec(LV, 1, T, dst=O(CY + QKV_CH * 4, 4, 0, CYW * 4), a=O(MO + (QKV_CH + VD) * 4, 4, 0, MOW * 4)),
            note="beta")
        P_DT = PRM + (2 * HID + 4 * QKV_CH) * 4  # dt_bias | -exp(A_log) | gated-norm weight
        p.V("ADD", Vec(LV, 1, T, dst=O(TMP, 4, 0, 128), a=O(MO + (QKV_CH + VD + LV) * 4, 4, 0, MOW * 4),
                       b=O(P_DT, 4, 0, 0)), regs=(0, 0, X_PRMP))
        p.V("SOFTPLUS", Vec(LV, 1, T, dst=O(TMP, 4, 0, 128), a=O(TMP, 4, 0, 128)))
        p.V("MUL", Vec(LV, 1, T, dst=O(TMP, 4, 0, 128), a=O(TMP, 4, 0, 128),
                       b=O(P_DT + LV * 4, 4, 0, 0)), regs=(0, 0, X_PRMP))
        p.V("EXP", Vec(LV, 1, T, dst=O(CY + (QKV_CH + LV) * 4, 4, 0, CYW * 4), a=O(TMP, 4, 0, 128)), note="decay")
        if spec:
            p.S("LI", X_T2, imm=dn_index * T * CYW * 4)
            p.D("X", Dma(CY, DLOG, T * CYW * 4), regs=(0, X_T2), wait=(sm.CY_ok,), sig=(sm.CYX_free,),
                note="this step's DeltaNet inputs -> replay log")
            p.fence("V", sig=(sm.CY_ok,), note="conv outputs ready")
        else:
            p.fence("V", sig=(sm.CYX_free,), note="(no log in plain decode)")
        # One bank group is shared by the R, V and X queues. Its release
        # is signalled by the final consumer, once per group.
        if spec:
            p.fence("V", wait=(sm.LOG_ok,), note="replay log arrived")
        p.S("LI", X_K1, imm=dn_index * LV * STATE_HEAD)
        p.S("LI", X_H, imm=0)

        def head_group(count):
            size = count * STATE_HEAD
            params = (("heads", count), ("work", 0), ("commit_spec", 1), ("lqk", LQK), ("lv", LV))
            dn = Vec(LD, LD, 1, dst=O(DNO, 4, 0, VD * 4), a=O(CY, 4, 0, CYW * 4),
                     b=O(LOGIN, 4, 0, CYW * 4), c=O(SB, 4, LD * 4, 0), par=params + (("T", T),))
            p.D("R", Dma(STB, SB, size), regs=(X_K1,), wait=(sm.SB_free,), sig=(sm.S_ok,),
                note="load resident DeltaNet heads")
            if spec:
                # Checkpoint committed replay to DRAM before any speculative
                # update. All four SRAM banks can then be temporary in place.
                p.V("DN", replace(dn, par=params + (("T", 0),)), regs=(0, X_H, 0, 0, X_NC),
                    wait=(sm.S_ok,), sig=(sm.DN_done,), note="DeltaNet committed replay")
                p.D("X", Dma(SB, STB, size), regs=(0, X_K1), wait=(sm.DN_done,), sig=(sm.S_checkpoint,),
                    note="checkpoint replayed heads")
                p.V("DN", dn, regs=(0, X_H), wait=(sm.S_checkpoint,), sig=(sm.SB_free,),
                    note="DeltaNet tentative tokens in resident banks")
            else:
                p.V("DN", dn, regs=(0, X_H), wait=(sm.S_ok,), sig=(sm.DN_done,),
                    note="DeltaNet committed tokens")
                p.D("X", Dma(SB, STB, size), regs=(0, X_K1), wait=(sm.DN_done,), sig=(sm.SB_free,),
                    note="store committed heads")
            p.S("ADDI", X_K1, X_K1, imm=size)
            p.S("ADDI", X_H, X_H, imm=count)

        if LV >= dn_heads:
            p.S("LI", X_CNT, imm=LV // dn_heads)
            with p.loop(X_CNT, note="resident DeltaNet head groups"):
                head_group(dn_heads)
        if LV % dn_heads:
            head_group(LV % dn_heads)
        if spec:
            p.fence("V", sig=(sm.LOG_free,), note="replay log used")
        # gated RMSNorm with z, then the output projection
        p.V("RSUMSQ", Vec(LD, LV, T, dst=O(SS, 4, 4, 128), a=O(DNO, 4, LD * 4, VD * 4)))
        p.V("RSQRT", Vec(LV, 1, T, dst=O(SS, 4, 0, 128), a=O(SS, 4, 0, 128), b=c(C_1_128), c=c(C_EPS)))
        p.V("MUL", Vec(LD, LV, T, dst=O(DNO, 4, LD * 4, VD * 4), a=O(DNO, 4, LD * 4, VD * 4), b=O(SS, 0, 4, 128)))
        p.V("MUL", Vec(LD, LV, T, dst=O(DNO, 4, LD * 4, VD * 4), a=O(DNO, 4, LD * 4, VD * 4),
                       b=O(P_DT + 2 * LV * 4, 4, 0, 0)), regs=(0, 0, X_PRMP))
        p.V("SILU", Vec(VD, 1, T, dst=O(TMP, 4, 0, VD * 4), a=O(MO + QKV_CH * 4, 4, 0, MOW * 4)))
        p.V("QUANT", Vec(VD, 1, T, dst=O(XN, 1, 0, VD, "i8"), a=O(DNO, 4, 0, VD * 4), b=O(TMP, 4, 0, VD * 4),
                         c=O(XNE, 1, 0, VD // 32, "e8")), sig=(sm.XN_ready,), note="gated norm -> int8")
        mm(VD, HID, XN, XNE, H, "acc", wait=(sm.XN_ready,), sig=(sm.H_ready,), note="DeltaNet out-projection")
        ffn(HID * 4)
        prm_flip()

    # ------------------------------------------------------------- attention layer
    def kv_append(att_index):
        """Write this pass's K (natural rows) and V (into the V^T staging ring, then whole 32-byte atoms)."""
        base = KVB + att_index * KV_LAYER
        vt_stage = VT + att_index * KVH * HD * 64
        sc_stage = SCL + att_index * KVH * 2 * 64 * 2
        p.fence("L", wait=(sm.KVQ_L,), note="new K, V quantized")
        for t in range(T):
            p.S("ADDI", X_T0, X_L, imm=t)
            p.S("ANDI", X_T1, X_T0, imm=63)                      # slot in the 64-position ring
            p.S("SLLI", X_T2, X_T1, imm=1)                       # fp16 slot offset
            p.D("L", Dma(VQ + t * KVH * HD, vt_stage, 1, HD, 1, 64, KVH, HD, HD * 64), regs=(0, X_T1),
                note="V -> V^T staging (transpose)")
            p.D("L", Dma(KVS + t * 4 * KVH, sc_stage, 2, KVH, 2, 2 * 64 * 2, 2, 2 * KVH, 64 * 2), regs=(0, X_T2),
                note="K,V scales -> staging", sig=(sm.VT_ok,) if t == T - 1 else ())
        # K rows: positions L .. L+T-1 of each head, contiguous
        p.S("SLLI", X_T0, X_L, imm=8)
        p.D("X", Dma(KQ, base, HD, T, KVH * HD, HD, KVH, HD, KH), regs=(0, X_T0), wait=(sm.KVQ_X,), note="K rows out")
        for i_, which in enumerate((0, T - 1)):                  # the 32-position V^T chunks touched (1 or 2)
            p.S("ADDI", X_T0, X_L, imm=which)
            p.S("SRLI", X_T1, X_T0, imm=5)                       # chunk c
            p.S("ANDI", X_T2, X_T1, imm=1)
            p.S("SLLI", X_T2, X_T2, imm=5)                       # staging column (c % 2) x 32
            p.S("SRLI", X_T3, X_T1, imm=3)
            p.S("SLLI", X_T3, X_T3, imm=16)                      # block (c / 8) x 64 KB
            p.S("ANDI", X_T1, X_T1, imm=7)
            p.S("SLLI", X_T1, X_T1, imm=5)
            p.S("ADD", X_T3, X_T3, X_T1)                         # + (c % 8) x 32
            p.D("X", Dma(vt_stage, base + KVH * KH, 32, HD, 64, HD, KVH, HD * 64, KH), regs=(X_T2, X_T3),
                wait=(sm.VT_ok,) if i_ == 0 else (), note="V^T 32-byte atoms out")
        for i_, which in enumerate((0, T - 1)):                  # 16-position scale atoms touched
            p.S("ADDI", X_T0, X_L, imm=which)
            p.S("SRLI", X_T1, X_T0, imm=4)
            p.S("ANDI", X_T2, X_T1, imm=3)
            p.S("SLLI", X_T2, X_T2, imm=5)
            p.S("SLLI", X_T3, X_T1, imm=5)
            last = i_ == 1
            p.D("X", Dma(sc_stage, base + 2 * KVH * KH, 32, 2, 64 * 2, lmax * 2, KVH, 2 * 64 * 2, 2 * lmax * 2),
                regs=(X_T2, X_T3), sig=(sm.KVW_W, sm.KVW_R) if last else (), note="scale atoms out")

    def attention_core(att_index, ring):
        """Online-softmax attention over all positions, tile by tile (1,024 positions), per KV head.
        Software pipelined: the scores of tile i are made while tile i-1 is finished."""
        base = KVB + att_index * KV_LAYER
        # positions to cover, rounded up to whole 256-position blocks
        if ring:
            p.S("ADDI", X_T0, X_L, imm=T)
            p.S("LI", X_T1, imm=WINDOW)
            p.S("MIN", X_T0, X_T0, X_T1)
        else:
            p.S("ADDI", X_T0, X_L, imm=T)
        p.S("ADDI", X_LR, X_T0, imm=255)
        p.S("SRLI", X_LR, X_LR, imm=8)
        p.S("SLLI", X_LR, X_LR, imm=8)
        p.S("ADDI", X_T0, X_LR, imm=1023)
        p.S("SRLI", X_K2, X_T0, imm=10)                          # number of tiles
        p.S("LI", X_G, imm=0)
        p.S("LI", X_RK, imm=0)
        p.S("LI", X_RSC, imm=0)
        p.S("LI", X_RQA, imm=0)
        p.S("LI", X_RQE, imm=0)
        p.S("LI", X_ROG, imm=0)
        p.S("LI", X_RML, imm=0)
        p.fence("W", wait=(sm.KVW_W,), note="new K/V written before they're streamed")
        p.fence("R", wait=(sm.KVW_R,), note="new scales written")
        p.fence("M", wait=(sm.QA_ready,), note="q ready")
        p.S("LI", X_CNT, imm=KVH)
        Mold, Mn, Mt, Lt, Al, Ls = (ML + i * R4 * 4 for i in range(6))
        lim_step = 0 if ring else 1

        def first_half():
            """Tile at X_POS (count X_TC, slab X_PAR): stream K, scores, load the scales."""
            p.S("LI", X_T0, imm=1024)
            p.S("SUB", X_T0, X_LR, X_POS)
            p.S("LI", X_T1, imm=1024)
            p.S("MIN", X_TC, X_T0, X_T1)
            p.S("SLLI", X_T2, X_POS, imm=8)
            p.S("ADD", X_T2, X_T2, X_RK)
            p.S("SRLI", X_T3, X_TC, imm=8)                       # blocks of 256 rows of 256 B
            p.D("W", Dma(base, 0, 256 * 256, 4, 256 * 256, 0), regs=(X_T2, 0, X_T3), note="K tile")
            p.M("MM", Mm(HD, 1024, R4, QA, HD, QAE, HD // 32, S_T, 1024 * 4, wfmt="int8", seg=4, override="n_out"),
                regs=(X_RQA, X_RQE, X_PAR, 0, X_TC), wait=(sm.S_free,), sig=(sm.K_done,), note="q . K")
            p.S("SLLI", X_T2, X_POS, imm=1)
            p.S("ADD", X_T2, X_T2, X_RSC)
            p.D("R", Dma(base + 2 * KVH * KH, KS_T, 2048, 2, lmax * 2, 2048), regs=(X_T2, X_PAR),
                wait=(sm.KS_free,), sig=(sm.KS_ok,), note="tile K, V scales")

        def second_half(last=False):
            """Finish the tile at X_POSP (count X_TCP, slab X_PARP). Only the last tile can hold positions past the
            tokens of this pass, so only it gets the causal mask in the fused form."""
            box = dict(n=1024, m=R4, override="n")
            if fused:
                p.fence("V", wait=(sm.P_free,))
                if last:
                    p.V("CAUSAL", Vec(1024, 4, T, dst=O(S_T, 4, 4096, 4 * 4096), a=O(S_T, 4, 4096, 4 * 4096),
                                      par=(("step", lim_step),), override="n"),
                        regs=(X_PARP, X_PARP, X_POSP, X_L, X_TCP), wait=(sm.K_done,), note="causal mask (last tile)")
                # SMTILE (R7): a = scores, b = K scales (V scales at b + vs), c = running max m (alpha at
                # c + 4 mlb, l at c + 5 mlb), dst = P int8 with its exponents at dst + e[0], e[1] bytes per row
                p.V("SMTILE", Vec(**box, dst=O(P_T, 1, 1024, 0, "i8"), a=O(S_T, 4, 4096), b=O(KS_T, 2, 0, 0, "f16"),
                                  c=O(Mold, 0, 4, 0), par=(("vs", 2048), ("e", (PE_T - P_T, 32)), ("mlb", R4 * 4))),
                    regs=(X_PARP, X_PARP, X_PARP, X_RML, X_TCP), wait=(sm.KS_ok,) if last else (sm.K_done, sm.KS_ok),
                    sig=(sm.P_ok, sm.S_free), note="fused softmax step")
            else:
                p.V("MUL", Vec(**box, dst=O(S_T, 4, 4096), a=O(S_T, 4, 4096), b=O(KS_T, 2, 0, 0, "f16")),
                    regs=(X_PARP, X_PARP, X_PARP, 0, X_TCP), wait=(sm.K_done, sm.KS_ok), note="x K scale")
                p.V("CAUSAL", Vec(1024, 4, T, dst=O(S_T, 4, 4096, 4 * 4096), a=O(S_T, 4, 4096, 4 * 4096),
                                  par=(("step", lim_step),), override="n"),
                    regs=(X_PARP, X_PARP, X_POSP, X_L, X_TCP))
                p.V("RMAX", Vec(**box, dst=O(Mt, 0, 4), a=O(S_T, 4, 4096)), regs=(X_RML, X_PARP, 0, 0, X_TCP))
                p.V("MAX", Vec(R4, dst=O(Mn), a=O(Mt), b=O(Mold)), regs=(X_RML, X_RML, X_RML))
                p.V("SUB", Vec(**box, dst=O(S_T, 4, 4096), a=O(S_T, 4, 4096), b=O(Mn, 0, 4)),
                    regs=(X_PARP, X_PARP, X_RML, 0, X_TCP))
                p.V("EXP", Vec(**box, dst=O(S_T, 4, 4096), a=O(S_T, 4, 4096)), regs=(X_PARP, X_PARP, 0, 0, X_TCP))
                p.V("RSUM", Vec(**box, dst=O(Lt, 0, 4), a=O(S_T, 4, 4096)), regs=(X_RML, X_PARP, 0, 0, X_TCP))
                p.V("SUB", Vec(R4, dst=O(Al), a=O(Mold), b=O(Mn)), regs=(X_RML, X_RML, X_RML))
                p.V("EXP", Vec(R4, dst=O(Al), a=O(Al)), regs=(X_RML, X_RML))
                p.V("FMA", Vec(R4, dst=O(Ls), a=O(Ls), b=O(Al), c=O(Lt)), regs=(X_RML, X_RML, X_RML, X_RML))
                p.V("COPY", Vec(R4, dst=O(Mold), a=O(Mn)), regs=(X_RML, X_RML))
                p.V("QUANT", Vec(**box, dst=O(P_T, 1, 1024, 0, "i8"), a=O(S_T, 4, 4096), b=O(KS_T + 2048, 2, 0, 0, "f16"),
                                 c=O(PE_T, 1, 32, 0, "e8")), regs=(X_PARP, X_PARP, X_PARP, X_PARP, X_TCP),
                    wait=(sm.P_free,), sig=(sm.P_ok, sm.S_free), note="P (x V scale) -> int8, 2^e per 32")
            p.S("SLLI", X_T2, X_POSP, imm=8)
            p.S("ADD", X_T2, X_T2, X_RK)
            p.S("SRLI", X_T3, X_TCP, imm=8)
            p.D("W", Dma(base + KVH * KH, 0, 256 * 256, 4, 256 * 256, 0), regs=(X_T2, 0, X_T3), note="V^T tile")
            p.M("MM", Mm(1024, HD, R4, P_T, 1024, PE_T, 32, OT_T, HD * 4, wfmt="int8", seg=4, override="n_in"),
                regs=(X_PARP, X_PARP, X_PARP, 0, X_TCP), wait=(sm.P_ok, sm.Ot_free), sig=(sm.Ot_ok, sm.P_free),
                note="P . V")
            p.V("FMA", Vec(HD, R4, dst=O(OG, 4, HD * 4), a=O(OG, 4, HD * 4), b=O(Al, 0, 4), c=O(OT_T, 4, HD * 4)),
                regs=(X_ROG, X_ROG, X_RML, X_PARP), wait=(sm.Ot_ok,), sig=(sm.Ot_free, sm.KS_free),
                note="O = O x alpha + P.V")

        with p.loop(X_CNT, note="KV heads"):
            p.V("COPY", Vec(R4, dst=O(Mold), a=c(C_NINF)), regs=(X_RML,))
            p.V("COPY", Vec(R4, dst=O(Ls), a=c(C_ZERO)), regs=(X_RML,))
            p.V("COPY", Vec(HD, R4, dst=O(OG, 4, HD * 4), a=c(C_ZERO)), regs=(X_ROG,))
            p.S("LI", X_POS, imm=0)
            p.S("LI", X_PAR, imm=0)
            first_half()
            p.S("ADDI", X_T0, X_K2, imm=-1)
            with p.loop(X_T0, note="attention tiles"):
                p.S("ADD", X_POSP, X_POS, 0)
                p.S("ADD", X_TCP, X_TC, 0)
                p.S("ADD", X_PARP, X_PAR, 0)
                p.S("ADDI", X_POS, X_POS, imm=1024)
                p.S("LI", X_T3, imm=SLAB)
                p.S("SUB", X_PAR, X_T3, X_PAR)
                first_half()
                second_half()
            p.S("ADD", X_POSP, X_POS, 0)
            p.S("ADD", X_TCP, X_TC, 0)
            p.S("ADD", X_PARP, X_PAR, 0)
            second_half(last=True)
            # out[t][head 4g + r] = O / l
            p.V("RECIP", Vec(R4, dst=O(Lt), a=O(Ls)), regs=(X_RML, X_RML))
            p.S("SLLI", X_T2, X_G, imm=12)                       # g x 1,024 floats (4 heads x 256)
            p.V("MUL", Vec(HD, 4, T, dst=O(ATT, 4, HD * 4, QD * 4), a=O(OG, 4, HD * 4, 4 * HD * 4), b=O(Lt, 0, 4, 16)),
                regs=(X_T2, X_ROG, X_RML), note="normalize -> output layout")
            for r_, step in ((X_G, 1), (X_RK, KH), (X_RSC, 2 * lmax * 2), (X_RQA, R4 * HD), (X_RQE, R4 * HD // 32),
                             (X_ROG, R4 * HD * 4), (X_RML, MLB)):
                p.S("ADDI", r_, r_, imm=step)

    def attention(att_index, prm_slot, ring=False, src=H, rows_note=""):
        prm_load(prm_slot)
        rmsnorm(src, O(PRM, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready,))
        mm(HID, QPW, XN, XNE, MO, wait=(sm.XN_ready,), dst_stride=MOW * 4, note="q + gate")
        mm(HID, KVH * HD, XN, XNE, MO + QPW * 4, dst_stride=MOW * 4, note="k")
        mm(HID, KVH * HD, XN, XNE, MO + (QPW + KVH * HD) * 4, dst_stride=MOW * 4, sig=(sm.MO_ready,), note="v")
        qn, kn = PRM + 2 * HID * 4, PRM + (2 * HID + HD) * 4
        for (off_, heads, wofs) in ((0, QH, qn), (QPW, KVH, kn)):
            a_ = O(MO + off_ * 4, 4, HD * 4, MOW * 4)
            p.V("RSUMSQ", Vec(HD, heads, T, dst=O(SS, 4, 4, 128), a=a_), wait=(sm.MO_ready,) if off_ == 0 else ())
            p.V("RSQRT", Vec(heads, 1, T, dst=O(SS, 4, 0, 128), a=O(SS, 4, 0, 128), b=c(C_1_256), c=c(C_EPS)))
            p.V("MUL", Vec(HD, heads, T, dst=a_, a=a_, b=O(SS, 0, 4, 128)))
            p.V("MUL", Vec(HD, heads, T, dst=a_, a=a_, b=O(wofs, 4, 0, 0)), regs=(0, 0, X_PRMP))
            p.V("ROPE", Vec(HD, heads, T, dst=a_, a=a_, b=O(CS, 4, 0, 256), par=(("rot", ROT),)))
        for g in range(KVH):                                     # q rows for KV head g: (t, r), x 1/16 exact
            p.V("QUANT", Vec(HD, 4, T, dst=O(QA + g * R4 * HD, 1, HD, 4 * HD, "i8"),
                             a=O(MO + g * 4 * HD * 4, 4, HD * 4, MOW * 4),
                             c=O(QAE + g * R4 * HD // 32, 1, HD // 32, 4 * HD // 32, "e8"), par=(("exp_bias", -4),)),
                sig=(sm.QA_ready,) if g == KVH - 1 else ())
        p.V("QKV", Vec(HD, KVH, T, dst=O(KQ, 1, HD, KVH * HD, "i8"), a=O(MO + QPW * 4, 4, HD * 4, MOW * 4),
                       c=O(KVS, 0, 2, 4 * KVH, "f16")))
        p.V("QKV", Vec(HD, KVH, T, dst=O(VQ, 1, HD, KVH * HD, "i8"), a=O(MO + (QPW + KVH * HD) * 4, 4, HD * 4, MOW * 4),
                       c=O(KVS + 2 * KVH, 0, 2, 4 * KVH, "f16")), sig=(sm.KVQ_L, sm.KVQ_X))
        kv_append(att_index)
        attention_core(att_index, ring)
        p.V("SIGMOID", Vec(QD, 1, T, dst=O(GT, 4, 0, QD * 4), a=O(MO + QD * 4, 4, 0, MOW * 4)))
        p.V("QUANT", Vec(QD, 1, T, dst=O(XN, 1, 0, QD, "i8"), a=O(ATT, 4, 0, QD * 4), b=O(GT, 4, 0, QD * 4),
                         c=O(XNE, 1, 0, QD // 32, "e8")), sig=(sm.XN_ready,), note="gated attention out -> int8")
        mm(QD, HID, XN, XNE, src, "acc", wait=(sm.XN_ready,), sig=(sm.H_ready,), note="o-projection")
        ffn(HID * 4, src)                                        # the MTP layer's FFN works on the MTP residual
        prm_flip()

    # ------------------------------------------------------------- embeddings, rope
    def embed_rows(tok_addr_list, dst=H, rows=None):
        rows = rows or len(tok_addr_list)
        for t, addr in enumerate(tok_addr_list):
            p.S("LI", X_T0, imm=addr)
            p.S("LW", X_T1, X_T0, imm=0)
            p.S("LI", X_T2, imm=EMB_ROW)
            p.S("MUL", X_T1, X_T1, X_T2)
            p.D("R", Dma(EMBT, EMB + t * EMB_ROW, EMB_ROW), regs=(X_T1,), sig=(sm.EMB_ok,) if t == rows - 1 else (),
                note="embedding row")
        p.V("DEQ", Vec(HID, 1, rows, dst=O(dst, 4, 0, HID * 4), a=O(EMB, 1, 0, EMB_ROW, "i8"),
                       b=O(EMB + HID, 2, 0, EMB_ROW, "f16")), wait=(sm.EMB_ok,), sig=(sm.H_ready,), note="embedding")

    def rope_rows(rows):
        for t in range(rows):
            p.S("ADDI", X_T0, X_L, imm=t)
            p.S("SRLI", X_T1, X_T0, imm=10)
            p.S("LI", X_T3, imm=384)
            p.S("MUL", X_T1, X_T1, X_T3)
            p.S("ANDI", X_T2, X_T0, imm=1023)
            p.S("MUL", X_T2, X_T2, X_T3)
            p.D("R", Dma(ROPE, RT + t * 768, 384), regs=(X_T1,), note="rope coarse angle")
            p.D("R", Dma(ROPE + 1024 * 384, RT + t * 768 + 384, 384), regs=(X_T2,), sig=(sm.RT_ok,) if t == rows - 1 else (),
                note="rope fine angle")
        # cos(A+B) = cA cB - sA sB ; sin(A+B) = sA cB + cA sB   (tables hold cos, sin, -sin)
        A, Bq = O(RT, 4, 0, 768), O(RT + 384, 4, 0, 768)
        def f(o, k_):
            return O(o.base + k_ * 128, 4, 0, 768)
        p.V("MUL", Vec(32, 1, rows, dst=O(TMP, 4, 0, 256), a=f(A, 1), b=f(Bq, 2)), wait=(sm.RT_ok,))
        p.V("FMA", Vec(32, 1, rows, dst=O(CS, 4, 0, 256), a=f(A, 0), b=f(Bq, 0), c=O(TMP, 4, 0, 256)))
        p.V("MUL", Vec(32, 1, rows, dst=O(TMP, 4, 0, 256), a=f(A, 1), b=f(Bq, 0)))
        p.V("FMA", Vec(32, 1, rows, dst=O(CS + 128, 4, 0, 256), a=f(A, 0), b=f(Bq, 1), c=O(TMP, 4, 0, 256)))

    # ------------------------------------------------------------- main model pass
    dn_index_slot = {}
    li = 0
    for l in range(NL):
        if (l + 1) % 4:
            dn_index_slot[li] = l
            li += 1

    def main_pass(spec):
        stream_base[0] = W_MAIN
        p.S("LI", X_WP, imm=0)
        rope_rows(T)
        p.S("LI", X_CNT, imm=8)
        # the 8 groups of (3 DeltaNet + 1 attention) layers are written out (each layer has its own offsets);
        # inside each layer, heads / tiles / FFN chunks are hardware loops
        for grp in range(NAT):
            for j in range(3):
                deltanet(grp * 3 + j, spec)
            attention(grp, prm_slot=grp * 4 + 3)
        prm_load(NL)
        if spec:     # the drafts take the final-norm hidden (fp32), as measured best (note 15 section 19: emb_first/post)
            rmsnorm(H, O(PRM, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready,))
            p.V("MUL", Vec(HID, 1, T, dst=O(HPOST, 4, 0, HID * 4), a=O(TMP, 4, 0, HID * 4), b=O(PRM, 4, 0, 0)),
                regs=(0, 0, X_PRMP), sig=(sm.PRM_free,), note="final-norm hidden for the drafts")
        else:
            rmsnorm(H, O(PRM, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready, sm.PRM_free))
        mm(HID, VOCAB, XN, XNE, YB, "argmax", wait=(sm.XN_ready, sm.YB_free) if spec else (sm.XN_ready,),
           sig=(sm.ARG_ok,), dst_stride=4,
           note="LM head, greedy")
        prm_flip()

    def draft_pass(j):
        """MTP draft j (1-based): e = norm(embed(tok)), h = norm(hidden), fc, one attention layer + FFN over the
        MTP's own 4K-position ring of KV, norm, draft LM head (32K vocabulary), id remap. One token row."""
        nonlocal T, R4
        T_save, T, R4 = T, 1, 4
        stream_base[0] = W_MTP
        p.S("LI", X_WP, imm=0)
        # draft j runs at position L + j - 2 (L = the new x0's position): its RoPE angle and its slot in the MTP's
        # KV ring. X_L is moved for the pass and moved back at its end.
        if j != 2:
            p.S("ADDI", X_L, X_L, imm=j - 2)
        rope_rows(1)
        p.S("LI", X_T0, imm=TK + 4 * (j - 1))
        p.S("LW", X_T1, X_T0, imm=0)
        p.S("LI", X_T2, imm=EMB_ROW)
        p.S("MUL", X_T1, X_T1, X_T2)
        p.D("R", Dma(EMBT, EMB, EMB_ROW), regs=(X_T1,), sig=(sm.EMB_ok,), note="draft input embedding")
        p.V("DEQ", Vec(HID, dst=O(EH), a=O(EMB, 1, 0, 0, "i8"), b=O(EMB + HID, 2, 0, 0, "f16")), wait=(sm.EMB_ok,))
        if j == 1:                                               # hidden state of row a of the verify pass
            p.S("SLLI", X_T0, X_A, imm=(HID * 4).bit_length() - 1)   # row a x HID floats
            p.V("COPY", Vec(HID, dst=O(EH + HID * 4), a=O(HPOST)), regs=(0, X_T0))
        else:                                                    # the previous draft's MTP hidden state (final-norm)
            p.V("COPY", Vec(HID, dst=O(EH + HID * 4), a=O(HMTPP)))
        prm_load(NL + 1)                                         # [enorm | hnorm | final norm]
        p.V("RSUMSQ", Vec(HID, 1, 2, dst=O(SS, 4, 0, 4), a=O(EH, 4, 0, HID * 4)))
        p.V("RSQRT", Vec(1, 1, 2, dst=O(SS, 4, 0, 4), a=O(SS, 4, 0, 4), b=c(C_1_4096), c=c(C_EPS)))
        p.V("MUL", Vec(HID, 1, 2, dst=O(EH, 4, 0, HID * 4), a=O(EH, 4, 0, HID * 4), b=O(SS, 0, 0, 4)))
        p.V("QUANT", Vec(HID, 1, 2, dst=O(XN, 1, 0, HID, "i8"), a=O(EH, 4, 0, HID * 4), b=O(PRM, 4, 0, HID * 4),
                         c=O(XNE, 1, 0, HID // 32, "e8")), regs=(0, 0, X_PRMP), sig=(sm.XN_ready,),
            note="[norm e ; norm h] -> one 8,192-wide int8 row")
        p.fence("V", sig=(sm.PRM_free,), note="MTP input norms used")
        prm_flip()
        weights(2 * HID, HID)
        p.M("MM", Mm(2 * HID, HID, 1, XN, 2 * HID, XNE, 2 * HID // 32, HMTP, HID * 4), wait=(sm.XN_ready,),
            sig=(sm.H_ready,), note="MTP fc")
        attention(NAT, prm_slot=NL + 2, ring=True, src=HMTP)
        # final MTP norm (slot 33 again) and the draft LM head
        p.S("LI", X_PRMO, imm=(NL + 1) * PRMB)
        p.D("R", Dma(PRMD, PRM, PRMB), regs=(X_PRMO, X_PRMP), wait=(sm.PRM_free,), sig=(sm.PRM_ok,), note="MTP norm")
        p.fence("V", wait=(sm.PRM_ok,))
        rmsnorm(HMTP, O(PRM + 2 * HID * 4, 4, 0, 0), wait=(sm.H_ready,), sig=(sm.XN_ready,), rows=1)
        p.V("MUL", Vec(HID, dst=O(HMTPP), a=O(TMP), b=O(PRM + 2 * HID * 4)), regs=(0, 0, X_PRMP), sig=(sm.PRM_free,),
            note="MTP final-norm hidden for the next draft")
        prm_flip()
        weights(HID, DRAFT_VOCAB)
        p.M("MM", Mm(HID, DRAFT_VOCAB, 1, XN, HID, XNE, HID // 32, DI, 4, epi="argmax"), wait=(sm.XN_ready,),
            sig=(sm.ARG_ok,), note="draft LM head, greedy")
        p.S("WAITS", imm=sm.ARG_ok)
        p.S("LI", X_T0, imm=DI)
        p.S("LW", X_T1, X_T0, imm=0)
        p.S("SLLI", X_T1, X_T1, imm=2)
        p.D("R", Dma(REMAP, TK + 4 * j, 4), regs=(X_T1,), sig=(sm.TOK_ok,), note="draft id -> token id")
        p.S("WAITS", imm=sm.TOK_ok)
        if j != 2:
            p.S("ADDI", X_L, X_L, imm=2 - j)
        T, R4 = T_save, 4 * T_save

    # ------------------------------------------------------------- the program
    for s_, n in (("PRM_free", 2), ("CH_free", 1), ("CW_free", 1), ("CYX_free", 1), ("LOG_free", 1),
                  ("SB_free", 1), ("GU_free", 2), ("S_free", 2), ("KS_free", 2), ("P_free", 2), ("Ot_free", 2),
                  ("YB_free", 1 if mode == "spec" else 0)):
        for _ in range(n):
            p.S("SIGS", imm=getattr(sm, s_))
    p.S("LI", X_PRMP, imm=0)
    if mode == "decode":
        p.S("LI", X_NC, imm=1)               # plain decode commits 1 token per step: the conv window moves 1 row
    p.S("LI", 30, imm=steps)
    with p.loop(30, note="steps"):
        if mode == "decode":
            embed_rows([YB])
            main_pass(False)
            p.S("WAITS", imm=sm.ARG_ok)
            p.S("ADDI", X_L, X_L, imm=1)
        else:
            embed_rows([TK + 4 * t for t in range(T)])
            main_pass(True)
            p.fence("V", wait=(sm.ARG_ok,), note="main argmax done")
            p.V("MATCH", Vec(k, dst=O(ACC, 4, 0, 0, "u32"), a=O(YB, 4, 0, 0, "u32"), b=O(TK + 4, 4, 0, 0, "u32")),
                sig=(sm.ACC_ok,), note="count accepted drafts")
            p.S("WAITS", imm=sm.ACC_ok)
            p.S("LI", X_T0, imm=ACC)
            p.S("LW", X_A, X_T0, imm=0)
            p.S("SLLI", X_T1, X_A, imm=2)
            p.S("LW", X_T2, X_T1, imm=YB)
            p.S("LI", X_T0, imm=TK)
            p.S("SW", 0, X_T0, X_T2, imm=0)                      # x0 for the next step = main token after a
            # this step's tokens are YB[0 .. a]: the a accepted drafts (equal to the main model's own) and the main
            # token after them. Count a + 1 >= 1 (a count register of 0 would mean "the descriptor's count").
            p.S("ADDI", X_T3, X_A, imm=1)
            p.D("S2H", Dma(YB, 0, 4, k + 1, s1_src=4, s1_dst=4), regs=(0, 0, X_T3), sig=(sm.YB_free,), note="this step's tokens -> host")
            p.S("ADDI", X_L, X_L, imm=1)
            p.S("ADD", X_L, X_L, X_A)
            p.S("ADDI", X_NC, X_A, imm=1)
            for j in range(1, k + 1):
                draft_pass(j)
    p.S("HALT")

    info = dict(T=T, k=k, mode=mode, sram=sa, dram=da, sems=sm, main_stream=main_stream, mtp_stream=mtp_stream,
                KV_LAYER=KV_LAYER, lmax=lmax, cfg=cfg, PRMB=PRMB, EMB_ROW=EMB_ROW, MOW=MOW, CYW=CYW,
                QKV_CH=QKV_CH, VD=VD, QD=QD, QPW=QPW, N_CHUNK=N_CHUNK, NDN=NDN, NAT=NAT,
                bufs=dict(H=H, XN=XN, XNE=XNE, YB=YB, TK=TK, CONST=CONST, MO=MO, CY=CY, DNO=DNO, ATT=ATT, TMP=TMP),
                dbase=dict(W_MAIN=W_MAIN, W_MTP=W_MTP, EMBT=EMBT, KVB=KVB, STB=STB, CLOG=CLOG, DLOG=DLOG, PRMD=PRMD,
                          ROPE=ROPE, REMAP=REMAP))
    return p, info
