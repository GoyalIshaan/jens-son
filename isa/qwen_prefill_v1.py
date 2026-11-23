"""Prefill in ISA v1: chunks of C = 128 prompt tokens through the whole model (note 28, part 2; note 31, part 2).

Differences from decode (qwen_v1.py), all forced by SRAM: 128 rows of fp32 are 2 MB per 4,096 values.
  - Projections are made head by head (budget note 27): per DeltaNet layer, b/a for all heads, then for each
    key head its q,k (256 outputs) and for each of its value heads v,z (256 outputs); per attention layer, per
    KV head its 4 query heads' q (1,024), k (256), v (256), and after attention their gate (1,024).
  - Each head's output is normalized and rounded to int8 straight into the out-projection's input.
  - Attention reads K and V in 256-position blocks into SRAM (MMS: weights from SRAM), and all 4 query heads of
    the KV head reuse each block, so the KV cache is read once per chunk.
  - The new chunk's K rows, V^T (128 positions = half a block, so whole 128-byte rows) and scales are written
    before attention reads them back.
  - FFN in chunks of 256 outputs; RMSNorm in 8 groups of 16 rows.
The chunk starts at position L (a multiple of 128). DeltaNet state is committed directly (no speculation).
The model is a parameter (`cfg`, qwen_v1.Model); the default is the 9B.
Token ids: the host leaves them in SRAM at TOKB (one u32 per prompt token, all chunks).
"""
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_v1 as I  # noqa: E402
from isa_v1 import Dma, Mm, Opnd as O, Prog, Vec  # noqa: E402
from qwen_v1 import Alloc, Sems, wbytes, WHDR, Q9B  # noqa: E402
from qwen_v1 import (X_L, X_NC, X_WP, X_PRMO, X_PRMP, X_T0, X_T1, X_T2, X_T3, X_CNT, X_G, X_RK, X_RSC,  # noqa: E402
                     X_POS, X_PAR, X_LR, X_H, X_SBP, X_K1, X_K2)

C = 128
FFC = 256
X_KK, X_RG = 22, 25          # key head x 512 bytes (conv history offset); rmsnorm row-group offset
X_TOK, X_TAP, X_GO, X_GA, X_BO, X_HB = 2, 17, 18, 19, 20, 24
X_GX, X_GXE = 5, 6           # attention: KV head g x 1,024 bytes into the o-projection input, and / 32 for its exponents
# token-id pointer; conv taps offset (PRM parity + head); lite DeltaNet per head (+ chunk) offsets into the G
# scan, the gamma / ratio rows and the beta rows; value head x 512 bytes


def build_prefill(chunks=1, lmax=40_000, fused=False, dn="seq", cfg=Q9B):
    assert dn in ("seq", "chunked", "lite")
    HID, VOCAB, QH, KVH, HD, ROT = cfg.hid, cfg.vocab, cfg.qh, cfg.kvh, cfg.hd, cfg.rot
    LQK, LV, LD, NL = cfg.lqk, cfg.lv, cfg.ld, cfg.layers
    FFN = cfg.ffn_p
    assert QH == 4 * KVH and HD == 256 and LD == 128 and NL % 4 == 0 and HID % 1024 == 0 and LV % LQK == 0
    NDN, NAT = NL * 3 // 4, NL // 4
    QKV_CH = 2 * LQK * LD + LV * LD            # conv channels: q | k | v
    KOFF, VOFF = LQK * LD, 2 * LQK * LD        # where k and v start in them
    VD, QD = LV * LD, QH * HD                  # DeltaNet / attention out-projection inputs
    XOW = max(VD, QD)
    VPK = LV // LQK                            # value heads per key head
    BAW = -(-2 * LV // 64) * 64                # b | a, padded to whole 64-output tiles
    BDR = 2 * LV * 4                           # beta | decay row (bytes)
    LGR = LV * 4                               # one token's row of per-head scalars (bytes)
    EMB_ROW = HID + HID // 32 * 2
    PRMB = max((2 * HID + QKV_CH * 4 + 2 * LV + LD) * 4, (2 * HID + 2 * HD) * 4, (5 * HID + 2 * HD) * 4)
    P_DT = (2 * HID + 4 * QKV_CH) * 4          # dt_bias | -exp(A_log) | gated-norm weight, in a layer's parameters
    STATE_HEAD = LD * LD * 4
    p = Prog()
    sm = Sems()
    sa = Alloc("SRAM", I.SRAM_BYTES)
    da = Alloc("DRAM")
    CONST = sa("constants", 256)
    TOKB = sa("prompt token ids (from the host)", chunks * C * 4)
    # always live, or filled ahead of time by DMA (never shared)
    H = sa("residual stream, 128 tokens (fp32)", C * HID * 4)
    XN = sa("normed layer input (int8)", C * HID)
    XNE = sa("  its exponents", C * HID // 32)
    TMP = sa("vector temporary (fp32)", max(16 * HID, C * HD, C * LD) * 4)
    SS = sa("per-row scalars", C * 64 * 4)
    PRM = sa("layer parameters x2", 2 * PRMB)
    CH = sa("conv history, one layer", 3 * QKV_CH * 4)
    SB = sa("DeltaNet state x2 (own banks)", 2 * STATE_HEAD)
    BLK = 2 * 256 * HD + 2 * 256 * 2
    KB = sa("K, V^T blocks + scales x2", 2 * BLK)
    RT = sa("rope rows + cos/sin", C * (768 + 256))
    # phase buffers: FFN, DeltaNet and attention compute buffers share one region (a layer uses one at a time;
    # each is written only by the matrix or vector unit, which run in program order, or by a DMA that waits for it)
    U = sa.top = (sa.top + 63) // 64 * 64
    phase = {}

    def sub(ph, what, size):
        a = phase.setdefault(ph, Alloc(ph, base=U))
        return a(what, size)
    FF8 = sub("ffn", "FFN intermediate (int8)", C * FFN)
    FF8E = sub("ffn", "  its exponents", C * FFN // 32)
    GU = sub("ffn", "FFN gate/up chunk x2", 2 * C * 2 * FFC * 4)
    XO_d = sub("dn", "out-projection input (int8)", C * XOW)
    XOE_d = sub("dn", "  its exponents", C * XOW // 32)
    BA = sub("dn", "DeltaNet b, a (fp32)", C * BAW * 4)
    BD = sub("dn", "beta, decay (fp32)", C * BDR)
    CWO = sub("dn", "conv history out, one layer", 3 * QKV_CH * 4)
    QKB = sub("dn", "one key head's q,k projections (fp32)", C * 256 * 4)
    VZB = sub("dn", "one value head's v,z projections (fp32)", C * 256 * 4)
    CQK = sub("dn", "conv'd q,k of one key head", C * 256 * 4)
    CV = sub("dn", "conv'd v of one value head", C * 128 * 4)
    DO = sub("dn", "one head's DeltaNet output", C * 128 * 4)
    EMB = sub("dn", "embedding rows (chunk start only)", C * EMB_ROW)

    def wside(rows):      # weight-side int8 operand: hi rows, lo rows (256 inputs each), fp16 scale, fp32 256/s
        return 2 * rows * 256 + 8 * rows

    def aside(rows):      # activation-side int8 operand: hi rows, lo rows, 16 exponent bytes per row (hi | lo)
        return 2 * rows * 256 + 16 * rows
    if dn == "lite":      # chunked delta rule from existing commands only (note 30, part 2): operands zero-padded to
        #                   whole 256-input segments (no lane mask), scales applied by the vector unit
        LQ = sub("dn", "k, q int8 hi, lo (padded to 256) + exps", aside(128))
        LKW = sub("dn", "k weight side hi, lo (padded) + fp16 scales + 256/s", wside(64))
        LSW = sub("dn", "state columns hi, lo (padded) + scales", wside(128))
        LKK = sub("dn", "K K^T, Q K^T (fp32)", 2 * 64 * 64 * 4)
        LKS = sub("dn", "S^T k, S^T q (fp32)", 2 * 64 * 128 * 4)
        LD_ = sub("dn", "decay D, -A, P (fp32)", 3 * 64 * 64 * 4)
        LPQ = sub("dn", "P hi, lo (padded) + exps", aside(64))
        LR = sub("dn", "r -> Delta (fp32)", 64 * 128 * 4)
        LDW = sub("dn", "Delta columns hi, lo (padded) + scales", wside(128))
        LKT = sub("dn", "Ktil^T hi, lo (padded) + exps", aside(128))
        LG = sub("dn", "G scan x2 (zero-padded), gamma, gamma_C ratios (all heads)", 2 * 2 * C * LGR + 2 * C * LGR)
        LPD = sub("dn", "P Delta (fp32)", 64 * 128 * 4)
        LUP = sub("dn", "Ktil, then Ktil^T Delta (fp32)", 128 * 128 * 4)
        LTM = sub("dn", "temporaries (fp32)", 128 * 128 * 4)
        G0, G1 = LG, LG + 2 * C * LGR          # [chunk][64 zero rows | 64 rows][head]: the scan reads row t - off
        LG2, LG3 = LG + 4 * C * LGR, LG + 5 * C * LGR   # gamma_t, gamma_C / gamma_s: [token][head]
        LA, LP = LD_ + 64 * 256, LD_ + 2 * 64 * 256
    if dn == "chunked":   # chunked delta rule (note 30): int8 hi + lo operand copies and fp32 intermediates
        DQ = sub("dn", "q, k int8 hi+lo (activation side) + exps", 2 * C * 256 + 2 * C * 8)
        DKW = sub("dn", "k int8 hi+lo (weight side) + scales", 2 * C * 128 + 2 * C * 2)
        DSQ = sub("dn", "state copy int8 hi+lo per column + scales", 2 * 128 * 128 + 2 * 128 * 2)
        DKK = sub("dn", "K K^T, Q K^T (fp32, 64 x 64 x 2)", 2 * 64 * 64 * 4)
        DKS = sub("dn", "S^T k, S^T q (fp32, 64 x 128 x 2)", 2 * 64 * 128 * 4)
        DM = sub("dn", "decay D, A, P (fp32) + P hi+lo", 3 * 64 * 64 * 4 + 2 * 64 * 64 + 2 * 64 * 2)
        DDL = sub("dn", "Delta (fp32) + hi+lo per column", 64 * 128 * 4 + 2 * 64 * 128 + 2 * 128 * 2)
        DKT = sub("dn", "Ktil^T hi+lo (activation side) + exps", 2 * 128 * 64 + 2 * 128 * 2)
        DPD = sub("dn", "P Delta (fp32)", 64 * 128 * 4)
    XO = sub("att", "out-projection input (int8)", C * XOW)
    XOE = sub("att", "  its exponents", C * XOW // 32)
    assert (XO, XOE) == (XO_d, XOE_d)
    QG = sub("att", "4 query heads' q, later their gate (fp32)", C * 1024 * 4)
    Q4 = sub("att", "q int8, 4 heads", 4 * C * HD)
    Q4E = sub("att", "  its exponents", 4 * C * HD // 32)
    KF = sub("att", "new k (fp32)", C * HD * 4)
    VF = sub("att", "new v (fp32)", C * HD * 4)
    KQ = sub("att", "new K int8 + scales", C * HD + C * 2)
    VQ = sub("att", "new V int8 + scales", C * HD + C * 2)
    VTT = sub("att", "new V transposed (256 x 128)", HD * C)
    # Bank layout (note 34): each slab is whole 4 KB stripes, so both ping-pong slots sit alike on the 64
    # banks, and OA starts 2 KB (half the banks) off OT_T. OA = OA x alpha + OT reads both, and with the
    # same offset in the stripe both landed on the same 32 banks.
    slab = -(-(C * 256 * 4 + C * 256 + C * 8 + C * HD * 4) // 4096) * 4096
    TILE = sub("att", "score / P / P.V slabs x2", 2 * slab)
    S_T, P_T, PE_T, OT_T = TILE, TILE + C * 1024, TILE + C * 1024 + C * 256, TILE + C * 1024 + C * 256 + C * 8
    att = phase["att"]
    att.top = (att.top + 63) // 64 * 64
    att.top += (OT_T + 2048 - att.top) % 4096
    OA = sub("att", "attention accumulators, 4 query heads", 4 * C * HD * 4)
    assert (OA - OT_T) % 4096 == 2048
    ML = sub("att", "softmax m, l, alpha, 4 heads", 4 * 6 * C * 4)
    sa.items.append(("phase region (max of: " + ", ".join(f"{k} {v.total() - U >> 10} KB" for k, v in phase.items())
                     + ")", U, max(v.total() for v in phase.values()) - U))
    sa.top = max(v.total() for v in phase.values())
    sa.phase = phase
    assert sa.top <= I.SRAM_BYTES, f"SRAM overflow: {sa.top}"

    def layer_bytes(kind):
        if kind == "dn":
            b = wbytes(HID, BAW) + LQK * wbytes(HID, 256) + LV * wbytes(HID, 256) + wbytes(VD, HID)
        else:
            b = KVH * (2 * wbytes(HID, 1024) + 2 * wbytes(HID, 256)) + wbytes(QD, HID)
        return b + 2 * (FFN // FFC) * wbytes(HID, FFC) + wbytes(FFN, HID)
    main_stream = sum(layer_bytes("at" if (l + 1) % 4 == 0 else "dn") for l in range(NL))
    W_MAIN = da("main weights in prefill stream order", main_stream)
    EMBT = da("embedding table", VOCAB * EMB_ROW)
    KH = lmax * HD
    KV_LAYER = 2 * KVH * KH + KVH * 2 * lmax * 2
    KVB = da("KV cache", NAT * KV_LAYER)
    STB = da("DeltaNet state", NDN * LV * STATE_HEAD)
    CLOG = da("conv history", NDN * 3 * QKV_CH * 4)
    PRMD = da("parameters", (NL + 3) * PRMB)
    ROPE = da("rope tables", 2 * 1024 * 384)

    def c(i):
        return O(CONST + 4 * i, 0, 0, 0)
    C_1_HID, C_EPS, C_1E6, C_1_128, C_RS128, C_NINF, C_ZERO, C_1_256, C_M126 = range(9)

    def weights(n_in, n_out):
        n = wbytes(n_in, n_out)
        tile = 64 * 576
        tiles = n_in * n_out // (1024 * 64)
        p.D("W", Dma(W_MAIN, 0, WHDR + tile, 1), regs=(X_WP,), note="header + first tile")
        if tiles > 1:
            p.D("W", Dma(W_MAIN + WHDR + tile, 0, tile, tiles - 1, tile, 0), regs=(X_WP,), note=f"weights {n_in}x{n_out}")
        p.S("ADDI", X_WP, X_WP, imm=n)
        p.M("TAB")

    def mm(n_in, n_out, act, exps, dst, dst_stride, epi="f32", wait=(), sig=(), regs=(), note=""):
        weights(n_in, n_out)
        p.M("MM", Mm(n_in, n_out, C, act, n_in, exps, n_in // 32, dst, dst_stride, epi=epi), regs=regs, wait=wait,
            sig=sig, note=note)

    def rmsnorm(w_off, wait, sig):
        """8 groups of 16 rows: sum of squares, 1/sqrt, x * r -> TMP, int8(TMP * w)."""
        p.S("LI", X_RG, imm=0)
        p.S("LI", X_T2, imm=0)
        p.S("LI", X_T3, imm=0)
        p.S("LI", X_CNT, imm=8)
        p.fence("V", wait=wait)
        with p.loop(X_CNT, note="rmsnorm row groups"):
            p.V("RSUMSQ", Vec(HID, 1, 16, dst=O(SS, 4, 0, 4), a=O(H, 4, 0, HID * 4)), regs=(0, X_RG))
            p.V("RSQRT", Vec(1, 1, 16, dst=O(SS, 4, 0, 4), a=O(SS, 4, 0, 4), b=c(C_1_HID), c=c(C_EPS)))
            p.V("MUL", Vec(HID, 1, 16, dst=O(TMP, 4, 0, HID * 4), a=O(H, 4, 0, HID * 4), b=O(SS, 0, 0, 4)),
                regs=(0, X_RG))
            p.V("QUANT", Vec(HID, 1, 16, dst=O(XN, 1, 0, HID, "i8"), a=O(TMP, 4, 0, HID * 4),
                             b=O(PRM + w_off, 4, 0, 0), c=O(XNE, 1, 0, HID // 32, "e8")), regs=(X_T2, 0, X_PRMP, X_T3))
            p.S("ADDI", X_RG, X_RG, imm=16 * HID * 4)
            p.S("ADDI", X_T2, X_T2, imm=16 * HID)
            p.S("SRLI", X_T3, X_T2, imm=5)
        p.fence("V", sig=sig, note="rmsnorm done")

    def prm_load(slot):
        p.S("LI", X_PRMO, imm=slot * PRMB)
        p.D("R", Dma(PRMD, PRM, PRMB), regs=(X_PRMO, X_PRMP), wait=(sm.PRM_free,), sig=(sm.PRM_ok,))
        p.fence("V", wait=(sm.PRM_ok,))

    def prm_flip():
        p.S("LI", X_T3, imm=PRMB)
        p.S("SUB", X_PRMP, X_T3, X_PRMP)

    def ffn(after_dn=False, last=False):
        rmsnorm(HID * 4, (sm.H_ready,), (sm.XN_ready, sm.PRM_free))
        p.fence("M", wait=(sm.XN_ready, sm.CWO_free) if after_dn else (sm.XN_ready,),
                note="FFN input ready" + (" (and the conv history is out of the shared region)" if after_dn else ""))
        p.S("LI", X_CNT, imm=FFN // FFC)
        p.S("LI", X_T0, imm=0)
        p.S("LI", X_T1, imm=0)
        p.S("LI", X_T2, imm=0)
        half = C * 2 * FFC * 4
        with p.loop(X_CNT, note="FFN chunks"):
            weights(HID, FFC)
            p.M("MM", Mm(HID, FFC, C, XN, HID, XNE, HID // 32, GU, 2 * FFC * 4), regs=(0, 0, X_T0),
                wait=(sm.GU_free,), note="gate chunk")
            weights(HID, FFC)
            p.M("MM", Mm(HID, FFC, C, XN, HID, XNE, HID // 32, GU + FFC * 4, 2 * FFC * 4), regs=(0, 0, X_T0),
                sig=(sm.GU_ok,), note="up chunk")
            p.V("SILU", Vec(FFC, 1, C, dst=O(TMP, 4, 0, FFC * 4), a=O(GU, 4, 0, 2 * FFC * 4)), regs=(0, X_T0),
                wait=(sm.GU_ok,))
            p.V("QUANT", Vec(FFC, 1, C, dst=O(FF8, 1, 0, FFN, "i8"), a=O(TMP, 4, 0, FFC * 4),
                             b=O(GU + FFC * 4, 4, 0, 2 * FFC * 4), c=O(FF8E, 1, 0, FFN // 32, "e8")),
                regs=(X_T1, 0, X_T0, X_T2), sig=(sm.GU_free, sm.FF_part))
            p.S("LI", X_T3, imm=half)
            p.S("SUB", X_T0, X_T3, X_T0)
            p.S("ADDI", X_T1, X_T1, imm=FFC)
            p.S("ADDI", X_T2, X_T2, imm=FFC // 32)
        p.S("LI", X_CNT, imm=FFN // FFC - 1)
        with p.loop(X_CNT):
            p.fence("M", wait=(sm.FF_part,))
        # the last layer's FFN-down frees the residual and the embedding rows for the next prompt chunk
        mm(FFN, HID, FF8, FF8E, H, HID * 4, "acc", wait=(sm.FF_part,),
           sig=(sm.H_free, sm.EMB_free) if last else (sm.H_ready,), note="FFN down")

    # ------------------------------------------------------------- DeltaNet layer, head by head
    def deltanet(dn_index, slot, last):
        prm_load(slot)
        rmsnorm(0, (sm.H_ready,), (sm.XN_ready,))
        mm(HID, BAW, XN, XNE, BA, BAW * 4, wait=(sm.XN_ready,), sig=(sm.BA_ok,), note="b, a (all heads)")
        p.V("SIGMOID", Vec(LV, 1, C, dst=O(BD, 4, 0, BDR), a=O(BA, 4, 0, BAW * 4)), wait=(sm.BA_ok,), note="beta")
        p.V("ADD", Vec(LV, 1, C, dst=O(BD + LV * 4, 4, 0, BDR), a=O(BA + LV * 4, 4, 0, BAW * 4),
                       b=O(PRM + P_DT, 4, 0, 0)), regs=(0, 0, X_PRMP))
        p.V("SOFTPLUS", Vec(LV, 1, C, dst=O(BD + LV * 4, 4, 0, BDR), a=O(BD + LV * 4, 4, 0, BDR)))
        if dn == "lite":
            # g = -exp(A_log) softplus(.) straight into the scan's padded rows; G = cumulative g inside each 64-token
            # chunk, all heads: a Hillis-Steele scan, 6 steps, ping-pong between two arrays whose 64 rows in front
            # of each chunk are zero, so row t - off reads 0 for t < off (R8's companion rule, dn_chunked.scan_hs)
            for G_ in (G0, G1):
                p.V("COPY", Vec(LV, 64, 2, dst=O(G_, 4, LGR, 2 * 64 * LGR), a=c(C_ZERO)), note="zero rows of the scan")
            p.V("MUL", Vec(LV, 64, 2, dst=O(G0 + 64 * LGR, 4, LGR, 2 * 64 * LGR), a=O(BD + LV * 4, 4, BDR, 64 * BDR),
                           b=O(PRM + P_DT + LV * 4, 4, 0, 0)), regs=(0, 0, X_PRMP), note="log decay g")
            for st in range(6):
                off = 1 << st
                src, dst_ = (G0, G1) if st % 2 == 0 else (G1, G0)
                p.V("ADD", Vec(LV, 64, 2, dst=O(dst_ + 64 * LGR, 4, LGR, 2 * 64 * LGR),
                               a=O(src + 64 * LGR, 4, LGR, 2 * 64 * LGR),
                               b=O(src + (64 - off) * LGR, 4, LGR, 2 * 64 * LGR)), note="scan step for G")
            p.V("EXP", Vec(LV, 64, 2, dst=O(LG2, 4, LGR, 64 * LGR), a=O(G0 + 64 * LGR, 4, LGR, 2 * 64 * LGR)),
                note="gamma")
            p.V("SUB", Vec(LV, 64, 2, dst=O(LG3, 4, LGR, 64 * LGR), a=O(G0 + 127 * LGR, 4, 0, 2 * 64 * LGR),
                           b=O(G0 + 64 * LGR, 4, LGR, 2 * 64 * LGR)))
            p.V("EXP", Vec(LV, C, 1, dst=O(LG3, 4, LGR, 0), a=O(LG3, 4, LGR, 0)), note="gamma_C / gamma_s")
            # the int8 operands' padding (inputs n..255): zero codes, and exponent -126 (an all-zero group's) so a
            # pad group never sets a segment's emax. Rewritten each layer: the region is shared with the FFN.
            for base, n, rows in ((LQ, 128, 128), (LPQ, 64, 64), (LKT, 64, 128)):
                p.V("COPY", Vec(256 - n, 2 * rows, 1, dst=O(base + n, 1, 256, 0, "i8"), a=c(C_ZERO)), note="pad codes")
                p.V("COPY", Vec(8 - n // 32, rows, 2, dst=O(base + 2 * rows * 256 + n // 32, 1, 16, 8, "e8"),
                                a=c(C_M126)), note="pad exponents")
            for base, n, rows in ((LKW, 128, 64), (LSW, 128, 128), (LDW, 64, 128)):
                p.V("COPY", Vec(256 - n, 2 * rows, 1, dst=O(base + n, 1, 256, 0, "i8"), a=c(C_ZERO)), note="pad codes")
        else:
            p.V("MUL", Vec(LV, 1, C, dst=O(BD + LV * 4, 4, 0, BDR), a=O(BD + LV * 4, 4, 0, BDR),
                           b=O(PRM + P_DT + LV * 4, 4, 0, 0)), regs=(0, 0, X_PRMP))
            p.V("EXP", Vec(LV, 1, C, dst=O(BD + LV * 4, 4, 0, BDR), a=O(BD + LV * 4, 4, 0, BDR)), note="decay")

        def split_act(src, n, m, pp, dst, what, reg=0):
            """activation side, box (n, m, pp) -> m*pp rows: hi = int8 + 2^e per 32; residual = x - hi 2^e (an e8
            operand reads 2^e of its 32-group); lo = int8 of the residual (3 cmds)"""
            rows = m * pp
            E = dst + 2 * rows * 256
            p.V("QUANT", Vec(n, m, pp, dst=O(dst, 1, 256, m * 256, "i8"), a=src, c=O(E, 1, 16, m * 16, "e8")),
                regs=(0, reg), note=f"{what}: hi")
            p.V("FMA", Vec(n, m, pp, dst=O(LTM, 4, n * 4, m * n * 4), a=O(dst, 1, 256, m * 256, "i8"),
                           b=O(E, 1, 16, m * 16, "e8"), c=src, par=(("neg", 1),)), regs=(0, 0, 0, reg),
                  note=f"{what}: residual")
            p.V("QUANT", Vec(n, rows, 1, dst=O(dst + rows * 256, 1, 256, 0, "i8"), a=O(LTM, 4, n * 4, 0),
                             c=O(E + 8, 1, 16, 0, "e8")), note=f"{what}: lo")

        def split_w(src, n, rows, dst, what, reg=0):
            """weight side, box (n, rows): hi = int8 + one fp16 scale s per row (QKV); residual;
            lo = int8(residual * 256/s), 256/s = 1/(s * 2^-8) (5 cmds)"""
            S16, RCP = dst + 2 * rows * 256, dst + 2 * rows * 256 + 4 * rows
            p.V("QKV", Vec(n, rows, 1, dst=O(dst, 1, 256, 0, "i8"), a=src, c=O(S16, 0, 2, 0, "f16")), regs=(0, reg),
                note=f"{what}: hi + scale")
            p.V("FMA", Vec(n, rows, 1, dst=O(LTM, 4, n * 4, 0), a=O(dst, 1, 256, 0, "i8"), b=O(S16, 0, 2, 0, "f16"),
                           c=src, par=(("neg", 1),)), regs=(0, 0, 0, reg), note=f"{what}: residual")
            p.V("RECIP", Vec(rows, 1, 1, dst=O(RCP, 4, 0, 0), a=O(S16, 2, 0, 0, "f16"), b=c(C_1_256)),
                note=f"{what}: 256/s")
            p.V("MUL", Vec(n, rows, 1, dst=O(LTM, 4, n * 4, 0), a=O(LTM, 4, n * 4, 0), b=O(RCP, 0, 4, 0)),
                note=f"{what}: x 256/s")
            p.V("COPY", Vec(n, rows, 1, dst=O(dst + rows * 256, 1, 256, 0, "i8"), a=O(LTM, 4, n * 4, 0)),
                note=f"{what}: lo")

        def mm3(n_out, rows, act, exps, wsrc, w_rows, dst, dst_stride, wait=(), sig=(), note=""):
            """3 int8 passes, n_in zero-padded to one 256-input segment: hi*hi, lo*hi, hi*lo (x 2^-8)"""
            p.M("MMS", Mm(256, n_out, rows, act, 256, exps, 16, dst, dst_stride, wfmt="int8", seg=4, wsrc=wsrc),
                wait=wait, note=note + " hi*hi")
            p.M("MMS", Mm(256, n_out, rows, act + rows * 256, 256, exps + 8, 16, dst, dst_stride, wfmt="int8", seg=4,
                          epi="acc", wsrc=wsrc), note=note + " lo*hi")
            p.M("MMS", Mm(256, n_out, rows, act, 256, exps, 16, dst, dst_stride, wfmt="int8", seg=4, epi="acc",
                          wsrc=wsrc + w_rows * 256, tscale=2.0 ** -8), sig=sig, note=note + " hi*lo")

        def lite_head(qk):
            """One value head, 128 tokens = two 64-token chunks, from existing commands only (note 30, part 2)."""
            # the two 64-token halves as a hardware loop: X_POS steps the q/k rows (64 x 1,024 B), X_PAR the
            # v / output rows (64 x 512 B), X_GO / X_GA / X_BO the per-head scalar rows; the state wait and the done
            # signal sit outside
            p.fence("V", wait=(sm.S_ok,))
            p.S("LI", X_POS, imm=0)
            p.S("LI", X_PAR, imm=0)
            p.S("SLLI", X_GO, X_H, imm=2)
            p.S("ADD", X_GA, X_GO, 0)
            p.S("ADD", X_BO, X_GO, 0)
            p.S("LI", X_T0, imm=2)
            Gc = G0 + 64 * LGR
            with p.loop(X_T0, note="two 64-token chunks"):
                split_act(O(qk + 512, 4, 1024, -512), 128, 64, 2, LQ, "k, q (act)", reg=X_POS)   # rows: k, then q
                split_w(O(qk + 512, 4, 1024, 0), 128, 64, LKW, "k (weight)", reg=X_POS)
                split_w(O(SB, 512, 4, 0), 128, 128, LSW, "state columns (weight)", reg=X_SBP)
                p.V("SUB", Vec(64, 1, 64, dst=O(LD_, 4, 0, 256), a=O(Gc, 0, 0, LGR), b=O(Gc, LGR, 0, 0)),
                    regs=(0, X_GO, X_GO), note="G_t - G_s")
                p.V("CAUSAL", Vec(64, 1, 64, dst=O(LD_, 4, 0, 256), a=O(LD_, 4, 0, 256), par=(("step", 1),)))
                p.V("EXP", Vec(64, 64, 1, dst=O(LD_, 4, 256, 0), a=O(LD_, 4, 256, 0)), sig=(sm.DNP,), note="decay D")
                mm3(64, 128, LQ, LQ + 2 * 128 * 256, LKW, 64, LKK, 256, wait=(sm.DNP,), note="K K^T, Q K^T")
                mm3(128, 128, LQ, LQ + 2 * 128 * 256, LSW, 128, LKS, 512, sig=(sm.DNA,), note="S^T k, S^T q")
                # -A, P, r with the weight-side scales applied here
                p.V("MUL", Vec(64, 64, 2, dst=O(LKK, 4, 256, 64 * 256), a=O(LKK, 4, 256, 64 * 256),
                               b=O(LKW + 2 * 64 * 256, 2, 0, 0, "f16")), wait=(sm.DNA,), note="K K^T, Q K^T x s_k")
                p.V("MUL", Vec(64, 64, 2, dst=O(LA, 4, 256, 64 * 256), a=O(LKK, 4, 256, 64 * 256),
                               b=O(LD_, 4, 256, 0)), note="x D -> A, P")
                p.V("MUL", Vec(64, 64, 1, dst=O(LA, 4, 256, 0), a=O(LA, 4, 256, 0), b=O(BD, 0, BDR, 0),
                               par=(("neg", 1),)), regs=(0, 0, X_BO), note="A x (-beta): the solve adds -A")
                p.V("MUL", Vec(128, 64, 2, dst=O(LKS, 4, 512, 64 * 512), a=O(LKS, 4, 512, 64 * 512),
                               b=O(LSW + 2 * 128 * 256, 2, 0, 0, "f16")), note="S^T k, S^T q x s_S")
                p.V("FMA", Vec(128, 64, 1, dst=O(LR, 4, 512, 0), a=O(LKS, 4, 512, 0), b=O(LG2, 0, LGR, 0),
                               c=O(CV, 4, 512, 0), par=(("neg", 1),)), regs=(0, 0, X_GA, X_PAR),
                      note="v - gamma S^T k")
                p.V("MUL", Vec(128, 64, 1, dst=O(LR, 4, 512, 0), a=O(LR, 4, 512, 0), b=O(BD, 0, BDR, 0)),
                    regs=(0, 0, X_BO), note="r")
                # the solve: 63 right-looking steps, r[s+1:] = fma(-A[s+1:, s], Delta_s, r[s+1:])  (R8)
                p.S("LI", X_G, imm=512)
                p.S("LI", X_RK, imm=0)
                p.S("LI", X_RSC, imm=0)
                p.S("LI", X_LR, imm=63)
                p.S("LI", X_CNT, imm=63)
                with p.loop(X_CNT, note="triangular solve"):
                    p.V("FMA", Vec(128, 63, 1, dst=O(LR + 512, 4, 512, 0), a=O(LA + 256, 0, 256, 0),
                                   b=O(LR, 4, 0, 0), c=O(LR + 512, 4, 512, 0), override="m"),
                        regs=(X_RSC, X_RK, X_RSC, X_RSC, X_LR), note="solve step")
                    p.S("ADD", X_RSC, X_RSC, X_G)
                    p.S("ADDI", X_RK, X_RK, imm=260)
                    p.S("ADDI", X_LR, X_LR, imm=-1)
                split_w(O(LR, 512, 4, 0), 64, 128, LDW, "Delta columns (weight)")
                split_act(O(LP, 4, 256, 0), 64, 64, 1, LPQ, "P (act)")
                p.V("MUL", Vec(128, 64, 1, dst=O(LUP, 4, 512, 0), a=O(qk + 512, 4, 1024, 0), b=O(LG3, 0, LGR, 0)),
                    regs=(0, X_POS, X_GA), note="Ktil")
                split_act(O(LUP, 512, 4, 0), 64, 128, 1, LKT, "Ktil^T (act)")
                p.fence("V", sig=(sm.DNB,))
                mm3(128, 64, LPQ, LPQ + 2 * 64 * 256, LDW, 128, LPD, 512, wait=(sm.DNB,), note="P Delta")
                mm3(128, 128, LKT, LKT + 2 * 128 * 256, LDW, 128, LUP, 512, sig=(sm.DNM,), note="Ktil^T Delta")
                p.V("MUL", Vec(128, 64, 1, dst=O(LPD, 4, 512, 0), a=O(LPD, 4, 512, 0),
                               b=O(LDW + 2 * 128 * 256, 2, 0, 0, "f16")), wait=(sm.DNM,), note="P Delta x s_Delta")
                p.V("FMA", Vec(128, 64, 1, dst=O(DO, 4, 512, 0), a=O(LKS + 64 * 512, 4, 512, 0), b=O(LG2, 0, LGR, 0),
                               c=O(LPD, 4, 512, 0)), regs=(X_PAR, 0, X_GA), note="o")
                p.V("MUL", Vec(128, 128, 1, dst=O(LUP, 4, 512, 0), a=O(LUP, 4, 512, 0),
                               b=O(LDW + 2 * 128 * 256, 2, 0, 0, "f16")), note="update x s_Delta")
                p.V("FMA", Vec(128, 128, 1, dst=O(SB, 4, 512, 0), a=O(SB, 4, 512, 0), b=O(LG2 + 63 * LGR, 0, 0, 0),
                               c=O(LUP, 4, 512, 0)), regs=(X_SBP, X_SBP, X_GA), note="S = gamma_C S + update")
                p.S("ADDI", X_POS, X_POS, imm=64 * 1024)
                p.S("ADDI", X_PAR, X_PAR, imm=64 * 512)
                p.S("ADDI", X_GO, X_GO, imm=2 * 64 * LGR)
                p.S("ADDI", X_GA, X_GA, imm=64 * LGR)
                p.S("ADDI", X_BO, X_BO, imm=64 * BDR)
            p.fence("V", sig=(sm.DN_done,))

        p.S("LI", X_T0, imm=dn_index * 3 * QKV_CH * 4)
        p.D("R", Dma(CLOG, CH, 3 * QKV_CH * 4), regs=(X_T0,), wait=(sm.CH_free,), sig=(sm.CH_ok,), note="conv history")
        p.fence("V", wait=(sm.CH_ok,))
        p.S("LI", X_K1, imm=dn_index * LV * STATE_HEAD)
        p.S("LI", X_H, imm=0)
        p.S("LI", X_HB, imm=0)
        p.S("LI", X_KK, imm=0)
        p.S("LI", X_SBP, imm=0)
        p.S("LI", X_CNT, imm=LQK)
        with p.loop(X_CNT, note="key heads"):
            mm(HID, 256, XN, XNE, QKB, 256 * 4, wait=(sm.QK_free,), sig=(sm.QK_ok,), note="q, k of one key head")
            p.S("SLLI", X_TAP, X_KK, imm=2)
            p.S("ADD", X_TAP, X_TAP, X_PRMP)
            p.V("CONV4", Vec(128, 2, C, dst=O(CQK, 4, 512, 1024), a=O(QKB, 4, 512, 1024),
                             b=O(PRM + 2 * HID * 4, 16, KOFF * 16, 0), c=O(CH, 4, KOFF * 4, QKV_CH * 4)),
                regs=(0, 0, X_TAP, X_KK), wait=(sm.QK_ok,), sig=(sm.CV1,), note="conv q, k")
            p.D("L", Dma(QKB + (C - 3) * 1024, CWO, 512, 2, 512, KOFF * 4, 3, 1024, QKV_CH * 4), regs=(0, X_KK),
                wait=(sm.CV1,), sig=(sm.QK_free,), note="last 3 rows -> conv history out")
            p.V("SILU", Vec(256, 1, C, dst=O(CQK, 4, 0, 1024), a=O(CQK, 4, 0, 1024)))
            p.V("RSUMSQ", Vec(128, 2, C, dst=O(SS, 4, 4, 8), a=O(CQK, 4, 512, 1024)))
            p.V("RSQRT", Vec(2, 1, C, dst=O(SS, 4, 0, 8), a=O(SS, 4, 0, 8), c=c(C_1E6)))
            p.V("MUL", Vec(128, 2, C, dst=O(CQK, 4, 512, 1024), a=O(CQK, 4, 512, 1024), b=O(SS, 0, 4, 8)))
            p.V("MUL", Vec(128, 1, C, dst=O(CQK, 4, 0, 1024), a=O(CQK, 4, 0, 1024), b=c(C_RS128)))
            if dn == "lite":                                     # its value heads (a loop: instruction memory)
                p.S("LI", X_K2, imm=VPK)
                vh = p.loop(X_K2, note="value heads")
            else:
                vh = contextlib.nullcontext()
            with vh:
                for _ in range(1 if dn == "lite" else VPK):
                    mm(HID, 256, XN, XNE, VZB, 256 * 4, wait=(sm.VZ_free, sm.VZL_free), sig=(sm.VZ_ok,),
                       note="v, z of one value head")
                    p.S("SLLI", X_TAP, X_HB, imm=2)
                    p.S("ADD", X_TAP, X_TAP, X_PRMP)
                    p.V("CONV4", Vec(128, 1, C, dst=O(CV, 4, 0, 512), a=O(VZB, 4, 0, 1024),
                                     b=O(PRM + (2 * HID + VOFF * 4) * 4, 16, 0, 0), c=O(CH + VOFF * 4, 4, 0, QKV_CH * 4)),
                        regs=(0, 0, X_TAP, X_HB), wait=(sm.VZ_ok,), sig=(sm.CV2,), note="conv v")
                    p.D("L", Dma(VZB + (C - 3) * 1024, CWO + VOFF * 4, 512, 3, 1024, QKV_CH * 4), regs=(0, X_HB),
                        wait=(sm.CV2,), sig=(sm.VZL_free,), note="last 3 rows -> conv history out")
                    p.V("SILU", Vec(128, 1, C, dst=O(CV, 4, 0, 512), a=O(CV, 4, 0, 512)))
                    p.D("R", Dma(STB, SB, STATE_HEAD), regs=(X_K1, X_SBP), wait=(sm.SB_free,), sig=(sm.S_ok,),
                        note="state in")
                    if dn == "seq":
                        p.V("DN", Vec(LD, LD, 1, dst=O(DO, 4, 0, 512), a=O(CQK, 4, 0, 1024), b=O(CV, 4, 0, 512),
                                      c=O(SB, 4, 512, 0), par=(("T", C), ("commit_spec", 1))),
                            regs=(0, X_H, 0, X_SBP, 0), wait=(sm.S_ok,), sig=(sm.DN_done,),
                            note="DeltaNet head, 128 tokens")
                    elif dn == "lite":
                        lite_head(CQK)
                    else:
                        # two 64-token chunks: V prep -> M (K K^T, Q K^T, S^T k, S^T q) -> V solve -> M (P Delta,
                        # Ktil^T Delta into the state) -> V output. Each M command does 3 int8 passes (hi*hi, hi*lo,
                        # lo*hi), counted as 3x the outputs, accumulated by the 'acc' epilogue.
                        for sc in range(2):
                            p.V("DNC", Vec(64, 128, 1, dst=O(DQ, 1, 0, 256), a=O(CQK + sc * 64 * 1024, 4, 0, 1024),
                                           b=O(SB, 4, 512, 0), c=O(DSQ, 1, 0, 128), par=(("phase", 0),)),
                                regs=(0, 0, X_SBP), wait=(sm.S_ok,) if sc == 0 else (), sig=(sm.DNP,),
                                note="chunked DN: prep (quantize q, k, state copy; decay)")
                            p.M("MMS", Mm(256, 3 * 64, 128, DQ, 256, DQ + 2 * C * 256, 8, DKK, 256, wfmt="int8", seg=4,
                                          wsrc=DKW), wait=(sm.DNP,), note="K K^T, Q K^T")
                            p.M("MMS", Mm(256, 3 * 128, 128, DQ, 256, DQ + 2 * C * 256, 8, DKS, 512, wfmt="int8",
                                          seg=4, wsrc=DSQ), sig=(sm.DNA,), note="S^T k, S^T q")
                            p.V("DNC", Vec(64, 128, 1, dst=O(DDL, 4, 0, 512), a=O(DKK, 4, 0, 256), b=O(DKS, 4, 0, 512),
                                           c=O(CV + sc * 64 * 512, 4, 0, 512), par=(("phase", 1),)),
                                wait=(sm.DNA,), sig=(sm.DNB,), note="chunked DN: solve (I + A) Delta = r")
                            p.M("MMS", Mm(256, 3 * 128, 64, DM, 64, DM + 64 * 64, 2, DPD, 512, wfmt="int8", seg=4,
                                          wsrc=DDL), wait=(sm.DNB,), note="P Delta")
                            p.M("MMS", Mm(256, 3 * 128, 128, DKT, 64, DKT + 2 * 128 * 64, 2, SB, 512, wfmt="int8",
                                          seg=4, epi="acc", wsrc=DDL), regs=(0, 0, X_SBP), sig=(sm.DNM,),
                                note="Ktil^T Delta -> state")
                            p.V("DNC", Vec(64, 128, 1, dst=O(DO + sc * 64 * 512, 4, 0, 512), a=O(DKS, 4, 0, 512),
                                           b=O(DPD, 4, 0, 512), par=(("phase", 2),)),
                                wait=(sm.DNM,), sig=(sm.DN_done,) if sc == 1 else (), note="chunked DN: output")
                    p.D("X", Dma(SB, STB, STATE_HEAD), regs=(X_SBP, X_K1), wait=(sm.DN_done,), sig=(sm.SB_free,))
                    p.V("RSUMSQ", Vec(128, 1, C, dst=O(SS, 4, 0, 4), a=O(DO, 4, 0, 512)))
                    p.V("RSQRT", Vec(1, 1, C, dst=O(SS, 4, 0, 4), a=O(SS, 4, 0, 4), b=c(C_1_128), c=c(C_EPS)))
                    p.V("MUL", Vec(128, 1, C, dst=O(DO, 4, 0, 512), a=O(DO, 4, 0, 512), b=O(SS, 0, 0, 4)))
                    p.V("MUL", Vec(128, 1, C, dst=O(DO, 4, 0, 512), a=O(DO, 4, 0, 512),
                                   b=O(PRM + P_DT + 2 * LV * 4, 4, 0, 0)), regs=(0, 0, X_PRMP))
                    p.V("SILU", Vec(128, 1, C, dst=O(TMP, 4, 0, 512), a=O(VZB + 512, 4, 0, 1024)))
                    p.S("SLLI", X_T1, X_H, imm=7)
                    p.S("SLLI", X_T2, X_H, imm=2)
                    p.V("QUANT", Vec(128, 1, C, dst=O(XO, 1, 0, VD, "i8"), a=O(DO, 4, 0, 512), b=O(TMP, 4, 0, 512),
                                     c=O(XOE, 1, 0, VD // 32, "e8")), regs=(X_T1, 0, 0, X_T2), sig=(sm.VZ_free,),
                        note="gated norm -> int8 into out-projection input")
                    p.S("ADDI", X_H, X_H, imm=1)
                    p.S("ADDI", X_HB, X_HB, imm=512)
                    p.S("ADDI", X_K1, X_K1, imm=STATE_HEAD)
                    p.S("LI", X_T3, imm=STATE_HEAD)
                    p.S("SUB", X_SBP, X_T3, X_SBP)
            p.S("ADDI", X_KK, X_KK, imm=512)
        p.fence("V", sig=(sm.XO_ready, sm.CH_free), note="all heads done")
        p.fence("L", sig=(sm.CWO_ok,), note="conv history rows all copied")
        p.S("LI", X_T0, imm=dn_index * 3 * QKV_CH * 4)
        p.D("X", Dma(CWO, CLOG, 3 * QKV_CH * 4), regs=(0, X_T0), wait=(sm.CWO_ok,), sig=(sm.CWO_free,),
            note="conv history out")
        mm(VD, HID, XO, XOE, H, HID * 4, "acc", wait=(sm.XO_ready,), sig=(sm.H_ready,), note="out-projection")
        ffn(after_dn=True, last=last)
        prm_flip()

    # ------------------------------------------------------------- attention layer
    def attention(att_index, slot, last):
        base = KVB + att_index * KV_LAYER
        prm_load(slot)
        rmsnorm(0, (sm.H_ready,), (sm.XN_ready,))
        p.fence("M", wait=(sm.XN_ready,))
        qn, kn = PRM + 2 * HID * 4, PRM + (2 * HID + HD) * 4
        # blocks to cover: (L + C) / 256 rounded up; the last one holds this chunk's own positions (L is a multiple
        # of 128, so all 128 fall in one block) and is the only one that needs the causal mask
        p.S("ADDI", X_LR, X_L, imm=C + 255)
        p.S("SRLI", X_K2, X_LR, imm=8)
        # the KV heads are a hardware loop (instruction memory, note 31 part 3): X_RK = g x KH and X_RSC = g x 2 lmax x 2
        # (this head's K / V^T rows and scales in DRAM), X_GX / X_GXE its columns of the o-projection input
        p.S("LI", X_RK, imm=0)
        p.S("LI", X_RSC, imm=0)
        p.S("LI", X_GX, imm=0)
        p.S("LI", X_GXE, imm=0)
        p.S("LI", X_K1, imm=KVH)
        with p.loop(X_K1, note="KV heads"):
            mm(HID, 1024, XN, XNE, QG, 1024 * 4, wait=(sm.QG_free,), note="q, KV head")
            mm(HID, HD, XN, XNE, KF, HD * 4, note="k")
            mm(HID, HD, XN, XNE, VF, HD * 4, sig=(sm.QG_ok,), note="v")
            qa = O(QG, 4, HD * 4, 1024 * 4)
            p.V("RSUMSQ", Vec(HD, 4, C, dst=O(SS, 4, 4, 64), a=qa), wait=(sm.QG_ok,))
            p.V("RSQRT", Vec(4, 1, C, dst=O(SS, 4, 0, 64), a=O(SS, 4, 0, 64), b=c(C_1_256), c=c(C_EPS)))
            p.V("MUL", Vec(HD, 4, C, dst=qa, a=qa, b=O(SS, 0, 4, 64)))
            p.V("MUL", Vec(HD, 4, C, dst=qa, a=qa, b=O(qn, 4, 0, 0)), regs=(0, 0, X_PRMP))
            p.V("ROPE", Vec(HD, 4, C, dst=qa, a=qa, b=O(RT + C * 768, 4, 0, 256), par=(("rot", ROT),)))
            for r in range(4):
                p.V("QUANT", Vec(HD, 1, C, dst=O(Q4 + r * C * HD, 1, 0, HD, "i8"), a=O(QG + r * HD * 4, 4, 0, 1024 * 4),
                                 c=O(Q4E + r * C * 8, 1, 0, 8, "e8"), par=(("exp_bias", -4),)))
            p.fence("V", sig=(sm.QG_free,), note="q rounded: QG free for the gate")
            for (src, dstq) in ((KF, KQ), (VF, VQ)):
                a_ = O(src, 4, 0, HD * 4)
                if src == KF:
                    p.V("RSUMSQ", Vec(HD, 1, C, dst=O(SS, 4, 0, 4), a=a_))
                    p.V("RSQRT", Vec(1, 1, C, dst=O(SS, 4, 0, 4), a=O(SS, 4, 0, 4), b=c(C_1_256), c=c(C_EPS)))
                    p.V("MUL", Vec(HD, 1, C, dst=a_, a=a_, b=O(SS, 0, 0, 4)))
                    p.V("MUL", Vec(HD, 1, C, dst=a_, a=a_, b=O(kn, 4, 0, 0)), regs=(0, 0, X_PRMP))
                    p.V("ROPE", Vec(HD, 1, C, dst=a_, a=a_, b=O(RT + C * 768, 4, 0, 256), par=(("rot", ROT),)))
                p.V("QKV", Vec(HD, 1, C, dst=O(dstq, 1, 0, HD, "i8"), a=a_, c=O(dstq + C * HD, 0, 0, 2, "f16")),
                    sig=(sm.KVQ, sm.KVQ_X) if src == VF else ())
            # write the new K rows, V^T half-block rows and scales; then attention may read them
            p.D("L", Dma(VQ, VTT, 1, HD, 1, C, C, HD, 1), wait=(sm.KVQ,), sig=(sm.VTT_ok,), note="V -> V^T (transpose)")
            p.S("SLLI", X_T0, X_L, imm=8)
            p.S("ADD", X_T1, X_RK, 0)
            p.S("ADD", X_T0, X_T0, X_T1)                         # K rows at position L of head g
            p.D("X", Dma(KQ, base, C * HD), regs=(0, X_T0), wait=(sm.KVQ_X,), note="new K rows")
            p.S("SRLI", X_T3, X_L, imm=8)
            p.S("SLLI", X_T3, X_T3, imm=16)                      # block L / 256 (64 KB each)
            p.S("ANDI", X_T0, X_L, imm=255)
            p.S("ADD", X_T3, X_T3, X_T0)                         # + column L % 256 inside the block
            p.S("ADD", X_T3, X_T3, X_T1)
            p.D("X", Dma(VTT, base + KVH * KH, C, HD, C, 256), regs=(0, X_T3), wait=(sm.VTT_ok,), note="new V^T rows")
            p.S("SLLI", X_T2, X_L, imm=1)
            p.S("ADD", X_T2, X_T2, X_RSC)
            p.D("X", Dma(KQ + C * HD, base + 2 * KVH * KH, C * 2, 2, VQ - KQ, lmax * 2),
                regs=(0, X_T2), sig=(sm.KVW,), note="new K, V scales")
            p.fence("R", wait=(sm.KVW,), note="new K/V in DRAM before reading blocks")
            for r in range(4):
                Mo = ML + r * 6 * C * 4
                p.V("COPY", Vec(C, dst=O(Mo), a=c(C_NINF)))
                p.V("COPY", Vec(C, dst=O(Mo + 5 * C * 4), a=c(C_ZERO)))
                p.V("COPY", Vec(HD, 1, C, dst=O(OA + r * C * HD * 4, 4, 0, HD * 4), a=c(C_ZERO)))
            p.S("LI", X_POS, imm=0)
            p.S("LI", X_PAR, imm=0)

            def block(masked):
                p.S("SLLI", X_T0, X_POS, imm=8)
                p.S("ADD", X_T0, X_T0, X_RK)
                p.D("R", Dma(base, KB, 256 * HD, 2, KVH * KH, 256 * HD), regs=(X_T0, X_PAR), wait=(sm.KB_free,),
                    note="K and V^T block -> SRAM")
                p.S("SLLI", X_T1, X_POS, imm=1)
                p.S("ADD", X_T1, X_T1, X_RSC)
                p.D("R", Dma(base + 2 * KVH * KH, KB + 2 * 256 * HD, 512, 2, lmax * 2, 512), regs=(X_T1, X_PAR),
                    sig=(sm.KB_ok,), note="block scales")
                order = [("qk", 0), ("qk", 1), ("pv", 0), ("qk", 2), ("pv", 1), ("qk", 3), ("pv", 2), ("pv", 3)]
                for kind, r in order:
                    sl = (r % 2) * slab
                    Mo = ML + r * 6 * C * 4
                    Mn, Mt, Lt, Al, Ls = (Mo + i * C * 4 for i in range(1, 6))
                    if kind == "qk":
                        p.M("MMS", Mm(HD, 256, C, Q4 + r * C * HD, HD, Q4E + r * C * 8, 8, S_T + sl, 1024,
                                      wfmt="int8", seg=4, wsrc=KB), regs=(0, 0, 0, X_PAR),
                            wait=(sm.KB_ok, sm.S_free) if r == 0 else (sm.S_free,), sig=(sm.K_done,), note="q . K block")
                        S_ = O(S_T + sl, 4, 1024)
                        box = dict(n=256, m=C)
                        if fused:
                            if masked:                           # this chunk's own block: -inf above the diagonal
                                p.V("CAUSAL", Vec(256, 1, C, dst=O(S_T + sl, 4, 0, 1024), a=O(S_T + sl, 4, 0, 1024),
                                                  par=(("step", 1),)), regs=(0, 0, X_POS, X_L), wait=(sm.K_done,),
                                    note="causal mask (last block)")
                            p.V("SMTILE", Vec(**box, dst=O(P_T + sl, 1, 256, 0, "i8"), a=S_, b=O(KB + 2 * 256 * HD, 2, 0, 0,
                                                                                                "f16"),
                                              c=O(Mo, 0, 4, 0), par=(("vs", 512), ("e", (PE_T - P_T, 8)), ("mlb", C * 4))),
                                regs=(0, 0, X_PAR, 0), wait=(sm.P_free,) if masked else (sm.K_done, sm.P_free),
                                sig=(sm.P_ok, sm.S_free), note="fused softmax step")
                        else:
                            p.V("MUL", Vec(**box, dst=S_, a=S_, b=O(KB + 2 * 256 * HD, 2, 0, 0, "f16")), regs=(0, 0, X_PAR),
                                wait=(sm.K_done,))
                            p.V("CAUSAL", Vec(256, 1, C, dst=O(S_T + sl, 4, 0, 1024), a=O(S_T + sl, 4, 0, 1024),
                                              par=(("step", 1),)), regs=(0, 0, X_POS, X_L))
                            p.V("RMAX", Vec(**box, dst=O(Mt, 0, 4), a=S_))
                            p.V("MAX", Vec(C, dst=O(Mn), a=O(Mt), b=O(Mo)))
                            p.V("SUB", Vec(**box, dst=S_, a=S_, b=O(Mn, 0, 4)))
                            p.V("EXP", Vec(**box, dst=S_, a=S_))
                            p.V("RSUM", Vec(**box, dst=O(Lt, 0, 4), a=S_))
                            p.V("SUB", Vec(C, dst=O(Al), a=O(Mo), b=O(Mn)))
                            p.V("EXP", Vec(C, dst=O(Al), a=O(Al)))
                            p.V("FMA", Vec(C, dst=O(Ls), a=O(Ls), b=O(Al), c=O(Lt)))
                            p.V("COPY", Vec(C, dst=O(Mo), a=O(Mn)))
                            p.V("QUANT", Vec(**box, dst=O(P_T + sl, 1, 256, 0, "i8"), a=S_,
                                             b=O(KB + 2 * 256 * HD + 512, 2, 0, 0, "f16"), c=O(PE_T + sl, 1, 8, 0, "e8")),
                                regs=(0, 0, X_PAR), wait=(sm.P_free,), sig=(sm.P_ok, sm.S_free))
                    else:
                        p.M("MMS", Mm(256, HD, C, P_T + sl, 256, PE_T + sl, 8, OT_T + sl, HD * 4, wfmt="int8", seg=4,
                                      wsrc=KB + 256 * HD), regs=(0, 0, 0, X_PAR), wait=(sm.P_ok, sm.Ot_free),
                            sig=(sm.Ot_ok, sm.P_free), note="P . V block")
                        p.V("FMA", Vec(HD, C, dst=O(OA + r * C * HD * 4, 4, HD * 4), a=O(OA + r * C * HD * 4, 4, HD * 4),
                                       b=O(Al, 0, 4), c=O(OT_T + sl, 4, HD * 4)), wait=(sm.Ot_ok,), sig=(sm.Ot_free,))
                p.fence("M", sig=(sm.KB_free,), note="block used by all 4 heads")
                p.S("ADDI", X_POS, X_POS, imm=256)
                p.S("LI", X_T3, imm=BLK)
                p.S("SUB", X_PAR, X_T3, X_PAR)
            p.S("ADDI", X_CNT, X_K2, imm=-1)
            with p.loop(X_CNT, note="256-position blocks before this chunk's own"):
                block(False)
            block(True)
            mm(HID, 1024, XN, XNE, QG, 1024 * 4, wait=(sm.QG_free,), sig=(sm.GATE_ok,), note="gate (after attention)")
            p.V("SIGMOID", Vec(1024, 1, C, dst=O(QG, 4, 0, 4096), a=O(QG, 4, 0, 4096)), wait=(sm.GATE_ok,))
            for r in range(4):
                Mo = ML + r * 6 * C * 4
                p.V("RECIP", Vec(C, dst=O(Mo + 3 * C * 4), a=O(Mo + 5 * C * 4)))
                p.V("MUL", Vec(HD, 1, C, dst=O(TMP, 4, 0, HD * 4), a=O(OA + r * C * HD * 4, 4, 0, HD * 4),
                               b=O(Mo + 3 * C * 4, 0, 0, 4)))
                p.V("QUANT", Vec(HD, 1, C, dst=O(XO + r * HD, 1, 0, QD, "i8"), a=O(TMP, 4, 0, HD * 4),
                                 b=O(QG + r * HD * 4, 4, 0, 4096), c=O(XOE + r * 8, 1, 0, QD // 32, "e8")),
                    regs=(X_GX, 0, 0, X_GXE), sig=(sm.QG_free,) if r == 3 else (), note="gated attention out -> int8")
            p.S("ADDI", X_RK, X_RK, imm=KH)
            p.S("ADDI", X_RSC, X_RSC, imm=2 * lmax * 2)
            p.S("ADDI", X_GX, X_GX, imm=4 * HD)
            p.S("ADDI", X_GXE, X_GXE, imm=4 * HD // 32)
        p.fence("V", sig=(sm.XO_ready,), note="all KV heads' outputs rounded")
        mm(QD, HID, XO, XOE, H, HID * 4, "acc", wait=(sm.XO_ready,), sig=(sm.H_ready,), note="o-projection")
        ffn(last=last)
        prm_flip()

    # ------------------------------------------------------------- program
    for s_, n in (("PRM_free", 2), ("CH_free", 1), ("SB_free", 2), ("GU_free", 2), ("S_free", 2), ("P_free", 2),
                  ("Ot_free", 2), ("QK_free", 1), ("VZ_free", 1), ("VZL_free", 1), ("QG_free", 1), ("KB_free", 2),
                  ("H_free", 1), ("EMB_free", 1)):
        for _ in range(n):
            p.S("SIGS", imm=getattr(sm, s_))
    p.S("LI", X_PRMP, imm=0)
    p.S("LI", X_NC, imm=chunks)
    p.S("LI", X_TOK, imm=TOKB)
    # the prompt-chunk loop is a counted branch, not a hardware loop: the lite DeltaNet layer already uses all 4
    # hardware-loop levels (D-025). Cost: one ADDI and one BNE bundle per 128-token chunk.
    p.label("chunk")
    p.S("LI", X_WP, imm=0)
    # 128 embedding rows: token ids from the host buffer (one DMA each), once the previous chunk is done with them
    # (the last row's own DMA signals: a fence in a DMA queue passes once the reads ahead of it are requested,
    # before their bytes land)
    p.S("LI", X_CNT, imm=C - 1)
    p.S("ADD", X_T2, X_TOK, 0)
    p.S("LI", X_T0, imm=0)
    p.fence("R", wait=(sm.EMB_free,), note="previous chunk's last FFN is done with the shared region")

    def emb_row(sig=()):
        p.S("LW", X_T1, X_T2, imm=0)
        p.S("LI", X_T3, imm=EMB_ROW)
        p.S("MUL", X_T1, X_T1, X_T3)
        p.D("R", Dma(EMBT, EMB, EMB_ROW), regs=(X_T1, X_T0), sig=sig, note="embedding row")
        p.S("ADDI", X_T2, X_T2, imm=4)
        p.S("ADDI", X_T0, X_T0, imm=EMB_ROW)
    with p.loop(X_CNT, note="embedding rows"):
        emb_row()
    emb_row(sig=(sm.EMB_ok,))
    p.V("DEQ", Vec(HID, 1, C, dst=O(H, 4, 0, HID * 4), a=O(EMB, 1, 0, EMB_ROW, "i8"),
                   b=O(EMB + HID, 2, 0, EMB_ROW, "f16")), wait=(sm.EMB_ok, sm.H_free), sig=(sm.H_ready,))
    # RoPE: the chunk's 128 positions share one coarse row (L >> 10) and take 128 consecutive fine rows (L & 1023)
    p.S("SRLI", X_T1, X_L, imm=10)
    p.S("LI", X_T3, imm=384)
    p.S("MUL", X_T1, X_T1, X_T3)
    p.S("ANDI", X_T2, X_L, imm=1023)
    p.S("MUL", X_T2, X_T2, X_T3)
    p.D("R", Dma(ROPE, RT, 384, C, 0, 768), regs=(X_T1,), note="rope coarse angle (one row, broadcast)")
    p.D("R", Dma(ROPE + 1024 * 384, RT + 384, 384, C, 384, 768), regs=(X_T2,), sig=(sm.RT_ok,), note="rope fine angles")
    # cos(A+B) = cA cB - sA sB ; sin(A+B) = sA cB + cA sB   (R6; tables hold cos, sin, -sin)
    CS = RT + C * 768

    def f(o, k_):
        return O(o + k_ * 128, 4, 0, 768)
    p.V("MUL", Vec(32, 1, C, dst=O(TMP, 4, 0, 128), a=f(RT, 1), b=f(RT + 384, 2)), wait=(sm.RT_ok,))
    p.V("FMA", Vec(32, 1, C, dst=O(CS, 4, 0, 256), a=f(RT, 0), b=f(RT + 384, 0), c=O(TMP, 4, 0, 128)))
    p.V("MUL", Vec(32, 1, C, dst=O(TMP, 4, 0, 128), a=f(RT, 1), b=f(RT + 384, 0)))
    p.V("FMA", Vec(32, 1, C, dst=O(CS + 128, 4, 0, 256), a=f(RT, 0), b=f(RT + 384, 1), c=O(TMP, 4, 0, 128)),
        note="cos/sin by angle addition")
    li = 0
    for l in range(NL):
        if (l + 1) % 4:
            deltanet(li, l, last=(l == NL - 1))
            li += 1
        else:
            attention(l // 4, l, last=(l == NL - 1))
    p.S("ADDI", X_L, X_L, imm=C)
    p.S("ADDI", X_TOK, X_TOK, imm=C * 4)
    p.S("ADDI", X_NC, X_NC, imm=-1)
    p.branch("BNE", X_NC, 0, "chunk", note="next prompt chunk")
    p.fence("V", wait=(sm.H_free,))
    p.S("HALT")
    info = dict(sram=sa, dram=da, sems=sm, cfg=cfg, chunks=chunks, lmax=lmax, dn=dn, main_stream=main_stream,
                KV_LAYER=KV_LAYER, PRMB=PRMB, EMB_ROW=EMB_ROW, QKV_CH=QKV_CH, VD=VD, QD=QD, BAW=BAW, NDN=NDN, NAT=NAT,
                FFC=FFC, C=C,
                bufs=dict(H=H, CONST=CONST, TOKB=TOKB, XN=XN, XO=XO),
                dbase=dict(W_MAIN=W_MAIN, EMBT=EMBT, KVB=KVB, STB=STB, CLOG=CLOG, PRMD=PRMD, ROPE=ROPE))
    return p, info
