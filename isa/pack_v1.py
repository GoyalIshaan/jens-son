"""Weight packer for ISA v1: a Hugging Face checkpoint -> the DRAM and SRAM images the program expects.

It does, once, every offline layout change qwen_v1.py's docstring lists, and writes each matrix in the order
the matrix unit reads it (the "conventions" section of design/spec/isa_v1_reference.md):

  lut4 matrix in the weight stream = a 32-byte header, then the data
    header: 16 int8 table entries (k/127), fp32 tscale = gmax/127, 12 zero bytes
    data:   for each tile of 64 outputs, for each 1,024-input chunk, for each output of the tile:
            512 bytes of 4-bit codes (input 2i in the low nibble, 2i+1 in the high one), then 64 E4M3 scales
  Outputs are zero-padded to whole 64-output tiles and inputs to whole 1,024-input beats (E4M3 scale 0).

It also returns the golden model's weight dict built from the *same* quantized tensors, so the chip and the
golden model compute with identical weights. The chip quantizes per stored matrix: the DeltaNet in-projection
(qkv | z | b | a) is one matrix with one table and one gmax, so the golden model gets row slices of it.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
import bitexact_vec as BV  # noqa: E402
import formats  # noqa: E402
from formats import QTensor  # noqa: E402

import isa_v1 as I  # noqa: E402
import qwen_v1 as Q  # noqa: E402

FMT = "lut4_g16_fp8_t8"


def qslice(qt, r0, r1):
    out = QTensor(qt.codes[r0:r1], qt.scales[r0:r1], qt.table, qt.group, (r1 - r0, qt.shape[1]))
    out.gmax = qt.gmax
    return out


def qperm(qt, idx):
    out = QTensor(qt.codes[idx], qt.scales[idx], qt.table, qt.group, qt.shape)
    out.gmax = qt.gmax
    return out


def qrows(qt, idx):
    """A QTensor of the rows idx (any order) of qt, same table and gmax."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    out = QTensor(qt.codes[idx], qt.scales[idx], qt.table, qt.group, (len(idx), qt.shape[1]))
    out.gmax = qt.gmax
    return out


def lut4_bytes(qt, n_out_p=None, n_in_p=None, r0=0, r1=None):
    """Serialize rows r0:r1 of a lut4_g16_fp8_t8 QTensor (header + data), padded."""
    r1 = qt.shape[0] if r1 is None else r1
    N, K = r1 - r0, qt.shape[1]
    n_out_p = n_out_p or -(-N // 64) * 64
    n_in_p = n_in_p or -(-K // 1024) * 1024
    gmax = float(qt.gmax)
    tab = torch.round(qt.table * 127).to(torch.int8).numpy()
    hdr = np.zeros(32, np.uint8)
    hdr[:16] = tab.view(np.uint8)
    hdr[16:20] = np.frombuffer(np.float32(gmax / 127.0).tobytes(), np.uint8)
    codes = np.zeros((n_out_p, n_in_p), np.uint8)
    codes[:N, :K] = qt.codes[r0:r1].reshape(N, K).numpy().astype(np.uint8)
    s8 = np.zeros((n_out_p, n_in_p // 16), np.uint8)
    s8[:N, :K // 16] = (qt.scales[r0:r1] / qt.gmax).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    tiles, chunks = n_out_p // 64, n_in_p // 1024
    c = codes.reshape(tiles, 64, chunks, 512, 2)
    packed = (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8).transpose(0, 2, 1, 3)       # [tile, chunk, o, 512]
    sc = s8.reshape(tiles, 64, chunks, 64).transpose(0, 2, 1, 3)                       # [tile, chunk, o, 64]
    data = np.concatenate([packed, sc], -1).reshape(-1)
    assert data.size == n_out_p * n_in_p * 9 // 16
    return np.concatenate([hdr, data])


def quantize_all(raw, cfg, cache_path=None):
    """Chip-format quantization of every matrix, per stored matrix. Returns {chip matrix name: QTensor}."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    torch.manual_seed(0)
    out = {}
    for l in range(cfg.layers):
        p = f"layers.{l}."
        if (l + 1) % 4:
            a = p + "linear_attn."
            merged = torch.cat([raw[a + "in_proj_qkv.weight"], raw[a + "in_proj_z.weight"],
                                raw[a + "in_proj_b.weight"], raw[a + "in_proj_a.weight"]]).float()
            out[a + "in_proj"] = formats.quantize_weight(merged, FMT)
            out[a + "out_proj.weight"] = formats.quantize_weight(raw[a + "out_proj.weight"].float(), FMT)
        else:
            a = p + "self_attn."
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                out[a + nm + ".weight"] = formats.quantize_weight(raw[a + nm + ".weight"].float(), FMT)
        for nm in ("gate_proj", "up_proj", "down_proj"):
            out[p + "mlp." + nm + ".weight"] = formats.quantize_weight(raw[p + "mlp." + nm + ".weight"].float(), FMT)
        print(f"  quantized layer {l}", flush=True)
    emb = raw["embed_tokens.weight"].float()
    out["lm_head.weight"] = formats.quantize_weight(raw.get("lm_head.weight", raw["embed_tokens.weight"]).float(), FMT)
    out["embed_tokens.weight"] = formats.quantize_weight(emb, "int8_g32")
    if cache_path:
        torch.save(out, cache_path)
    return out


def golden_weights(raw, qw, cfg):
    """The golden model's W: the chip's quantized matrices under the golden model's names."""
    import qwen35_golden as G
    W = {k: v for k, v in raw.items() if not k.endswith(G.MATMUL_SUFFIXES)}
    QKV_CH = 2 * cfg.lqk * cfg.ld + cfg.lv * cfg.ld
    VD = cfg.lv * cfg.ld
    for l in range(cfg.layers):
        p = f"layers.{l}."
        if (l + 1) % 4:
            a = p + "linear_attn."
            m = qw[a + "in_proj"]
            W[a + "in_proj_qkv.weight"] = qslice(m, 0, QKV_CH)
            W[a + "in_proj_z.weight"] = qslice(m, QKV_CH, QKV_CH + VD)
            W[a + "in_proj_b.weight"] = qslice(m, QKV_CH + VD, QKV_CH + VD + cfg.lv)
            W[a + "in_proj_a.weight"] = qslice(m, QKV_CH + VD + cfg.lv, QKV_CH + VD + 2 * cfg.lv)
            W[a + "out_proj.weight"] = qw[a + "out_proj.weight"]
        else:
            for nm in ("q_proj", "k_proj", "v_proj", "o_proj"):
                W[p + "self_attn." + nm + ".weight"] = qw[p + "self_attn." + nm + ".weight"]
        for nm in ("gate_proj", "up_proj", "down_proj"):
            W[p + "mlp." + nm + ".weight"] = qw[p + "mlp." + nm + ".weight"]
    W["lm_head.weight"] = qw["lm_head.weight"]
    W["embed_tokens.weight"] = qw["embed_tokens.weight"]
    return W


def build_images(raw, qw, info):
    """DRAM and SRAM images for a program built by qwen_v1.build(cfg=...)."""
    cfg = info["cfg"]
    db, bufs = info["dbase"], info["bufs"]
    HID, LD, LV, HD = cfg.hid, cfg.ld, cfg.lv, cfg.hd
    QKV_CH, MOW, QPW, PRMB, EMB_ROW = info["QKV_CH"], info["MOW"], info["QPW"], info["PRMB"], info["EMB_ROW"]
    FFN, FFC = cfg.ffn_p, Q.FF_CHUNK
    dram = np.zeros(info["dram"].total(), np.uint8)
    sram = np.zeros(I.SRAM_BYTES, np.uint8)

    # ---- the main weight stream, in program order
    parts = []

    def ffn(p):
        g, u, dn = (qw[p + "mlp." + n + ".weight"] for n in ("gate_proj", "up_proj", "down_proj"))
        for c in range(FFN // FFC):
            r0, r1 = c * FFC, min((c + 1) * FFC, cfg.ffn)
            for m in (g, u):
                if r0 < cfg.ffn:
                    parts.append(lut4_bytes(m, FFC, None, r0, r1))
                else:                                                # a chunk entirely in the padding
                    parts.append(lut4_bytes(qslice(m, 0, 0), FFC, HID))
        parts.append(lut4_bytes(dn, None, FFN))
    for l in range(cfg.layers):
        p = f"layers.{l}."
        if (l + 1) % 4:
            a = p + "linear_attn."
            parts.append(lut4_bytes(qw[a + "in_proj"], MOW))
            parts.append(lut4_bytes(qw[a + "out_proj.weight"]))
        else:
            a = p + "self_attn."
            qp = qw[a + "q_proj.weight"]                      # rows per head: [query 256 | gate 256] -> all queries, all gates
            idx = torch.cat([torch.arange(h * 2 * HD, h * 2 * HD + HD) for h in range(cfg.qh)] +
                            [torch.arange(h * 2 * HD + HD, (h + 1) * 2 * HD) for h in range(cfg.qh)])
            parts.append(lut4_bytes(qperm(qp, idx)))
            for nm in ("k_proj", "v_proj", "o_proj"):
                parts.append(lut4_bytes(qw[a + nm + ".weight"]))
        ffn(p)
    parts.append(lut4_bytes(qw["lm_head.weight"]))
    stream = np.concatenate(parts)
    assert stream.size == info["main_stream"], (stream.size, info["main_stream"])
    dram[db["W_MAIN"]:db["W_MAIN"] + stream.size] = stream

    common_images(raw, qw, cfg, dram, sram, db, bufs, PRMB, EMB_ROW, QKV_CH)
    return dram, sram


def prefill_stream(qw, cfg, info):
    """The weight stream of qwen_prefill_v1.build_prefill, in program order: per DeltaNet layer b|a, then per key
    head its q|k rows and per value head its v|z rows, the out-projection; per attention layer, per KV head the
    queries of its 4 query heads, k, v, then their gates; the o-projection; the FFN in 256-row chunks (gate, up),
    then down. Every matrix is a row selection of the decode program's matrix (same table, same gmax)."""
    HID, LD, LV, LQK, HD = cfg.hid, cfg.ld, cfg.lv, cfg.lqk, cfg.hd
    QKV_CH, VD, BAW, FFC = info["QKV_CH"], info["VD"], info["BAW"], info["FFC"]
    FFN, VPK = cfg.ffn_p, cfg.lv // cfg.lqk
    parts = []
    ar = torch.arange
    for l in range(cfg.layers):
        p = f"layers.{l}."
        if (l + 1) % 4:
            m = qw[p + "linear_attn.in_proj"]
            parts.append(lut4_bytes(qslice(m, QKV_CH + VD, QKV_CH + VD + 2 * LV), BAW))      # b | a
            for kk in range(LQK):
                parts.append(lut4_bytes(qrows(m, torch.cat([ar(kk * LD, (kk + 1) * LD),
                                                            ar(LQK * LD + kk * LD, LQK * LD + (kk + 1) * LD)]))))
                for h in range(kk * VPK, (kk + 1) * VPK):
                    parts.append(lut4_bytes(qrows(m, torch.cat([ar(2 * LQK * LD + h * LD, 2 * LQK * LD + (h + 1) * LD),
                                                                ar(QKV_CH + h * LD, QKV_CH + (h + 1) * LD)]))))
            parts.append(lut4_bytes(qw[p + "linear_attn.out_proj.weight"]))
        else:
            a = p + "self_attn."
            qp = qw[a + "q_proj.weight"]                      # rows per head: [query 256 | gate 256]
            for g in range(cfg.kvh):
                hs = range(4 * g, 4 * g + 4)
                parts.append(lut4_bytes(qrows(qp, torch.cat([ar(h * 2 * HD, h * 2 * HD + HD) for h in hs]))))
                parts.append(lut4_bytes(qslice(qw[a + "k_proj.weight"], g * HD, (g + 1) * HD)))
                parts.append(lut4_bytes(qslice(qw[a + "v_proj.weight"], g * HD, (g + 1) * HD)))
                parts.append(lut4_bytes(qrows(qp, torch.cat([ar(h * 2 * HD + HD, (h + 1) * 2 * HD) for h in hs]))))
            parts.append(lut4_bytes(qw[a + "o_proj.weight"]))
        g_, u_, dn = (qw[p + "mlp." + n + ".weight"] for n in ("gate_proj", "up_proj", "down_proj"))
        for c in range(FFN // FFC):
            r0, r1 = c * FFC, min((c + 1) * FFC, cfg.ffn)
            for m in (g_, u_):
                parts.append(lut4_bytes(m, FFC, None, r0, r1) if r0 < cfg.ffn else lut4_bytes(qslice(m, 0, 0), FFC, HID))
        parts.append(lut4_bytes(dn, None, FFN))
    stream = np.concatenate(parts)
    assert stream.size == info["main_stream"], (stream.size, info["main_stream"])
    return stream


def build_images_prefill(raw, qw, info, tokens):
    """DRAM and SRAM images for a program built by qwen_prefill_v1.build_prefill(cfg=...), with the prompt's token
    ids left in SRAM at TOKB (the host's job). KV cache, DeltaNet state and conv history start at zero."""
    cfg = info["cfg"]
    db, bufs = info["dbase"], info["bufs"]
    assert len(tokens) == info["chunks"] * info["C"]
    dram = np.zeros(info["dram"].total(), np.uint8)
    sram = np.zeros(I.SRAM_BYTES, np.uint8)
    stream = prefill_stream(qw, cfg, info)
    dram[db["W_MAIN"]:db["W_MAIN"] + stream.size] = stream
    common_images(raw, qw, cfg, dram, sram, db, bufs, info["PRMB"], info["EMB_ROW"], info["QKV_CH"])
    sram[bufs["TOKB"]:bufs["TOKB"] + 4 * len(tokens)] = np.asarray(tokens, np.uint32).view(np.uint8)
    return dram, sram


def common_images(raw, qw, cfg, dram, sram, db, bufs, PRMB, EMB_ROW, QKV_CH):
    """What decode and prefill images share: embedding table, per-layer parameters, RoPE tables, constants."""
    HID, LV, HD = cfg.hid, cfg.lv, cfg.hd
    LD = cfg.ld
    # ---- embedding: int8 row + fp16 scale per 32
    e = qw["embed_tokens.weight"]
    q8 = e.table[e.codes.long()].reshape(cfg.vocab, HID).numpy().astype(np.int8)
    s16 = e.scales.reshape(cfg.vocab, HID // 32).numpy().astype(np.float16)
    rows = np.concatenate([q8.view(np.uint8), s16.view(np.uint8)], 1)
    assert rows.shape[1] == EMB_ROW
    dram[db["EMBT"]:db["EMBT"] + rows.size] = rows.reshape(-1)

    # ---- per-layer parameters (fp32): norms as (1 + w), conv taps, dt_bias, -exp(A_log), gated-norm weight
    def put_f32(base, arr):
        a = np.ascontiguousarray(arr, np.float32).reshape(-1)
        dram[base:base + a.nbytes] = a.view(np.uint8)
    for l in range(cfg.layers):
        p, base = f"layers.{l}.", db["PRMD"] + l * PRMB
        put_f32(base, 1.0 + raw[p + "input_layernorm.weight"].float())
        put_f32(base + HID * 4, 1.0 + raw[p + "post_attention_layernorm.weight"].float())
        if (l + 1) % 4:
            a = p + "linear_attn."
            put_f32(base + 2 * HID * 4, raw[a + "conv1d.weight"].float().squeeze(1))       # [channels, 4]
            o = base + (2 * HID + 4 * QKV_CH) * 4
            put_f32(o, raw[a + "dt_bias"].float())
            put_f32(o + LV * 4, -raw[a + "A_log"].float().exp())
            put_f32(o + 2 * LV * 4, raw[a + "norm.weight"].float())
        else:
            a = p + "self_attn."
            put_f32(base + 2 * HID * 4, 1.0 + raw[a + "q_norm.weight"].float())
            put_f32(base + (2 * HID + HD) * 4, 1.0 + raw[a + "k_norm.weight"].float())
    put_f32(db["PRMD"] + cfg.layers * PRMB, 1.0 + raw["norm.weight"].float())

    # ---- RoPE tables: 2 x 1,024 rows of [cos 32 | sin 32 | -sin 32] (R6)
    cH, sH, cL, sL = BV.rope_tables(theta=1e7, rot=cfg.rot)
    tab = np.concatenate([np.concatenate([c.numpy(), s.numpy(), -s.numpy()], 1) for c, s in ((cH, sH), (cL, sL))])
    put_f32(db["ROPE"], tab)

    # ---- SRAM constants: 1/HID, eps, 1e-6, 1/128, 1/sqrt(128), -inf, 0, 1/256, -126 (an all-zero group's exponent)
    consts = np.array([1.0 / HID, cfg.eps, 1e-6, 1.0 / LD, 1.0 / np.sqrt(LD), -np.inf, 0.0, 1.0 / HD, -126.0],
                      np.float32)
    sram[bufs["CONST"]:bufs["CONST"] + consts.nbytes] = consts.view(np.uint8)


# ---------------------------------------------------------------- speculative decoding: the MTP head (note 31 part 3)
MTP_P = "mtp.layers.0."


def quantize_mtp(raw_mtp, cache_path=None):
    """Chip-format quantization of the MTP head's matrices (fc and its one attention layer + FFN)."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    out = {n: formats.quantize_weight(t.float(), FMT) for n, t in raw_mtp.items() if t.dim() == 2}
    if cache_path:
        torch.save(out, cache_path)
    return out


def draft_vocab(tok, texts, n):
    """The draft LM head's vocabulary: the n most frequent token ids over `texts`, then (to fill up to n) the
    lowest unseen ids. Returned in rank order; the chip's remap table maps draft index -> token id."""
    import collections
    cnt = collections.Counter()
    for t in texts:
        cnt.update(tok.encode(t).ids)
    ranked = [t for t, _ in cnt.most_common()]
    seen = set(ranked)
    ranked += [i for i in range(n * 2) if i not in seen][:max(0, n - len(ranked))]
    return ranked[:n]


def spec_images(raw, raw_mtp, qw, qmtp, info, remap):
    """DRAM and SRAM images for qwen_v1.build("spec", cfg=...): the main images plus the MTP weight stream (fc, the
    attention layer in the main layers' layout, the FFN, then the draft LM head = the tied LM head's rows `remap`),
    the remap table and the MTP's parameter slots (NL + 1: e norm | h norm | final norm; NL + 2: its layer)."""
    cfg = info["cfg"]
    dram, sram = build_images(raw, qw, info)
    db, PRMB, HD, HID = info["dbase"], info["PRMB"], cfg.hd, cfg.hid
    FFN, FFC = cfg.ffn_p, Q.FF_CHUNK
    a = MTP_P + "self_attn."
    parts = [lut4_bytes(qmtp["mtp.fc.weight"])]
    qp = qmtp[a + "q_proj.weight"]
    idx = torch.cat([torch.arange(h * 2 * HD, h * 2 * HD + HD) for h in range(cfg.qh)] +
                    [torch.arange(h * 2 * HD + HD, (h + 1) * 2 * HD) for h in range(cfg.qh)])
    parts.append(lut4_bytes(qperm(qp, idx)))
    for nm in ("k_proj", "v_proj", "o_proj"):
        parts.append(lut4_bytes(qmtp[a + nm + ".weight"]))
    g, u, dn = (qmtp[MTP_P + "mlp." + n + ".weight"] for n in ("gate_proj", "up_proj", "down_proj"))
    for c in range(FFN // FFC):
        r0, r1 = c * FFC, min((c + 1) * FFC, cfg.ffn)
        for m in (g, u):
            parts.append(lut4_bytes(m, FFC, None, r0, r1) if r0 < cfg.ffn else lut4_bytes(qslice(m, 0, 0), FFC, HID))
    parts.append(lut4_bytes(dn, None, FFN))
    assert len(remap) == Q.DRAFT_VOCAB
    parts.append(lut4_bytes(qrows(qw["lm_head.weight"], remap)))
    stream = np.concatenate(parts)
    assert stream.size == info["mtp_stream"], (stream.size, info["mtp_stream"])
    dram[db["W_MTP"]:db["W_MTP"] + stream.size] = stream
    rm = np.asarray(remap, np.uint32)
    dram[db["REMAP"]:db["REMAP"] + rm.nbytes] = rm.view(np.uint8)

    def put_f32(base, arr):
        x = np.ascontiguousarray(arr, np.float32).reshape(-1)
        dram[base:base + x.nbytes] = x.view(np.uint8)
    b1 = db["PRMD"] + (cfg.layers + 1) * PRMB
    for i, n in enumerate(("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight")):
        put_f32(b1 + i * HID * 4, 1.0 + raw_mtp[n].float())
    b2 = db["PRMD"] + (cfg.layers + 2) * PRMB
    put_f32(b2, 1.0 + raw_mtp[MTP_P + "input_layernorm.weight"].float())
    put_f32(b2 + HID * 4, 1.0 + raw_mtp[MTP_P + "post_attention_layernorm.weight"].float())
    put_f32(b2 + 2 * HID * 4, 1.0 + raw_mtp[a + "q_norm.weight"].float())
    put_f32(b2 + (2 * HID + HD) * 4, 1.0 + raw_mtp[a + "k_norm.weight"].float())
    return dram, sram
