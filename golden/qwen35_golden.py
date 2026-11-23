"""Golden model of Qwen3.5-9B (text only), written from scratch for the chip project.

Why our own code instead of Hugging Face's:
- every operation is spelled out, in the same order as the layer map
  (model_map/), so each one can be matched to a chip instruction later
- number formats are pluggable: weights, matmul inputs, KV cache and the
  DeltaNet state can each be rounded the way the chip would round them
- it is checked against Hugging Face's implementation (golden/beam_app.py)

Arithmetic is fp32 throughout. Weights stay in their stored form (bf16, or
4-bit codes + scales from formats.py) and are expanded to fp32 right before
each matrix multiply, the same way the chip expands codes through its lookup
table.
"""
import json
import math
import os

import chip_sfu
import torch
import torch.nn.functional as F
from safetensors import safe_open

from formats import QTensor, quant_act, quant_kv, quantize_weight

PREFIX = "model.language_model."


class Config:
    def __init__(self, model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        t = cfg.get("text_config", cfg)
        self.hidden = t["hidden_size"]
        self.ffn = t["intermediate_size"]
        self.layers = t["num_hidden_layers"]
        self.layer_types = t["layer_types"]
        self.vocab = t["vocab_size"]
        self.eps = t["rms_norm_eps"]
        self.q_heads = t["num_attention_heads"]
        self.kv_heads = t["num_key_value_heads"]
        self.head_dim = t["head_dim"]
        rp = t.get("rope_parameters", {})
        self.rope_theta = rp.get("rope_theta", 1e7)
        self.rot_dim = int(self.head_dim * rp.get("partial_rotary_factor", 1.0))
        self.lin_k_heads = t["linear_num_key_heads"]
        self.lin_v_heads = t["linear_num_value_heads"]
        self.lin_k_dim = t["linear_key_head_dim"]
        self.lin_v_dim = t["linear_value_head_dim"]
        self.conv_k = t["linear_conv_kernel_dim"]


# Which weights count as which kind, for per-kind number formats.
def weight_kind(name):
    if name.startswith("lm_head"):
        return "lm_head"
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if ".mlp." in name:
        return "ffn"
    if ".self_attn." in name:
        return "attn"
    if ".linear_attn." in name:
        return "deltanet"
    return "other"


MATMUL_SUFFIXES = ("_proj.weight", "in_proj_qkv.weight", "in_proj_z.weight",
                   "in_proj_b.weight", "in_proj_a.weight", "lm_head.weight",
                   "embed_tokens.weight")


# ------------------------------------------------------------------ special functions (progression step 2)
# Precision of the chip's special-function unit, as explicit mantissa bits of each result (None = exact fp32).
# A unit with relative error about 2^-b is emulated by rounding the exact result to b mantissa bits.
SF = {"exp": None, "sig": None, "softplus": None, "rsqrt": None, "recip": None}
# Chip-exact pieces (golden v2, experiment set 8). "sfu": the real table + polynomial algorithms (chip_sfu.py);
# "rope": angles from two 1,024-row tables by angle addition; "softmax": online softmax in 1,024-position
# tiles with the V scale folded into P before P is rounded (note 28); "dnq": DeltaNet q x fp32(1/sqrt(128)).
CHIP = {"sfu": False, "rope": False, "softmax": False, "dnq": False}


def _rb(y, b):
    if b is None:
        return y
    m, e = torch.frexp(y)                                   # y = m * 2^e, 0.5 <= |m| < 1
    return torch.ldexp(torch.round(m * 2.0 ** (b + 1)) / 2.0 ** (b + 1), e)


def sf_exp(x):
    if CHIP["sfu"]:
        return chip_sfu.exp(x)
    return _rb(torch.exp(x), SF["exp"])


def sf_sigmoid(x):
    if CHIP["sfu"]:
        return chip_sfu.sigmoid(x)
    return _rb(torch.sigmoid(x), SF["sig"])


def sf_silu(x):
    if CHIP["sfu"]:
        return chip_sfu.silu(x)
    return F.silu(x) if SF["sig"] is None else x * sf_sigmoid(x)


def sf_softplus(x):
    if CHIP["sfu"]:
        return chip_sfu.softplus(x)
    return _rb(F.softplus(x), SF["softplus"])


def sf_rsqrt(x):
    if CHIP["sfu"]:
        return chip_sfu.rsqrt(x)
    return _rb(torch.rsqrt(x), SF["rsqrt"])


def sf_recip(x):
    if CHIP["sfu"]:
        return chip_sfu.recip(x)
    return _rb(1.0 / x, SF["recip"])


def sf_softmax(sc):
    if CHIP["sfu"]:
        e = sf_exp(sc - sc.amax(-1, keepdim=True))
        return e * sf_recip(e.sum(-1, keepdim=True))
    if SF["exp"] is None and SF["recip"] is None:
        return torch.softmax(sc, -1)
    e = sf_exp(sc - sc.amax(-1, keepdim=True))
    return e * _rb(1.0 / e.sum(-1, keepdim=True), SF["recip"])


def _probe(pr, i, kind, ts):
    """record: keep the reference tensors (int8 codes for matrix inputs, bf16 otherwise) on the CPU.
    compare: per layer, the relative error of each tensor and, for matrix inputs, the share of int8 codes that
    change (and change by more than 1)."""
    import bitexact as B
    for name, x in ts.items():
        key = (i, name)
        if name.endswith("_in"):
            q, e = B.quant_act_exact(x.float())
            v = torch.ldexp(q, e.repeat_interleave(32, -1))
            if pr["mode"] == "record":
                pr["ref"][key] = (q.to(torch.int8).cpu(), v.bfloat16().cpu())
            else:
                q0, v0 = pr["ref"][key]
                q0, v0 = q0.to(x.device).float(), v0.to(x.device).float()
                pr["out"].setdefault(name, []).append(dict(
                    layer=i, kind=kind, rel=((v - v0).norm() / v0.norm()).item(),
                    code_change=(q != q0).float().mean().item(), code_change_gt1=((q - q0).abs() > 1).float().mean().item()))
        else:
            if pr["mode"] == "record":
                pr["ref"][key] = x.bfloat16().cpu()
            else:
                x0 = pr["ref"][key].to(x.device).float()
                pr["out"].setdefault(name, []).append(dict(layer=i, kind=kind,
                                                           rel=((x.float() - x0).norm() / x0.norm()).item()))


class Numerics:
    """How the chip rounds things. Defaults = full precision reference."""

    def __init__(self, weights=None, act=None, kv=None, state="fp32", attn_q=None, attn_p=None):
        # weights: dict kind -> format name, e.g. {"default": "int4_g32", "ffn": "int3_g32"}
        self.weights = weights or {}
        self.act = act            # None or "int8_g32"
        self.kv = kv              # None, "int8_g64" or "int4_g32"
        self.state = state        # "fp32", "bf16", "fp32_onepass", or "chunked_fp32/int8/split" (prefill, note 30)
        # attention on the int8 lanes (note 27): the query and the softmax probabilities become int8 too
        self.attn_q = attn_q      # None, or an activation format for q (groups of 32 along the head dim)
        self.attn_p = attn_p      # None, an activation format for P (groups of 32 positions), or "int8_fixed127"

    def weight_format(self, name):
        """Keys: "default", a kind ("ffn", "deltanet", "attn", "lm_head", "embed"), or a kind with
        a layer range, e.g. "ffn@0-7". Most specific wins. A value of None means keep bf16."""
        if not name.endswith(MATMUL_SUFFIXES):
            return None
        kind = weight_kind(name)
        li = int(name.split(".")[1]) if name.startswith("layers.") else None
        if li is not None:
            for key, fmt in self.weights.items():
                if key.startswith(kind + "@"):
                    a, b = map(int, key.split("@")[1].split("-"))
                    if a <= li <= b:
                        return fmt
        return self.weights.get(kind, self.weights.get("default"))

    def describe(self):
        return dict(weights=self.weights or "bf16", act=self.act or "fp32",
                    kv=self.kv or "fp32", state=self.state, attn_q=self.attn_q or "fp32",
                    attn_p=self.attn_p or "fp32")


def load_tensors(model_dir, device="cuda"):
    """Load only the text model and LM head, as stored (bf16), keyed without prefix."""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    by_file = {}
    for k, f in idx["weight_map"].items():
        if k.startswith(PREFIX) or k == "lm_head.weight":
            by_file.setdefault(f, []).append(k)
    out = {}
    for f, keys in sorted(by_file.items()):
        with safe_open(os.path.join(model_dir, f), framework="pt", device=device) as sf:
            for k in keys:
                out[k[len(PREFIX):] if k.startswith(PREFIX) else k] = sf.get_tensor(k)
    return out


class Cache:
    """Per-layer KV cache (full attention) and conv + recurrent state (DeltaNet)."""

    def __init__(self, cfg, device):
        self.pos = 0
        self.k, self.v = {}, {}
        self.conv, self.S = {}, {}
        conv_ch = (2 * cfg.lin_k_heads * cfg.lin_k_dim + cfg.lin_v_heads * cfg.lin_v_dim)
        for i, t in enumerate(cfg.layer_types):
            if t == "full_attention":
                self.k[i] = torch.zeros(0, cfg.kv_heads, cfg.head_dim, device=device)
                self.v[i] = torch.zeros(0, cfg.kv_heads, cfg.head_dim, device=device)
            else:
                self.conv[i] = torch.zeros(cfg.conv_k - 1, conv_ch, device=device)
                self.S[i] = torch.zeros(cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, device=device)


class Golden:
    def __init__(self, model_dir, numerics=None, device="cuda", tensors=None):
        """The bf16 originals stay in host RAM (`raw`); `W` holds what the GPU uses:
        bf16 copies, or 4-bit codes + scales, depending on the numerics."""
        self.cfg = Config(model_dir)
        self.dev = device
        raw = tensors if tensors is not None else load_tensors(model_dir, "cpu")
        self.set_numerics(numerics or Numerics(), raw)

    def set_numerics(self, numerics, raw=None):
        """(Re)build the weight store for a numerics setting. `raw` = bf16 tensors."""
        if raw is not None:
            self.raw = raw
        self.num = numerics
        self.W = {}
        self.weight_error = {}      # kind -> [sum sq err, sum sq weight, n weights, format]
        torch.cuda.empty_cache()
        for name, t in self.raw.items():
            fmt = numerics.weight_format(name)
            g = t.to(self.dev)
            if fmt:
                q = quantize_weight(g, fmt)
                self.W[name] = q
                e = self.weight_error.setdefault(weight_kind(name), [0.0, 0.0, 0, fmt])
                for r in range(0, g.shape[0], 32768):
                    ref = g[r:r + 32768].float()
                    e[0] += (q.dequant_slice(r, r + 32768) - ref).pow(2).sum().item()
                    e[1] += ref.pow(2).sum().item()
                e[2] += g.numel()
            else:
                self.W[name] = g
            del g

    # ---------------------------------------------------------------- basics
    def w(self, name):
        t = self.W[name]
        return t.dequant() if isinstance(t, QTensor) else t.float()

    def mm(self, x, name, rows=32768):
        """x [T, K] fp32 times weight [N, K] -> [T, N]. Matmul input rounded per numerics.
        Big matrices (the LM head) are expanded to fp32 a slice of rows at a time."""
        x = quant_act(x, self.num.act)
        if getattr(self, "collect", None) is not None and name in self.collect:
            H, n = self.collect[name]
            self.collect[name] = (H + x.t() @ x, n + x.shape[0])
        t = self.W[name]
        N = t.shape[0]
        if N <= rows:
            return x @ self.w(name).t()
        out = []
        for r in range(0, N, rows):
            ws = t.dequant_slice(r, r + rows) if isinstance(t, QTensor) else t[r:r + rows].float()
            out.append(x @ ws.t())
        return torch.cat(out, -1)

    def rms(self, x, name):
        """Qwen3.5 RMSNorm: stored weight is an offset from 1 (zero-centred)."""
        y = x * sf_rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.eps)
        return y * (1.0 + self.W[name].float())

    def embed(self, tokens):
        t = self.W["embed_tokens.weight"]
        return t.dequant_rows(tokens) if isinstance(t, QTensor) else t[tokens].float()

    # ---------------------------------------------------------------- rope
    def rope(self, x, positions):
        """Rotate the first rot_dim (64) of each head's 256 dims. x [T, H, D]."""
        c = self.cfg
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.rot_dim, 2, device=x.device).float() / c.rot_dim))
        if CHIP["rope"]:
            # chip: cos/sin of (p >> 10) * 1024 and of (p & 1023) from two fp32 tables (made in float64
            # offline), combined by angle addition in fp32
            inv64 = 1.0 / (c.rope_theta ** (torch.arange(0, c.rot_dim, 2, device=x.device).double() / c.rot_dim))
            pl = positions.long()
            A = ((pl >> 10) * 1024).double()[:, None] * inv64[None, :]
            Bq = (pl & 1023).double()[:, None] * inv64[None, :]
            cA, sA, cB, sB = A.cos().float(), A.sin().float(), Bq.cos().float(), Bq.sin().float()
            fc, fs = cA * cB - sA * sB, sA * cB + cA * sB
            cos = torch.cat([fc, fc], -1)[:, None, :]
            sin = torch.cat([fs, fs], -1)[:, None, :]
        else:
            fr = positions.float()[:, None] * inv[None, :]           # [T, rot/2]
            cos = torch.cat([fr.cos(), fr.cos()], -1)[:, None, :]    # [T, 1, rot]
            sin = torch.cat([fr.sin(), fr.sin()], -1)[:, None, :]
        r, keep = x[..., :c.rot_dim], x[..., c.rot_dim:]
        half = c.rot_dim // 2
        rot_half = torch.cat([-r[..., half:], r[..., :half]], -1)
        return torch.cat([r * cos + rot_half * sin, keep], -1)

    # ---------------------------------------------------------------- blocks
    def full_attention(self, i, h, cache, positions):
        c, p = self.cfg, f"layers.{i}.self_attn."
        T = h.shape[0]
        qg = self.mm(h, p + "q_proj.weight").view(T, c.q_heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(T, -1)   # per-head [query | gate]
        k = self.mm(h, p + "k_proj.weight").view(T, c.kv_heads, c.head_dim)
        v = self.mm(h, p + "v_proj.weight").view(T, c.kv_heads, c.head_dim)
        q = self.rope(self.rms(q, p + "q_norm.weight"), positions)
        k = self.rope(self.rms(k, p + "k_norm.weight"), positions)
        if getattr(self.num, "attn_q", None):
            q = quant_act(q, self.num.attn_q)

        cache.k[i] = torch.cat([cache.k[i], quant_kv(k, self.num.kv)])
        cache.v[i] = torch.cat([cache.v[i], quant_kv(v, self.num.kv)])
        K, V = cache.k[i], cache.v[i]                                          # [S, kvH, D]
        rep = c.q_heads // c.kv_heads
        S = K.shape[0]
        Kh, Vh = K.transpose(0, 1)[:, None], V.transpose(0, 1)[:, None]       # [kvH, 1, S, D]
        qh = q.transpose(0, 1).reshape(c.kv_heads, rep, T, c.head_dim)         # query head = kv*rep + r
        key_pos = torch.arange(S, device=h.device)
        chunk = 1024 if S <= 8192 else max(32, (1 << 27) // (S * c.q_heads))  # bound memory (256 at 32K, 64 at 128K)
        outs = []
        for t0 in range(0, T, chunk):
            sc = qh[:, :, t0:t0 + chunk] @ Kh.transpose(-1, -2) / math.sqrt(c.head_dim)   # [kvH, rep, t, S]
            mask = key_pos[None, :] > positions[t0:t0 + chunk, None]           # causal
            sc = sc.masked_fill(mask[None, None], float("-inf"))
            if CHIP["softmax"]:
                outs.append(self._tiled_attention(sc, Vh))
                continue
            P = sf_softmax(sc)
            ap = getattr(self.num, "attn_p", None)
            if ap == "int8_fixed127":                     # probabilities in [0, 1] on a fixed 1/127 grid
                P = torch.round(P * 127) / 127
            elif ap:
                pad = (-P.shape[-1]) % 32                  # groups of 32 positions; zero padding leaves amax unchanged
                P = quant_act(F.pad(P, (0, pad)), ap)[..., :P.shape[-1]] if pad else quant_act(P, ap)
            outs.append(P @ Vh)                                                # [kvH, rep, t, D]
        out = torch.cat(outs, 2).reshape(c.q_heads, T, c.head_dim)             # [H, T, D]
        out = out.transpose(0, 1).reshape(T, -1) * sf_sigmoid(gate)
        return self.mm(out, p + "o_proj.weight")

    def _tiled_attention(self, sc, Vh, tile=1024):
        """Chip order (note 28): online softmax over 1,024-position tiles. Per tile: running max m, p = exp(s - m),
        l = l * alpha + sum p, P = int8(p x V scale) with a power-of-two scale per 32 positions, O = O * alpha + P . V_int.
        V's int8 values and per-(position, head) scale are recovered exactly from the dequantized cache."""
        sv = Vh.abs().amax(-1) / 127.0                                         # [kvH, 1, S]
        vint = torch.where(sv[..., None] > 0, torch.round(Vh / sv[..., None].clamp(min=1e-30)), torch.zeros_like(Vh))
        S = sc.shape[-1]
        m = torch.full(sc.shape[:-1], float("-inf"), device=sc.device)
        l = torch.zeros(sc.shape[:-1], device=sc.device)
        O = torch.zeros(*sc.shape[:-1], Vh.shape[-1], device=sc.device)
        ap = self.num.attn_p
        for j0 in range(0, S, tile):
            s = sc[..., j0:j0 + tile]
            mn = torch.maximum(m, s.amax(-1))
            p = sf_exp(s - mn[..., None])
            al = sf_exp(m - mn)
            l = l * al + p.sum(-1)
            m = mn
            pv = p * sv[..., j0:j0 + tile][:, :, None, :]
            if ap:
                pad = (-pv.shape[-1]) % 32
                pv = quant_act(F.pad(pv, (0, pad)), ap)[..., :pv.shape[-1]] if pad else quant_act(pv, ap)
            O = O * al[..., None] + pv @ vint[..., j0:j0 + tile, :]
        return O * sf_recip(l)[..., None]

    def deltanet(self, i, h, cache):
        c, p = self.cfg, f"layers.{i}.linear_attn."
        T = h.shape[0]
        kd, vd = c.lin_k_heads * c.lin_k_dim, c.lin_v_heads * c.lin_v_dim
        qkv = self.mm(h, p + "in_proj_qkv.weight")                             # [T, 8192]
        z = self.mm(h, p + "in_proj_z.weight").view(T, c.lin_v_heads, c.lin_v_dim)
        b = self.mm(h, p + "in_proj_b.weight")                                 # [T, 32]
        a = self.mm(h, p + "in_proj_a.weight")

        # causal depthwise conv, kernel 4, over q/k/v channels, then SiLU
        w = self.W[p + "conv1d.weight"].float().squeeze(1)                     # [8192, 4]
        xp = torch.cat([cache.conv[i], qkv])                                   # [3 + T, 8192]
        cache.conv[i] = xp[-(c.conv_k - 1):].clone()
        y = sum(w[:, j] * xp[j:j + T] for j in range(c.conv_k))
        y = sf_silu(y)
        q, k, v = y[:, :kd], y[:, kd:2 * kd], y[:, 2 * kd:]
        q = q.view(T, c.lin_k_heads, c.lin_k_dim)
        k = k.view(T, c.lin_k_heads, c.lin_k_dim)
        v = v.view(T, c.lin_v_heads, c.lin_v_dim)

        beta = sf_sigmoid(b)
        g = -self.W[p + "A_log"].float().exp() * sf_softplus(a + self.W[p + "dt_bias"].float())   # exp(A_log) is a constant
        rep = c.lin_v_heads // c.lin_k_heads                                   # each k head serves 2 v heads
        q = q.repeat_interleave(rep, dim=1)
        k = k.repeat_interleave(rep, dim=1)
        if CHIP["dnq"]:
            q = q * sf_rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * torch.tensor(1 / math.sqrt(c.lin_k_dim),
                                                                               dtype=torch.float32)
        else:
            q = q * sf_rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(c.lin_k_dim)
        k = k * sf_rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

        S = cache.S[i]                                                         # [32, 128k, 128v]
        outs = []
        if self.num.state.startswith("chunked") and T % 64 == 0:
            # prefill's chunked delta rule (note 30): "chunked_fp32" (reordered maths only), "chunked_int8"
            # (matrix-unit operands int8, one pass), "chunked_split" (both operands as int8 high + low parts)
            import dn_chunked
            mode = {"chunked_fp32": False, "chunked_int8": "int8", "chunked_split": "split",
                    "chunked_lite": "split_lite"}[self.num.state]
            o, cache.S[i] = dn_chunked.run(S, q, k, v, beta, g, chip=mode, exp=sf_exp)
        n_seq = 0 if self.num.state.startswith("chunked") and T % 64 == 0 else T
        onepass = self.num.state == "fp32_onepass"
        for t in range(n_seq):
            if onepass:
                # chip order (note 18): one read of the old S gives both S·k and S·q;
                # o = d·(S_old·q) + (k·q)·delta, then S_new = d·S_old + k ⊗ delta
                d = sf_exp(g[t])[:, None]
                sk = (S * k[t][:, :, None]).sum(1)
                sq = (S * q[t][:, :, None]).sum(1)
                delta = (v[t] - d * sk) * beta[t][:, None]
                outs.append(d * sq + (k[t] * q[t]).sum(-1, keepdim=True) * delta)
                S = d[:, :, None] * S + k[t][:, :, None] * delta[:, None, :]
                continue
            S = S * sf_exp(g[t])[:, None, None]                                  # forget
            kv_mem = (S * k[t][:, :, None]).sum(1)                             # S·k   [32, 128v]
            delta = (v[t] - kv_mem) * beta[t][:, None]                         # correction
            S = S + k[t][:, :, None] * delta[:, None, :]                       # write along k
            if self.num.state == "bf16":
                S = S.bfloat16().float()
            outs.append((S * q[t][:, :, None]).sum(1))                         # read with q
        if outs:
            cache.S[i] = S
            o = torch.stack(outs)                                              # [T, 32, 128]

        o = o * sf_rsqrt(o.pow(2).mean(-1, keepdim=True) + c.eps)
        o = o * self.W[p + "norm.weight"].float() * sf_silu(z)                  # gated RMSNorm
        return self.mm(o.reshape(T, vd), p + "out_proj.weight")

    def ffn(self, i, h, chunk=4096):
        """SwiGLU. At long context the 12,288-wide intermediate is made 4K tokens at a time
        (same arithmetic, less GPU memory)."""
        p = f"layers.{i}.mlp."
        outs = []
        for t0 in range(0, h.shape[0], chunk):
            x = h[t0:t0 + chunk]
            outs.append(self.mm(sf_silu(self.mm(x, p + "gate_proj.weight")) * self.mm(x, p + "up_proj.weight"),
                                p + "down_proj.weight"))
        return torch.cat(outs) if len(outs) > 1 else outs[0]

    # ---------------------------------------------------------------- model
    @torch.no_grad()
    def forward(self, tokens, cache, trace=None, return_hidden=False, return_prenorm=False):
        """Run T new tokens (1 for decode, many for prefill). Returns logits [T, vocab],
        or the final normalised hidden state [T, hidden] if return_hidden (for chunked logits).
        If `trace` is a list, the hidden state after the embedding and after every layer is appended."""
        T = tokens.shape[0]
        positions = torch.arange(cache.pos, cache.pos + T, device=tokens.device)
        h = self.embed(tokens)
        if trace is not None:
            trace.append(h.detach().cpu())
        pr = getattr(self, "probe", None)       # layer-by-layer comparison (note 30): {"mode": "record"|"compare"}
        for i, t in enumerate(self.cfg.layer_types):
            x = self.rms(h, f"layers.{i}.input_layernorm.weight")
            if t == "full_attention":
                mx = self.full_attention(i, x, cache, positions)
            else:
                mx = self.deltanet(i, x, cache)
            h = h + mx
            xf = self.rms(h, f"layers.{i}.post_attention_layernorm.weight")
            f = self.ffn(i, xf)
            h = h + f
            if pr is not None:
                _probe(pr, i, t, dict(mixer_in=x, mixer_out=mx, ffn_in=xf, ffn_out=f, residual=h))
            if trace is not None:
                trace.append(h.detach().cpu())
        cache.pos += T
        pre = h
        h = self.rms(h, "norm.weight")
        if return_prenorm:                      # (before final norm, after final norm) for the MTP head
            return pre, h
        return h if return_hidden else self.mm(h, "lm_head.weight")

    def logits(self, h):
        return self.mm(h, "lm_head.weight")

    def new_cache(self):
        return Cache(self.cfg, self.dev)

    @torch.no_grad()
    def greedy(self, prompt, n_new):
        cache = self.new_cache()
        logits = self.forward(prompt, cache)
        out = []
        nxt = logits[-1].argmax()
        for _ in range(n_new):
            out.append(int(nxt))
            logits = self.forward(nxt.view(1), cache)
            nxt = logits[-1].argmax()
        return out
