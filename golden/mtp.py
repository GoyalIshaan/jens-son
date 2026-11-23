"""Qwen3.5-9B's built-in multi-token-prediction (MTP) head, used as the chip's draft model.

From the checkpoint (15 tensors under `mtp.`; embedding and LM head are shared with the main model):
    e = RMSNorm_e(embed(next token))          mtp.pre_fc_norm_embedding
    h = RMSNorm_h(main model hidden state)    mtp.pre_fc_norm_hidden
    x = fc([e ; h])                           mtp.fc: 8192 -> 4096
    x = one decoder layer                     gated full attention (same as the main model's) + SwiGLU
    out = RMSNorm(x)                          mtp.norm
    draft = argmax(lm_head(out))              optionally over a trimmed vocabulary
The next draft in a chain feeds (out, draft) back in at the next position.

Two details are not stated anywhere we can cite, so they are *measured* rather than assumed
(see beam_exp4.py, "variants"): the order of the concatenation, and whether the main hidden
state is taken before or after the main model's final norm. Each norm's weight convention
(stored as w or as an offset from 1) is read off the stored values.

`drafts()` runs every position of a sequence at once (teacher-forced main hidden states),
including chained drafts 2 and 3. A chained query at position i attends to the MTP keys of
positions <= i plus the keys of its own earlier chain steps, exactly as during decoding.
"""
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from formats import quant_kv, quantize_weight

P = "mtp.layers.0."


def load_mtp(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for k, f in idx.items():
        if k.startswith("mtp."):
            with safe_open(os.path.join(model_dir, f), framework="pt", device="cpu") as sf:
                out[k] = sf.get_tensor(k)
    return out


class MTP:
    def __init__(self, gold, model_dir, fmt=None, order="emb_first", hidden="post"):
        self.g, self.order, self.hidden = gold, order, hidden
        raw = load_mtp(model_dir)
        self.n_tensors = len(raw)
        self.matrix_params = sum(t.numel() for t in raw.values() if t.dim() == 2)
        self.weight_error = {}
        for n, t in raw.items():
            t = t.to(gold.dev)
            if fmt and t.dim() == 2:
                q = quantize_weight(t, fmt)
                err = (q.dequant() - t.float()).pow(2).sum().item() / t.float().pow(2).sum().item()
                self.weight_error[n] = err ** 0.5
                gold.W[n] = q
            else:
                gold.W[n] = t
        # norms: same zero-centred convention as the main model (y * (1 + w)), as in vLLM's Qwen3-Next MTP,
        # which uses the main model's RMSNorm class. Means are recorded next to the main model's for comparison.
        self.norm_mean = {n: raw[n].float().mean().item() for n in raw if raw[n].dim() == 1}
        self.offset = {n: True for n in self.norm_mean}
        self.main_norm_mean = {n: gold.W[n].float().mean().item() for n in
                               ("norm.weight", "layers.3.input_layernorm.weight", "layers.3.self_attn.q_norm.weight",
                                "layers.3.self_attn.k_norm.weight", "layers.3.post_attention_layernorm.weight")}
        self.shapes = {n: list(t.shape) for n, t in raw.items()}

    # ---------------------------------------------------------------- pieces
    def rms(self, x, name):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.g.cfg.eps)
        w = self.g.W[name].float()
        return y * (1.0 + w) if self.offset[name] else y * w

    def inputs(self, H, tok):
        e = self.rms(self.g.embed(tok), "mtp.pre_fc_norm_embedding.weight")
        h = self.rms(H, "mtp.pre_fc_norm_hidden.weight")
        x = torch.cat([e, h], -1) if self.order == "emb_first" else torch.cat([h, e], -1)
        return self.g.mm(x, "mtp.fc.weight")

    def qkv(self, x, positions):
        g, c, p = self.g, self.g.cfg, P + "self_attn."
        T = x.shape[0]
        xn = self.rms(x, P + "input_layernorm.weight")
        qg = g.mm(xn, p + "q_proj.weight").view(T, c.q_heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:].reshape(T, -1)
        k = g.mm(xn, p + "k_proj.weight").view(T, c.kv_heads, c.head_dim)
        v = g.mm(xn, p + "v_proj.weight").view(T, c.kv_heads, c.head_dim)
        q = g.rope(self.rms(q, p + "q_norm.weight"), positions)
        k = g.rope(self.rms(k, p + "k_norm.weight"), positions)
        return q, gate, quant_kv(k, g.num.kv), quant_kv(v, g.num.kv)

    def attend(self, q, K, V, extra):
        """q [T,16,D]; K,V [T,4,D] teacher-forced keys (causal); extra: [(k,v)] chain keys, one per position."""
        c = self.g.cfg
        T, D, rep = q.shape[0], c.head_dim, c.q_heads // c.kv_heads
        qh = q.transpose(0, 1).reshape(c.kv_heads, rep, T, D)
        Kh, Vh = K.transpose(0, 1)[:, None], V.transpose(0, 1)[:, None]
        sc = qh @ Kh.transpose(-1, -2) / math.sqrt(D)                                   # [4, rep, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
        sc = sc.masked_fill(causal[None, None], float("-inf"))
        ex = [(qh * k.transpose(0, 1)[:, None]).sum(-1, keepdim=True) / math.sqrt(D) for k, _ in extra]
        pr = torch.softmax(torch.cat([sc] + ex, -1), -1)
        out = pr[..., :T] @ Vh
        for j, (_, v) in enumerate(extra):
            out = out + pr[..., T + j:T + j + 1] * v.transpose(0, 1)[:, None]
        return out.reshape(c.q_heads, T, D).transpose(0, 1)                             # [T, 16, D]

    def finish(self, x, att, gate):
        g, p = self.g, P
        T = x.shape[0]
        x = x + g.mm(att.reshape(T, -1) * torch.sigmoid(gate), p + "self_attn.o_proj.weight")
        xn = self.rms(x, p + "post_attention_layernorm.weight")
        x = x + g.mm(F.silu(g.mm(xn, p + "mlp.gate_proj.weight")) * g.mm(xn, p + "mlp.up_proj.weight"),
                     p + "mlp.down_proj.weight")
        return x, self.rms(x, "mtp.norm.weight")

    def argmax(self, h, vocab=None, chunk=256):
        out = []
        for r in range(0, h.shape[0], chunk):
            lg = self.g.logits(h[r:r + chunk])
            out.append(lg.argmax(-1) if vocab is None else vocab[lg[:, vocab].argmax(-1)])
        return torch.cat(out)

    # ---------------------------------------------------------------- windowed drafting (long sequences)
    def attend_at(self, q, idx, K, V, extra, window=None, chunk=128):
        """Like attend(), but only for query positions `idx` (sorted), in chunks, so a 32K sequence
        never builds a T x T score matrix. Query at position i sees MTP keys j with j <= i and, if
        `window` is set, j > i - window; plus its own chain keys `extra` (always inside the window)."""
        c = self.g.cfg
        D, rep = c.head_dim, c.q_heads // c.kv_heads
        outs = []
        for s in range(0, len(idx), chunk):
            ii = idx[s:s + chunk]
            n = ii.shape[0]
            hi = int(ii.max()) + 1
            lo = 0 if window is None else max(0, int(ii.min()) - window + 1)
            Kc, Vc = K[lo:hi].transpose(0, 1)[:, None], V[lo:hi].transpose(0, 1)[:, None]  # [4,1,L,D]
            qh = q[s:s + n].transpose(0, 1).reshape(c.kv_heads, rep, n, D)
            sc = qh @ Kc.transpose(-1, -2) / math.sqrt(D)                                   # [4,rep,n,L]
            j = torch.arange(lo, hi, device=q.device)[None, :]
            bad = j > ii[:, None]
            if window is not None:
                bad |= j <= ii[:, None] - window
            sc = sc.masked_fill(bad[None, None], float("-inf"))
            ex = [(qh * k[s:s + n].transpose(0, 1)[:, None]).sum(-1, keepdim=True) / math.sqrt(D) for k, _ in extra]
            pr = torch.softmax(torch.cat([sc] + ex, -1), -1)
            L = hi - lo
            o = pr[..., :L] @ Vc
            for e, (_, v) in enumerate(extra):
                o = o + pr[..., L + e:L + e + 1] * v[s:s + n].transpose(0, 1)[:, None]
            outs.append(o.reshape(c.q_heads, n, D).transpose(0, 1))
        return torch.cat(outs)

    @torch.no_grad()
    def drafts_at(self, H, tok_next, idx, k_max, vocab=None, window=None):
        """Drafts for the positions `idx` only. H, tok_next as in drafts(). Returns [k_max, len(idx)].
        With window=None this equals drafts()[:, idx] (checked in beam_exp4.mtp_window)."""
        T = H.shape[0]
        pos = torch.arange(T, device=H.device)
        x_all = self.inputs(H, tok_next)
        q_all, gate_all, K, V = self.qkv(x_all, pos)
        x, q, gate = x_all[idx], q_all[idx], gate_all[idx]
        del q_all, gate_all
        pre, post = self.finish(x, self.attend_at(q, idx, K, V, [], window), gate)
        toks, extra = [], []
        for j in range(k_max):
            a = self.argmax(post, vocab)
            toks.append(a)
            if j == k_max - 1:
                break
            x = self.inputs(post if self.hidden == "post" else pre, a)
            q, gate, k, v = self.qkv(x, idx + j + 1)
            extra.append((k, v))
            pre, post = self.finish(x, self.attend_at(q, idx, K, V, extra, window), gate)
        return torch.stack(toks)

    # ---------------------------------------------------------------- drafting
    @torch.no_grad()
    def drafts(self, H, tok_next, k_max, vocab=None):
        """H [T,4096]: main hidden at positions 0..T-1; tok_next[i] = token at i+1.
        Returns [k_max, T]: draft j (0-based) at position i predicts the token at i + j + 2."""
        T = H.shape[0]
        pos = torch.arange(T, device=H.device)
        x = self.inputs(H, tok_next)
        q, gate, K, V = self.qkv(x, pos)
        pre, post = self.finish(x, self.attend(q, K, V, []), gate)
        toks, extra = [], []
        for j in range(k_max):
            a = self.argmax(post, vocab)
            toks.append(a)
            if j == k_max - 1:
                break
            x = self.inputs(post if self.hidden == "post" else pre, a)
            q, gate, k, v = self.qkv(x, pos + j + 1)
            extra.append((k, v))
            pre, post = self.finish(x, self.attend(q, K, V, extra), gate)
        return torch.stack(toks)
