"""The chunked gated delta rule for prefill, as the chip would run it (note 30).

The sequential DeltaNet updates the 128 x 128 state once per token (3 passes over it on the vector unit). The chunked
form takes C = 64 tokens at once and turns most of the work into matrix products, which can run on the matrix unit.
Same maths, different order of operations.

Per head, for one chunk of C tokens with incoming state S0 [dk, dv] (rows = key dim, columns = value dim), and per
token t: decay d_t = exp(g_t), write strength beta_t, key k_t (unit length), query q_t, value v_t:

    G_t   = g_1 + ... + g_t                          (cumulative log decay inside the chunk; G_t <= 0)
    gam_t = exp(G_t)                                 (decay from the chunk start to token t)
    D[t,s] = exp(G_t - G_s) for s <= t, else 0       (decay from token s to token t; always <= 1)
    A[t,s] = beta_t (k_t . k_s) D[t,s] for s < t     (strictly lower triangular)
    r_t   = beta_t (v_t - gam_t S0^T k_t)
    solve (I + A) Delta = r                          (forward substitution, C steps; Delta [C, dv])
    o_t   = gam_t S0^T q_t + sum_{s<=t} (q_t . k_s) D[t,s] Delta_s
    S_C   = gam_C S0 + sum_s exp(G_C - G_s) k_s Delta_s^T

Delta_s is exactly the sequential rule's correction term delta_s, so this is the sequential recurrence rearranged.

Where each piece runs on the chip ("chip" mode):
    matrix unit, int8 x int8, fp32 accumulation:
        K K^T, Q K^T        rows k_t / q_t (activation side: int8, power-of-two scale per 32, D-019)
                            weights k_s (weight side: int8, one fp16 scale per k_s, like the KV cache)
        S0^T k, S0^T q      weights = the columns of S0 (int8, one fp16 scale per column): the state's copy
        P Delta             rows P_t = (Q K^T * D)[t, :] (activation side), weights = the columns of Delta
        Ktil^T Delta        rows = the columns of Ktil = k_s exp(G_C - G_s) (activation side), same Delta weights
    vector unit, fp32: G, D, A, r, the triangular solve, gam * (S0^T q) + P Delta, gam_C * S0 + update.
The state S itself stays fp32; only the copy that goes into the matrix unit is int8.
"""
import torch

import bitexact as B

CHECK = True      # stop at the first non-finite value, naming the tensor (cheap: one sync per chunk)


def _qa(x):
    """Activation side of the matrix unit: int8 with one power-of-two scale per 32 along the last dim (D-019)."""
    q, e = B.quant_act_exact(x.float())
    return torch.ldexp(q, e.repeat_interleave(32, -1))


def _qw_parts(x):
    """Weight side for int8 raw operands: int8 with one fp32 scale s per row (applied to each output after the
    product). The low part is the remainder in units of s/256, so all three passes share the row's scale and the
    low passes differ only by an exact 2^-8."""
    s = x.abs().amax(-1, keepdim=True) / 127
    s = torch.where(s > 0, s, torch.ones_like(s))
    hi = torch.clamp(torch.round(x / s), -127, 127)
    lo = torch.clamp(torch.round((x - hi * s) / (s / 256)), -127, 127)
    return hi * s, lo * (s / 256)


def _qw(x):
    return _qw_parts(x)[0]


def _mm_lite(a, b_rows):
    """The no-new-hardware version (note 30, part 2): weight side int8 with one fp16 scale s per row (QKV), the low
    part in units of s/256; the matrix unit sums int8 products (3 passes, the hi*lo pass times 2^-8 by tscale);
    the vector unit then multiplies each output by its s."""
    s = (b_rows.abs().amax(-1, keepdim=True) / 127).half().float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    hi = torch.clamp(torch.round(b_rows / s), -127, 127)
    lo = torch.clamp(torch.round((b_rows - hi * s) * (256 / s)), -127, 127)
    ah = _qa(a)
    al = _qa(a - ah)
    acc = ah @ hi.transpose(-1, -2) + al @ hi.transpose(-1, -2) + (ah @ lo.transpose(-1, -2)) * 2.0 ** -8
    return acc * s.transpose(-1, -2)


def scan_hs(g):
    """Hillis-Steele inclusive scan over the last dim (6 steps for 64): the lite program's order for G."""
    x = g.clone()
    off = 1
    while off < x.shape[-1]:
        y = x.clone()
        y[..., off:] = x[..., off:] + x[..., :-off]
        x, off = y, off * 2
    return x


def _mm(a, b_rows, chip):
    """a [..., M, K] @ b_rows[..., N, K]^T, with the matrix unit's operand formats in chip mode.
    chip: False (fp32), True or "int8" (one pass), "split_a" (the activation side as two int8 passes, high part and
    the rounded remainder, added by the 'acc' epilogue), "split" (both sides split: three passes, hi*hi + hi*lo +
    lo*hi)."""
    if not chip:
        return a @ b_rows.transpose(-1, -2)
    if chip == "split_lite":
        return _mm_lite(a, b_rows)
    ah = _qa(a)
    wh, wl = _qw_parts(b_rows)
    out = ah @ wh.transpose(-1, -2)
    if chip in ("split_a", "split"):
        out = out + _qa(a - ah) @ wh.transpose(-1, -2)
    if chip == "split":
        out = out + ah @ wl.transpose(-1, -2)
    return out


def _check(where, **ts):
    bad = [k for k, t in ts.items() if not torch.isfinite(t).all()]
    if bad:
        info = {k: (t.abs().nan_to_num(posinf=float("inf")).amax().item(), (~torch.isfinite(t)).sum().item())
                for k, t in ts.items()}
        raise FloatingPointError(f"non-finite in {bad} at {where}: (amax, count) {info}")


def chunk_step(S, q, k, v, beta, g, chip=False, exp=torch.exp, stats=None):
    """One chunk for all heads. S [H, dk, dv]; q, k [H, C, dk]; v [H, C, dv]; beta, g [H, C].
    Returns (outputs [H, C, dv], new S)."""
    C = q.shape[1]
    G = scan_hs(g) if chip == "split_lite" else torch.cumsum(g, -1)      # [H, C]
    diff = G[:, :, None] - G[:, None, :]                                 # G_t - G_s
    low = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    D = torch.where(low, exp(torch.where(low, diff, torch.zeros_like(diff))), torch.zeros_like(diff))
    gam = exp(G)                                                         # [H, C]
    KK = _mm(k, k, chip)                                                 # [H, C, C]
    QK = _mm(q, k, chip)
    St = S.transpose(-1, -2)                                             # [H, dv, dk]: rows = columns of S
    KS = _mm(k, St, chip)                                                # [H, C, dv] = (S^T k_t)_t
    QS = _mm(q, St, chip)
    A = torch.tril(beta[:, :, None] * KK * D, -1)
    r = beta[:, :, None] * (v - gam[:, :, None] * KS)
    I = torch.eye(C, device=q.device).expand_as(A)
    Delta = torch.linalg.solve_triangular(I + A, r, upper=False, unitriangular=True)
    if CHECK:
        _check("after solve", S=S, q=q, k=k, v=v, beta=beta, g=g, G=G, D=D, gam=gam, KK=KK, QK=QK, KS=KS, QS=QS,
               A=A, r=r, Delta=Delta)
    if stats is not None:
        stats["delta_over_r"] = max(stats.get("delta_over_r", 0.0),
                                    (Delta.abs().amax() / r.abs().amax().clamp_min(1e-30)).item())
    Dt = Delta.transpose(-1, -2)                                         # [H, dv, C]: rows = columns of Delta
    P = QK * D
    O = gam[:, :, None] * QS + _mm(P, Dt, chip)
    Kt = k * exp(G[:, -1:] - G)[:, :, None]                              # [H, C, dk]
    upd = _mm(Kt.transpose(-1, -2), Dt, chip)                            # [H, dk, dv]
    S = gam[:, -1, None, None] * S + upd
    if CHECK:
        _check("chunk end", O=O, upd=upd, S=S)
    return O, S


def run(S, q, k, v, beta, g, C=64, chip=False, exp=torch.exp, stats=None):
    """Whole sequence. q, k [T, H, dk]; v [T, H, dv]; beta, g [T, H] (the golden model's layout).
    T must be a multiple of C (prefill chunks are 128 tokens). Returns (o [T, H, dv], S)."""
    T = q.shape[0]
    assert T % C == 0, T
    outs = []
    for c0 in range(0, T, C):
        sl = slice(c0, c0 + C)
        o, S = chunk_step(S, q[sl].transpose(0, 1), k[sl].transpose(0, 1), v[sl].transpose(0, 1),
                          beta[sl].T, g[sl].T, chip=chip, exp=exp, stats=stats)
        outs.append(o.transpose(0, 1))
    return torch.cat(outs), S


def sequential(S, q, k, v, beta, g, dtype=torch.float32):
    """The reference recurrence (qwen35_golden's default order), in a chosen precision."""
    S = S.to(dtype)
    outs = []
    for t in range(q.shape[0]):
        S = S * torch.exp(g[t].to(dtype))[:, None, None]
        kt, qt = k[t].to(dtype), q[t].to(dtype)
        delta = (v[t].to(dtype) - (S * kt[:, :, None]).sum(1)) * beta[t].to(dtype)[:, None]
        S = S + kt[:, :, None] * delta[:, None, :]
        outs.append((S * qt[:, :, None]).sum(1))
    return torch.stack(outs), S


def solve_r8(A, r):
    """Order rule R8's solve: right-looking forward substitution with fused multiply-adds (bitexact_vec.fma).
    A [C, C] strictly lower, r [C, dv]. After each Delta_s, every later r_t <- fma(-A[t, s], Delta_s, r_t)."""
    import bitexact_vec as BV
    r = r.clone()
    C = A.shape[0]
    for s in range(C - 1):
        r[s + 1:] = BV.fma(-A[s + 1:, s:s + 1].expand(-1, r.shape[1]), r[s:s + 1].expand(C - s - 1, -1), r[s + 1:])
    return r
