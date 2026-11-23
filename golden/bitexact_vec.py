"""Golden v2, part 2: the vector unit's operation order, bit-exact (progression step 4, note 29).

The vector unit has L = 512 fp32 lanes (D-023), each with a fused multiply-add (one rounding). The bits of a
result depend on the order of fp32 operations, so this file fixes that order; the RTL and the instruction-level
simulator must follow it. All arithmetic is fp32 with round-to-nearest-even; results below 2^-126 flush to +0.

Proposed order rules (part of D-022's contract):
  R1 reductions over n values (RSUM, RSUMSQ): lane j accumulates x[j], x[j+L], x[j+2L], ... in sequence
     (RSUMSQ: fma(x, x, acc)); then a pairwise tree over the L lanes: level by level, (0+1), (2+3), ...
     With n < L the missing lanes hold +0. RMAX is order-free.
  R2 elementwise ops: one rounding each; FMA = fma(a, b, c) with one rounding; f(a*b + c) for the special
     functions rounds a*b + c once (fma), then applies the function (chip_sfu.py).
  R3 RMSNorm: s = RSUMSQ(x); r = rsqrt(fma(s, 1/n, eps)); y = x * r; out = int8(y * w)  (two roundings, in that order)
  R4 DeltaNet head step (one token, one-pass order, note 18), lanes = 4 rows x 128 columns:
       d = exp(g)
       sk[j] = sum over rows i of S[i][j] * k[i], sq[j] likewise with q:
               lane (r, j) does fma over rows i = r, r+4, r+8, ... in order; then (p0 + p1) + (p2 + p3)
       kq = R1 reduction of k[i] * q[i] (products rounded, then summed by R1)
       delta[j] = (v[j] - d * sk[j]) * beta            (mul, sub, mul: three roundings)
       o[j]     = fma(kq, delta[j], d * sq[j])
       S[i][j]  = fma(k[i], delta[j], d * S[i][j])
     Replay (commit) of logged tokens uses the same sk, delta and S update, without sq or o.
  R5 kernel-4 conv: y = fma(w3, x[t], fma(w2, x[t-1], fma(w1, x[t-2], w0 * x[t-3])))
  R6 RoPE (first 64 of 256 dims, halves of 32). Angles from two fp32 tables made in float64 offline:
       A = (p >> 10) * 1024 * f_i, B = (p & 1023) * f_i, f_i = theta^(-2i/64), i = 0..31
       c = fma(cA, cB, sA * (-sB)),  s = fma(cA, sB, sA * cB)          (products rounded first; the program's order)
       y_lo[i] = fma(-s, x_hi[i], c * x_lo[i]),  y_hi[i] = fma(s, x_lo[i], c * x_hi[i]);  dims 64..255 unchanged
  R8 chunked DeltaNet triangular solve (prefill, D-025): Delta = r; for s = 0..62, for every t > s:
       Delta[t] = fma(An[t, s], Delta[s], Delta[t]), with An = -A already negated (one rounding per step)
  R7 fused softmax tile (SMTILE), per row: s' = s * ks (rounded); masked -> -inf; mt = max; mn = max(m, mt);
     (masked = an input of -inf, written by a CAUSAL command before SMTILE (2026-11-21, note 31 part 2): it stays
     -inf whatever ks is; the `mask` argument / the limit field of smtile_row.hex is the same thing)
     p = exp(s' - mn); lt = R1 sum of p; alpha = exp(m - mn); l = fma(l, alpha, lt); P = int8_p2(p * vs)
"""
import torch

import bitexact as B
import chip_sfu as S

L = 512


def _f(x):
    return torch.where(x.abs() < 2.0 ** -126, torch.zeros_like(x), x)


def fma(a, b, c):
    """fp32 fused multiply-add with one rounding, computed exactly: a*b is exact in float64 (48 bits); the add is
    done with TwoSum (s + e = a*b + c exactly); rounding s to fp32 is then correct unless s sits exactly on an
    fp32 midpoint while e != 0 (the double-rounding case), which is fixed by stepping toward e."""
    a, b, c = torch.broadcast_tensors(a.float(), b.float(), c.float())
    x, y = a.double() * b.double(), c.double()
    s_ = x + y
    bp = s_ - x
    e = (x - (s_ - bp)) + (y - bp)
    r = s_.float()
    toward = torch.where(s_ > r.double(), torch.full_like(r, float("inf")), torch.full_like(r, float("-inf")))
    nb = torch.nextafter(r, toward)
    mid = (r.double() + nb.double()) / 2
    tie = (s_ == mid) & (e != 0) & (s_ != r.double())
    # at a midpoint, the exact value is past it in the direction of e: pick the neighbour on that side
    fix = torch.where((e > 0) == (nb.double() > r.double()), nb, r)
    return _f(torch.where(tie, fix, r))


def _check_fma(n=200_000, seed=1):
    """Against an exact rational reference on random inputs built to hit midpoints."""
    from fractions import Fraction
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g)
    b = torch.randn(n, generator=g)
    c = -(a.double() * b.double()).float() + torch.randn(n, generator=g) * 1e-6
    r = fma(a, b, c)
    bad = 0
    for i in range(0, n, 997):
        ex = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
        want = torch.tensor(float(ex), dtype=torch.float64).float()     # Fraction -> nearest double -> fp32
        lo, hi = sorted([want, torch.nextafter(want, torch.tensor(float("inf") if float(ex) > float(want) else
                                                                    float("-inf")))])
        best = min((lo, hi), key=lambda t: (abs(Fraction(float(t)) - ex), int(t.view(torch.int32)) & 1))
        if float(r[i]) != float(best) and abs(float(best)) >= 2.0 ** -126:
            bad += 1
    return bad


def reduce_sum(x, square=False):
    """R1 over the last dimension."""
    x = x.float()
    n = x.shape[-1]
    pad = (-n) % L
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.reshape(*x.shape[:-1], -1, L)
    acc = torch.zeros(*x.shape[:-2], L, dtype=torch.float32)
    for c in range(x.shape[-2]):
        v = x[..., c, :]
        acc = fma(v, v, acc) if square else _f(acc + v)
    while acc.shape[-1] > 1:
        acc = _f(acc[..., 0::2] + acc[..., 1::2])
    return acc[..., 0]


def rmsnorm_int8(x, w, eps):
    s = reduce_sum(x, square=True)
    r = S.rsqrt(fma(s, torch.tensor(1.0 / x.shape[-1]), torch.tensor(eps, dtype=torch.float32)))
    y = _f(x * r[..., None])
    return B.quant_act_exact(_f(y * w))


def dn_token(Sm, k, q, v, beta, g, out=True, d=None):
    """R4 for one head, one token. Sm [128, 128] (rows i = key dim, columns j = value dim). The DN command gets
    the decay d = exp(g) already made by an EXP command; pass d to skip the exp."""
    if d is None:
        d = S.exp(g.reshape(1))[0]
    parts_k, parts_q = [], []
    for r in range(4):
        ak = torch.zeros(128)
        aq = torch.zeros(128)
        for i in range(r, 128, 4):
            ak = fma(Sm[i], k[i].expand(128), ak)
            if out:
                aq = fma(Sm[i], q[i].expand(128), aq)
        parts_k.append(ak)
        parts_q.append(aq)
    sk = _f(_f(parts_k[0] + parts_k[1]) + _f(parts_k[2] + parts_k[3]))
    delta = _f(_f(v - _f(d * sk)) * beta)
    o = None
    if out:
        sq = _f(_f(parts_q[0] + parts_q[1]) + _f(parts_q[2] + parts_q[3]))
        kq = reduce_sum(_f(k * q))
        o = fma(kq.expand(128), delta, _f(d * sq))
    Sn = fma(k[:, None].expand(128, 128), delta[None, :].expand(128, 128), _f(d * Sm))
    return Sn, o


def dn_step(Sm, log, toks):
    """A DN command: replay the logged (committed) tokens, then run the new tokens on a copy (note 28).
    log/toks: lists of (k, q, v, beta, g). Returns (committed state, outputs of the new tokens)."""
    for (k, q, v, beta, g) in log:
        Sm, _ = dn_token(Sm, k, q, v, beta, g, out=False)
    work, outs = Sm.clone(), []
    for (k, q, v, beta, g) in toks:
        work, o = dn_token(work, k, q, v, beta, g)
        outs.append(o)
    return Sm, outs


def conv4(x, hist, w):
    """R5: kernel-4 causal depthwise conv over time. x [T, C] new rows, hist [3, C] the 3 previous rows (oldest
    first), w [C, 4] with w[:, 0] for the oldest input. Returns (y [T, C] before SiLU, new history [3, C])."""
    xp = torch.cat([hist.float(), x.float()])
    T = x.shape[0]
    y = _f(w[:, 0] * xp[0:T])
    for j in (1, 2, 3):
        y = fma(w[:, j].expand_as(y), xp[j:j + T], y)
    return y, xp[-3:].clone()


def rope_tables(theta=1e7, rot=64, n=1024):
    """The two fp32 tables (float64 offline, rounded once): rows p -> (cos, sin) of p * 1024 * f and of p * f."""
    f = 1.0 / (theta ** (torch.arange(0, rot, 2).double() / rot))
    hi = torch.arange(n).double()[:, None] * 1024 * f
    lo = torch.arange(n).double()[:, None] * f
    return hi.cos().float(), hi.sin().float(), lo.cos().float(), lo.sin().float()


def rope_cs(pos, tables):
    """R6, the angle addition: positions [T] (int) -> c, s [T, 32]."""
    cH, sH, cL, sL = tables
    a, b = pos.long() >> 10, pos.long() & 1023
    cA, sA, cB, sB = cH[a], sH[a], cL[b], sL[b]
    c = fma(cA, cB, _f(sA * -sB))
    s = fma(cA, sB, _f(sA * cB))
    return c, s


def rope(x, c, s, rot=64):
    """R6, the rotation: x [T, H, D]; c, s [T, rot/2]."""
    h = rot // 2
    lo, hi = x[..., :h].float(), x[..., h:rot].float()
    c, s = c[:, None, :].expand_as(lo), s[:, None, :].expand_as(lo)
    ylo = fma(-s, hi, _f(c * lo))
    yhi = fma(s, lo, _f(c * hi))
    return torch.cat([ylo, yhi, x[..., rot:].float()], -1)


def solve_r8(An, r):
    """R8: An [C, C] strictly lower, already negated (-A); r [C, dv]. Right-looking forward substitution."""
    d = r.float().clone()
    C = An.shape[0]
    for s_ in range(C - 1):
        d[s_ + 1:] = fma(An[s_ + 1:, s_:s_ + 1].expand(-1, d.shape[1]), d[s_:s_ + 1].expand(C - s_ - 1, -1), d[s_ + 1:])
    return d


def smtile_row(s, ks, vs, m, l, mask=None):
    """R7 for one row of one tile. Returns (P int8 values, P exponents, new m, new l, alpha).
    mask None: the -inf inputs (set by CAUSAL) are the masked ones."""
    if mask is None:
        mask = torch.isneginf(s)
    s = _f(s.float() * ks.float())
    if mask is not None:
        s = torch.where(mask, torch.full_like(s, float("-inf")), s)
    mn = torch.maximum(m, s.max())
    p = S.exp(s - mn)
    lt = reduce_sum(p)
    alpha = S.exp((m - mn).reshape(1))[0]
    ln = fma(l, alpha, lt)
    pv = _f(p * vs.float())
    pad = (-pv.shape[-1]) % 32
    q, e = B.quant_act_exact(torch.nn.functional.pad(pv, (0, pad)))
    return q, e, mn, ln, alpha


def self_test(seed=0):
    """The exact-order versions must agree with plain fp32 torch to ~fp32 rounding, and be deterministic."""
    g = torch.Generator().manual_seed(seed)
    res = {}
    x = torch.randn(3, 4096, generator=g) * 3
    res["reduce_sum_rel"] = ((reduce_sum(x).double() - x.double().sum(-1)).abs() / x.double().abs().sum(-1)).max().item()
    Sm = torch.randn(128, 128, generator=g) * 0.1
    k = torch.nn.functional.normalize(torch.randn(128, generator=g), dim=0)
    q = torch.nn.functional.normalize(torch.randn(128, generator=g), dim=0) / 128 ** 0.5
    v = torch.randn(128, generator=g)
    beta, gg = torch.tensor(0.7), torch.tensor(-0.05)
    Sn, o = dn_token(Sm, k, q, v, beta, gg)
    d = torch.exp(gg.double())
    sk = (Sm.double() * k.double()[:, None]).sum(0)
    sq = (Sm.double() * q.double()[:, None]).sum(0)
    delta = (v.double() - d * sk) * beta.double()
    o_ref = d * sq + (k.double() * q.double()).sum() * delta
    S_ref = d * Sm.double() + k.double()[:, None] * delta[None, :]
    res["dn_o_rel"] = ((o.double() - o_ref).abs().max() / o_ref.abs().max()).item()
    res["dn_S_rel"] = ((Sn.double() - S_ref).abs().max() / S_ref.abs().max()).item()
    # R5 / R6 / R8 against float64
    x = torch.randn(9, 64, generator=g); hist = torch.randn(3, 64, generator=g); w = torch.randn(64, 4, generator=g) * 0.5
    y, _ = conv4(x, hist, w)
    xp = torch.cat([hist, x]).double()
    yr = sum(w[:, j].double() * xp[j:j + 9] for j in range(4))
    res["conv4_rel"] = ((y.double() - yr).abs().max() / yr.abs().max()).item()
    tb = rope_tables()
    pos = torch.tensor([0, 1, 1023, 1024, 4097, 32767, 262143])
    c, s_ = rope_cs(pos, tb)
    f = 1.0 / (1e7 ** (torch.arange(0, 64, 2).double() / 64))
    ang = pos.double()[:, None] * f
    res["rope_cs_abs"] = max((c.double() - ang.cos()).abs().max().item(), (s_.double() - ang.sin()).abs().max().item())
    xr = torch.randn(7, 2, 256, generator=g)
    yr_ = rope(xr, c, s_)
    lo, hi = xr[..., :32].double(), xr[..., 32:64].double()
    C_, S_ = ang.cos()[:, None], ang.sin()[:, None]
    ref = torch.cat([lo * C_ - hi * S_, hi * C_ + lo * S_, xr[..., 64:].double()], -1)
    res["rope_rel"] = ((yr_.double() - ref).abs().max() / ref.abs().max()).item()
    A = torch.tril(torch.randn(64, 64, generator=g) * 0.1, -1)
    rr = torch.randn(64, 128, generator=g)
    dl = solve_r8(-A, rr)
    dref = torch.linalg.solve_triangular(torch.eye(64).double() + A.double(), rr.double(), upper=False)
    res["solve_r8_rel"] = ((dl.double() - dref).abs().max() / dref.abs().max()).item()
    Sn2, o2 = dn_token(Sm, k, q, v, beta, gg)
    res["deterministic"] = bool(torch.equal(Sn, Sn2) and torch.equal(o, o2))
    return res


if __name__ == "__main__":
    print(self_test())
    print("fma mismatches vs exact rational reference:", _check_fma())
