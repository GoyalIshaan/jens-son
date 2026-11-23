"""Golden v2, part 1: the matrix unit's arithmetic, bit-exact (progression step 4).

This is the contract the RTL (rtl/mxu) and the instruction-level simulator must match bit for bit.

Activations (D-019): per 32 values, exponent e = the smallest integer with amax <= 127 * 2^e, computed from
integers only: amax = m * 2^x with m in [0.5, 1) gives e = x - 7 if m <= 127/128 else x - 6. Each value is
q = clamp(round_half_even(v / 2^e), -127, 127).

Weights (D-005, lut4_g16_fp8_t8): code c -> table entry k[c] (int8, -127..127, meaning k/127); per 16 weights an
E4M3 scale s8 = M * 2^E (M an integer); per tensor an fp32 scale gmax. Weight = (k / 127) * s8 * gmax.

One output, one beat (1,024 inputs = 64 groups of 16):
  S_g  = sum of 16 products k * q                       exact integer, |S_g| <= 258,064 (19 bits)
  P_g  = S_g * M_g (signed)                             exact, 24 bits;   E_g = E_g(weight) + e(act, its 32-block)
  emax = max over all 64 groups of E_g (zero groups included, as the RTL does)
  T_g  = floor(P_g * 2^(16 - (emax - E_g)))             16 fraction bits kept, rounded down (note 26)
  v    = fp32_RNE(sum T_g * 2^(emax - 16)), flushed to +0 below 2^-126
  acc  = fp32_RNE(acc + v), beat after beat in input order (also flushed)
Then, once per output (the matrix unit's output stage, found here: the per-tensor scale is not a power of two):
  y    = fp32_RNE(acc * fp32(gmax / 127))
"""
import math

import torch

TINY = 2.0 ** -126


def act_exponent(amax):
    """Integer rule for D-019's scale exponent. amax > 0, any shape (float32)."""
    m, x = torch.frexp(amax.float())
    return torch.where(m <= 127 / 128, x - 7, x - 6)


def quant_act_exact(x, g=32):
    """x [..., K] fp32 -> (q int8 values as float, e exponents [..., K/g])."""
    shp = x.shape
    y = x.float().reshape(*shp[:-1], shp[-1] // g, g)
    amax = y.abs().amax(-1)
    e = torch.where(amax > 0, act_exponent(amax.clamp_min(1e-38)), torch.full_like(amax, -126, dtype=torch.int32))
    # a group whose exponent would fall below -126 (amax under about 2^-119) is all zero: its values would be
    # flushed to zero anyway (fp32 below 2^-126 flushes, R-rules), and 2^-e would overflow fp32
    tiny = e < -126
    e = torch.where(tiny, torch.full_like(e, -126), e)
    q = torch.clamp(torch.round(torch.ldexp(y, -e[..., None])), -127, 127)
    q = torch.where(tiny[..., None], torch.zeros_like(q), q)
    return q.reshape(shp), e


def e4m3_split(s8):
    """E4M3 values (as float) -> (M integer, E) with s8 = M * 2^E exactly (sign kept in M)."""
    a = s8.abs()
    m, x = torch.frexp(a)                              # a = m 2^x, m in [0.5, 1)
    M = torch.round(m * 16)                            # 4 significant bits (8..15) for normals
    E = x - 4
    sub = a < 2.0 ** -6                                # E4M3 subnormals: M * 2^-9, M in 1..7
    M = torch.where(sub, torch.round(a * 2.0 ** 9), M)
    E = torch.where(sub, torch.full_like(E, -9), E)
    M = torch.where(a == 0, torch.zeros_like(M), M)
    return M * torch.sign(s8), E


def weight_parts(qt):
    """QTensor (lut4_g16_fp8_t8) -> integer table entries per weight, E4M3 (M, E) per 16, tensor scale."""
    N, K = qt.shape
    k = torch.round(qt.table * 127)[qt.codes.long()].reshape(N, K)
    assert qt.group == 16 and qt.gmax is not None
    s8 = (qt.scales / qt.gmax).to(torch.float8_e4m3fn).float()
    assert torch.equal((s8 * qt.gmax).clamp_min(1e-12), qt.scales), "scales are not E4M3 x gmax"
    M, E = e4m3_split(s8)
    tscale = torch.tensor(float(qt.gmax) / 127.0, dtype=torch.float32)
    return k, M, E, tscale


def _flush(t):
    return torch.where(t.abs() < TINY, torch.zeros_like(t), t)


def mm_exact(q, e, k, M, E, tscale, F=16, n_chunk=1024, beat=1024):
    """q [T, K] int8-valued, e [T, K/32]; k [N, K]; M, E [N, K/16]. Returns y [T, N] fp32, bit-exact.
    beat: inputs per beat and output, 1,024 (seg 1) or 256 (seg 4: each segment has its own emax, note 26)."""
    T, K = q.shape
    N = k.shape[0]
    G = K // 16
    beats = K // beat
    qd = q.double().reshape(T, G, 16)
    ea = e.repeat_interleave(2, dim=-1).double()                      # each 16-group uses its 32-block exponent
    out = torch.empty(T, N, dtype=torch.float32)
    for n0 in range(0, N, n_chunk):
        kd = k[n0:n0 + n_chunk].double().reshape(-1, G, 16)
        S = torch.einsum("tgi,ngi->tng", qd, kd)                       # exact integers
        P = S * M[n0:n0 + n_chunk].double()[None]
        Eg = E[n0:n0 + n_chunk].double()[None] + ea[:, None, :]
        P = P.reshape(T, -1, beats, beat // 16)
        Eg = Eg.reshape(T, -1, beats, beat // 16)
        emax = Eg.amax(-1, keepdim=True)
        Tg = torch.floor(P * torch.pow(2.0, F - (emax - Eg)))
        v = _flush((Tg.sum(-1) * torch.pow(2.0, emax[..., 0] - F)).float())
        acc = torch.zeros(T, v.shape[1], dtype=torch.float32)
        for b in range(beats):
            acc = _flush(acc + v[..., b])
        out[:, n0:n0 + n_chunk] = _flush(acc * tscale)
    return out


def self_test(seed=0, T=4, N=512, K=4096):
    """Random but realistic: gaussian weights quantized with the chip format, gaussian activations with outliers.
    Checks (1) the integer exponent rule against formats.quant_act, (2) the exact matrix product against the
    float reference (dequantized x @ dequantized w), which must agree to fp32 accumulation rounding."""
    import formats
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g) * 0.02
    qt = formats.quantize_weight(w, "lut4_g16_fp8_t8")
    x = torch.randn(T, K, generator=g)
    x[:, ::97] *= 30                                                  # a few outliers, as in real activations
    q, e = quant_act_exact(x)
    xr = torch.ldexp(q.reshape(T, K // 32, 32), e[..., None]).reshape(T, K)
    xf = formats.quant_act(x, "int8_g32_p2")
    act_mismatch = (xr != xf).sum().item()
    k, M, E, ts = weight_parts(qt)
    y = mm_exact(q, e, k, M, E, ts)
    ref = (xf.double() @ qt.dequant().double().T)
    rel = ((y.double() - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    return dict(act_values_different=act_mismatch, max_rel_vs_float64=rel, y_mean_abs=y.abs().mean().item())


if __name__ == "__main__":
    r = self_test()
    print(r)
    print(f"exact product vs float64 reference: max relative difference {r['max_rel_vs_float64']:.2e} "
          f"(2^{math.log2(max(r['max_rel_vs_float64'], 1e-30)):.1f})")
