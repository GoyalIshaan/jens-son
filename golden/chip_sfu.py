"""The chip's special-function unit as actual algorithms (D-020), in fp32 torch.

Experiment set 6 emulated the unit by rounding exact results to b mantissa bits. This file computes them
the way the hardware will: integer shifts, small tables, one polynomial or Newton step, then rounding to
the unit's output width. Golden v2 (progression step 4) uses these; experiment set 8 tests their quality.

  exp(x)     2^(x log2 e): integer part as an exponent shift, fraction via a 64-entry table of 2^(i/64)
             times a quadratic in the remainder.                                  16 mantissa bits
  1/x        256-entry table on the mantissa + one Newton step.                    12 bits (D-020)
  1/sqrt(x)  256-entry table on [1, 4) + one Newton step.                          16 bits
  sigmoid    1 / (1 + exp(-|x|)) from the two units above, mirrored for x < 0.     12 bits
  silu       x * sigmoid(x) (one fp32 multiply).
  softplus   max(x, 0) + ln(1 + exp(-|x|)); ln from a 64-entry log2 table + quadratic
             (a 3-term series when exp(-|x|) < 2^-6).                         12 bits

Conventions: results below 2^-126 flush to 0 (as the fp32 adder, note 26); inputs are finite or -inf.
"""
import math

import torch

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
_cache = {}


def _tab(name, dev):
    key = (name, dev)
    if key not in _cache:
        i = torch.arange(256 if name != "exp2" and name != "log2" else 64, dtype=torch.float64)
        if name == "exp2":
            t = 2.0 ** (i / 64)
        elif name == "log2":
            t = torch.stack([torch.log2(1 + i / 64), 1 / ((1 + i / 64) * LN2), -1 / (2 * LN2 * (1 + i / 64) ** 2)])
        elif name == "recip":
            t = 1 / (1 + (i + 0.5) / 256)
        elif name == "rsqrt":
            t = 1 / torch.sqrt(1 + (i + 0.5) * 3 / 256)
        _cache[key] = t.float().to(dev)
    return _cache[key]


def round_bits(y, b):
    m, e = torch.frexp(y)
    return torch.ldexp(torch.round(m * 2.0 ** (b + 1)) / 2.0 ** (b + 1), e)


def _flush(y):
    return torch.where(y.abs() < 2.0 ** -126, torch.zeros_like(y), y)


def _chunked(f):
    """Large tensors (attention scores) in pieces of 4M values: the algorithms make several full-size
    temporaries, and an 8K x 8K score block would not fit next to the model."""
    def g(x, *a, **k):
        if x.numel() <= 1 << 22:
            return f(x, *a, **k)
        flat = x.reshape(-1)
        out = torch.empty(flat.shape, dtype=torch.float32, device=x.device)
        for i in range(0, flat.numel(), 1 << 22):
            out[i:i + (1 << 22)] = f(flat[i:i + (1 << 22)], *a, **k)
        return out.reshape(x.shape)
    g.__name__, g.__doc__ = f.__name__, f.__doc__
    return g


@_chunked
def exp(x, bits=16):
    x = x.float()
    y = (x * LOG2E).clamp(-160.0, 127.0)
    n = torch.floor(y)
    f = y - n                                                  # [0, 1)
    i = torch.clamp(torch.floor(f * 64), 0, 63)
    r = (f - i / 64) * LN2                                     # 0 <= r < ln2/64
    m = _tab("exp2", x.device)[i.long()] * (1 + r + r * r * 0.5)
    out = _flush(torch.ldexp(m, n))
    return round_bits(out, bits)


@_chunked
def recip(x, bits=12):
    x = x.float()
    m, e = torch.frexp(x.abs())                                # |x| = m 2^e, m in [0.5, 1)
    m2 = m * 2
    i = torch.clamp(torch.floor((m2 - 1) * 256), 0, 255)
    y0 = _tab("recip", x.device)[i.long()]
    y1 = y0 * (2 - m2 * y0)
    out = torch.sign(x) * torch.ldexp(y1, -(e - 1))
    return round_bits(_flush(out), bits)


@_chunked
def rsqrt(x, bits=16):
    x = x.float()
    m, e = torch.frexp(x)                                      # x = m 2^e
    k = torch.floor((e - 1).float() / 2)                       # x = u 4^k, u in [1, 4)
    u = torch.ldexp(x, (-2 * k).int())
    i = torch.clamp(torch.floor((u - 1) / 3 * 256), 0, 255)
    y0 = _tab("rsqrt", x.device)[i.long()]
    y1 = y0 * (1.5 - 0.5 * u * y0 * y0)
    return round_bits(torch.ldexp(y1, (-k).int()), bits)


@_chunked
def sigmoid(x, bits=12):
    x = x.float()
    z = exp(-x.abs())
    s = recip(1 + z)
    return round_bits(torch.where(x >= 0, s, z * s), bits)


def silu(x):
    """x * sigmoid(x): one fp32 multiply on the vector unit's fma, so the result follows its rule (below 2^-126 and
    every zero -> +0). Found by the RTL test, 2026-11-21: the plain product gave -0 and kept subnormals."""
    return _flush(x.float() * sigmoid(x))


def _ln1p_small(z):
    """ln(1 + z) for z in [0, 1]: log2 of the mantissa from a 64-entry table and a quadratic."""
    u = 1 + z
    m, e = torch.frexp(u)                                      # u = m 2^e, m in [0.5, 1)
    mm, ee = m * 2, (e - 1).float()
    i = torch.clamp(torch.floor((mm - 1) * 64), 0, 63)
    r = mm - 1 - i / 64
    t = _tab("log2", z.device)
    il = i.long()
    l2 = t[0][il] + r * t[1][il] + r * r * t[2][il]
    big = (l2 + ee) * LN2
    small = z * (1 - z * (0.5 - z / 3))                       # z < 2^-6: series, keeps relative accuracy
    return torch.where(z < 2.0 ** -6, small, big)


@_chunked
def softplus(x, bits=12):
    x = x.float()
    return round_bits(torch.clamp(x, min=0) + _ln1p_small(exp(-x.abs())), bits)


def self_test(n=200_000, seed=0):
    """Max relative error of each function against float64, over wide random inputs."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    xs = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1)
    cases = {
        "exp": (lambda t: exp(t), lambda t: torch.exp(t), xs * 80),
        "recip": (lambda t: recip(t), lambda t: 1 / t, torch.exp(xs * 40) * torch.sign(xs)),
        "rsqrt": (lambda t: rsqrt(t), lambda t: 1 / torch.sqrt(t), torch.exp(xs * 40)),
        "sigmoid": (lambda t: sigmoid(t), lambda t: torch.sigmoid(t), xs * 30),
        "softplus": (lambda t: softplus(t), lambda t: torch.nn.functional.softplus(t), xs * 30),
    }
    for k, (f, ref, x) in cases.items():
        y = f(x.float()).double()
        r = ref(x)
        ok = r.abs() > 2.0 ** -120
        rel = ((y - r).abs() / r.abs())[ok]
        out[k] = (rel.max().item(), math.log2(rel.max().item()))
    return out


if __name__ == "__main__":
    for k, (e, b) in self_test().items():
        print(f"{k:9s} max relative error {e:.2e} (2^{b:.1f})")
