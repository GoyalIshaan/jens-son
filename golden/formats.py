"""Number formats the chip might use, as the golden model applies them.

Weights are stored the way the chip stores them: a small integer code per
weight, plus one scale per group. Expanding a weight is
    value = TABLE[code] * scale[group]
which is exactly the lookup-table design in design/10_number_formats.md:
int4, FP4 (E2M1) and 3-bit formats differ only in the table and the scale rule.

Rounding is plain round-to-nearest (no calibration, no GPTQ/AWQ). Results are
therefore a pessimistic baseline: smarter quantizers usually do better.
"""
import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_TABLE = torch.tensor(sorted(set([-x for x in E2M1] + E2M1)))      # 15 distinct values

# name -> (group size, kind, levels)
WEIGHT_FORMATS = {
    # round 2 (better rounding at the same storage cost)
    "int4_g32_mse": (32, "int_mse", (-8, 7)),         # scale searched to minimise group error
    "int4_g16_fp8": (16, "int_fp8", (-8, 7)),         # int4 levels with NVFP4's scale scheme
    "nvfp4_mse": (16, "e2m1_fp8_mse", None),           # NVFP4 with scale search
    "lut4_g32": (32, "lut", None),                     # 16-level table learned per tensor (k-means)
    "lut4_g32_mse": (32, "lut_mse", None),             # learned table + scale search
    "lut4_g16_fp8": (16, "lut_fp8", None),             # learned table + NVFP4-style FP8 scale per 16
    "int8_g32": (32, "int", (-128, 127)),              # 8.5 bits; for the embedding (one row read per token)
    "lut3_g32": (32, "lut", 8),                        # 8-level learned table: 3 bits + fp16/32 = 3.5 bits
    "int3_g16_fp8": (16, "int_fp8", (-4, 3)),          # 3-bit ints, FP8 scale per 16 = 3.5 bits
    # round 1 (plain round-to-nearest)
    "int4_g32": (32, "int", (-8, 7)),
    "int4_g128": (128, "int", (-8, 7)),
    "int3_g32": (32, "int", (-4, 3)),
    "mxfp4": (32, "e2m1_pow2", None),
    "nvfp4": (16, "e2m1_fp8", None),
    # round 4: the winner with its learned table entries rounded the way the chip stores them
    "lut4_g16_fp8_t8": (16, "lut_fp8", None),          # entries as int8 (k/127): int8 x int8 multiplier
    "lut4_g16_fp8_te4m3": (16, "lut_fp8", None),       # entries as FP8 E4M3: fp8 x int8 multiplier
    "lut4_g16_fp8_t6": (16, "lut_fp8", None),          # entries as int6 (k/31): margin check
    "lut4_g16_fp8_t5": (16, "lut_fp8", None),          # entries as int5 (k/15): where does it break?
}
# how each learned table's entries are stored (default: full precision)
TABLE_ENTRY = {"lut4_g16_fp8_t8": "int8", "lut4_g16_fp8_te4m3": "e4m3",
               "lut4_g16_fp8_t6": "int6", "lut4_g16_fp8_t5": "int5"}


def round_table(table, fmt):
    """Round a learned table (values in [-1, 1]) to its stored entry format."""
    rule = TABLE_ENTRY.get(fmt)
    if rule is None:
        return table
    if rule == "e4m3":
        return table.to(torch.float8_e4m3fn).float()
    q = 2 ** (int(rule[3:]) - 1) - 1                      # int8 -> 127, int6 -> 31, int5 -> 15
    return torch.clamp(torch.round(table * q), -q, q) / q


class QTensor:
    """codes (int8 index into table) + per-group fp32 scale. Shape [N, K]."""

    def __init__(self, codes, scales, table, group, shape):
        self.codes, self.scales, self.table, self.group, self.shape = codes, scales, table, group, shape

    def dequant(self):
        vals = self.table[self.codes.long()]                                   # [N, K/g, g]
        return (vals * self.scales[..., None]).reshape(self.shape)

    def dequant_slice(self, r0, r1):
        vals = self.table[self.codes[r0:r1].long()]
        return (vals * self.scales[r0:r1][..., None]).reshape(-1, self.shape[1])

    def dequant_rows(self, rows):
        vals = self.table[self.codes[rows].long()]
        return (vals * self.scales[rows][..., None]).reshape(len(rows), self.shape[1])

    def nbytes_model(self, code_bits, scale_bits):
        n = self.shape[0] * self.shape[1]
        return n * code_bits / 8 + (n // self.group) * scale_bits / 8


def _nearest_index(x, table):
    """Index of the nearest table value (table sorted ascending)."""
    mids = (table[1:] + table[:-1]) / 2
    return torch.bucketize(x, mids)


SEARCH = torch.linspace(0.70, 1.00, 16)      # scale multipliers tried by the *_mse formats


def _sample_groups(w, g, n_groups=1 << 16):
    """Up to n_groups random groups of g weights, each scaled to max |w| = 1 (memory-light)."""
    groups = w.view(-1, g)
    if groups.shape[0] > n_groups:
        gen = torch.Generator(device=w.device).manual_seed(0)
        groups = groups[torch.randint(0, groups.shape[0], (n_groups,), device=w.device, generator=gen)]
    xs = groups.float()
    return xs / xs.abs().amax(-1, keepdim=True).clamp_min(1e-12)


def _kmeans_table(x, k=16, iters=25, sample=1 << 20):
    """Learn k levels for values already scaled to [-1, 1] (per-tensor lookup table)."""
    flat = x.reshape(-1)
    if flat.numel() > sample:
        gen = torch.Generator(device=flat.device).manual_seed(0)
        flat = flat[torch.randint(0, flat.numel(), (sample,), device=flat.device, generator=gen)]
    table = torch.quantile(flat.float(), torch.linspace(0.005, 0.995, k, device=flat.device))
    for _ in range(iters):
        idx = _nearest_index(flat, table)
        for j in range(k):
            m = idx == j
            if m.any():
                table[j] = flat[m].mean()
        table, _ = table.sort()
    return table


def _round_to_table(xn, table, kind):
    """xn: values divided by their group scale. Returns (index, rounded value)."""
    if kind.startswith("int"):
        lo = table[0]
        q = torch.clamp(torch.round(xn), table[0], table[-1])
        return (q - lo).to(torch.int16), q
    idx = _nearest_index(xn.clamp(table[0], table[-1]), table)
    return idx.to(torch.int16), table[idx.long()]


@torch.no_grad()
def quantize_weight(w, fmt, chunk_rows=4096):
    g, kind, levels = WEIGHT_FORMATS[fmt]
    N, K = w.shape
    assert K % g == 0, (fmt, w.shape)
    dev = w.device
    n_levels = (levels[1] - levels[0] + 1) if kind.startswith("int") else 16
    codes = torch.empty(N, K // g, g, dtype=torch.int8 if n_levels <= 128 else torch.int16, device=dev)
    scales = torch.empty(N, K // g, dtype=torch.float32, device=dev)
    search = kind.endswith("_mse")
    base = kind[:-4] if search else kind

    if base.startswith("int"):
        lo, hi = levels
        table = torch.arange(lo, hi + 1, dtype=torch.float32, device=dev)
        top = float(hi)                                  # group max maps to +7
    elif base.startswith("lut"):
        table = round_table(_kmeans_table(_sample_groups(w, g), k=levels or 16), fmt)
        top = 1.0                                        # group max maps to 1.0 (table spans [-1, 1])
    else:
        table = E2M1_TABLE.to(dev)
        top = 6.0
    gmax = w.float().abs().max() / (448.0 * top) if base.endswith("_fp8") else None
    mults = SEARCH.to(dev) if search else torch.ones(1, device=dev)

    for r in range(0, N, chunk_rows):
        x = w[r:r + chunk_rows].float().view(-1, K // g, g)
        amax = x.abs().amax(-1).clamp_min(1e-12)
        best_err, best_idx, best_s = None, None, None
        for m in mults:
            if base == "e2m1_pow2":
                s = torch.exp2(torch.floor(torch.log2(amax)) - 2)
            elif base.endswith("_fp8"):
                s8 = (amax * m / top / gmax).clamp(max=448.0).to(torch.float8_e4m3fn).float()
                s = (s8 * gmax).clamp_min(1e-12)
            else:
                s = (amax * m / top).half().float()      # fp16 scale, as stored
            idx, q = _round_to_table(x / s[..., None], table, base)
            if not search:
                best_idx, best_s = idx, s
                break
            err = (q * s[..., None] - x).pow(2).sum(-1)
            if best_err is None:
                best_err, best_idx, best_s = err, idx, s
            else:
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_idx = torch.where(better[..., None], idx, best_idx)
                best_s = torch.where(better, s, best_s)
        codes[r:r + chunk_rows] = best_idx
        scales[r:r + chunk_rows] = best_s
    qt = QTensor(codes, scales, table, g, (N, K))
    qt.gmax = gmax                                       # per-tensor fp32 scale (fp8 formats), for golden v2
    return qt


def _group_round(x, g, qmax):
    """Symmetric integer rounding in groups of g along the last dim, back to float."""
    shp = x.shape
    y = x.reshape(*shp[:-1], shp[-1] // g, g)
    s = (y.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    return (torch.clamp(torch.round(y / s), -qmax - 1, qmax) * s).reshape(shp)


def _bf16_round_up(s):
    """Smallest bf16 value >= s (s > 0, fp32): round the fp32 bit pattern up to the next multiple of 2^16."""
    b = s.contiguous().view(torch.int32)
    up = torch.where((b & 0xFFFF) != 0, (b | 0xFFFF) + 1, b)
    return up.view(torch.float32)


def _act_scaled(x, g, scale_rule):
    """int8 activations (values -127..127) in groups of g, with the scale's precision set by scale_rule:
    - "bf16"     scale = amax/127 rounded to the nearest bf16 (8-bit mantissa); values past 127 are clamped
    - "bf16_up"  scale = amax/127 rounded up to a bf16, so nothing is clamped
    - "p2_up"    scale = 2^ceil(log2(amax/127)), a power of two, so nothing is clamped
    - "mx"       OCP MX style: scale = 2^(floor(log2 amax) - 6), elements clamped to +/-127
    Returns the dequantized value (int8 * scale), as the chip would see it."""
    shp = x.shape
    y = x.reshape(*shp[:-1], shp[-1] // g, g)
    amax = y.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if scale_rule == "bf16":
        s = (amax / 127).to(torch.bfloat16).float()
    elif scale_rule == "bf16_up":
        s = _bf16_round_up(amax / 127)
    elif scale_rule == "p2_up":
        s = torch.exp2(torch.ceil(torch.log2(amax / 127)))
    elif scale_rule == "mx":
        s = torch.exp2(torch.floor(torch.log2(amax)) - 6)
    else:
        raise ValueError(scale_rule)
    return (torch.clamp(torch.round(y / s), -127, 127) * s).reshape(shp)


# The chip's activation format (D-019, 2026-11-20): int8, power-of-two scale per 32, rounded up.
CHIP_ACT = "int8_g32_p2"

ACT_SCALED = {"int8_g32_bf16s": "bf16", "int8_g32_bf16up": "bf16_up", "int8_g32_p2": "p2_up", "int8_g32_mx": "mx"}


def quant_act(x, fmt):
    if fmt is None:
        return x
    if fmt == "int8_g32":
        return _group_round(x, 32, 127)
    if fmt in ACT_SCALED:
        return _act_scaled(x, 32, ACT_SCALED[fmt])
    if fmt == "int8_g128":
        return _group_round(x, 128, 127)
    if fmt == "int8_tok":                                  # one scale for the whole vector
        return _group_round(x, x.shape[-1], 127)
    if fmt == "fp8_g32":
        return _fp8_group(x, 32)
    raise ValueError(fmt)


def quant_kv(x, fmt):
    if fmt is None:
        return x
    if fmt == "int8_g64":
        return _group_round(x, 64, 127)
    if fmt == "int4_g32":
        return _group_round(x, 32, 7)
    if fmt == "int4_g64":
        return _group_round(x, 64, 7)
    if fmt == "int8_head":                                 # one scale per 256-value head vector
        return _group_round(x, x.shape[-1], 127)
    if fmt == "int8_head_chip":                            # the chip's QKV command, bit-exact (golden v2, D-024):
        amax = x.abs().amax(-1, keepdim=True)              # s = fp16(amax / 127); q = int8(fp32(x * fp32(1 / s)))
        s = (amax / 127).half().float()
        r = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
        return torch.clamp(torch.round(x * r), -127, 127) * s
    if fmt == "fp8_head":
        return _fp8_group(x, x.shape[-1])
    raise ValueError(fmt)


# ---------------------------------------------------------------- helpers for GPTQ
class Spec:
    """Everything needed to quantize one column at a time with a format."""

    def __init__(self, w, fmt):
        g, kind, levels = WEIGHT_FORMATS[fmt]
        self.g, self.fmt = g, fmt
        base = kind[:-4] if kind.endswith("_mse") else kind
        self.base = base
        dev = w.device
        if base.startswith("int"):
            lo, hi = levels
            self.table = torch.arange(lo, hi + 1, dtype=torch.float32, device=dev)
            self.top = float(hi)
        elif base.startswith("lut"):
            self.table = round_table(_kmeans_table(_sample_groups(w, g), k=levels or 16), fmt)
            self.top = 1.0
        else:
            self.table = E2M1_TABLE.to(dev)
            self.top = 6.0
        self.gmax = (w.abs().max().float() / (448.0 * self.top)) if base.endswith("_fp8") else None

    def scale(self, xg):
        """xg [N, g] -> per-row group scale [N], using the format's scale rule."""
        amax = xg.abs().amax(-1).clamp_min(1e-12)
        if self.base == "e2m1_pow2":
            return torch.exp2(torch.floor(torch.log2(amax)) - 2)
        if self.base.endswith("_fp8"):
            s8 = (amax / self.top / self.gmax).clamp(max=448.0).to(torch.float8_e4m3fn).float()
            return (s8 * self.gmax).clamp_min(1e-12)
        return (amax / self.top).half().float()

    def round(self, xn):
        return _round_to_table(xn, self.table, self.base)


def _fp8_group(x, g):
    shp = x.shape
    y = x.reshape(*shp[:-1], shp[-1] // g, g)
    s = (y.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return ((y / s).to(torch.float8_e4m3fn).float() * s).reshape(shp)
