"""Bank clocks per vector beat in the 9B prefill chunk, from real addresses (note 34).
  python3 isa/bank_conflicts.py
"""
# Bank clocks per vector beat in the 9B prefill chunk, from real addresses.
# Every beat: the lines its lanes touch (reads of a, b, c coalesced; the write separately); single-ported banks
# serve one line per clock, so the beat needs max over banks of the lines that bank must serve.
# Mappings: 64 or 128 banks, plain (line mod NB) or skewed (line + line/NB + line/NB^2 mod NB, chip_sram SKEW=1).
# Each descriptor is measured exactly on its first few uses and its ratio applied to the rest.
import os, sys, math, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import bench_rtl_rates as B
from fsim_v1 import NP
SZ = {k: v().itemsize for k, v in NP.items()}
B.FIX.clear()
MAPS = {"64 plain": (64, 0), "64 skew": (64, 1), "64 skew+wbuf": (64, 5), "64 ideal": (64, 9),
        "128 plain": (128, 0), "128 skew": (128, 1), "128 skew+wbuf": (128, 5), "128 ideal": (128, 9)}
WIDE = ("QUANT", "DEQ", "QKV")
ROW128 = ("EXP", "SIGMOID", "SILU", "SOFTPLUS", "RSQRT", "RECIP", "RMAX", "MATCH")
SKIP = ("NOP", "DN", "DNC", "SMTILE")
SAMPLES = 3

def bank(line, nb, skew):
    if not skew:
        return line % nb
    lb = int(math.log2(nb))
    return (line + (line >> lb) + (line >> (2 * lb))) % nb

def beats(op, n, m, p):
    """Lists of (p, m, n) index arrays, one per beat, as the RTL walks them."""
    width = 128 if op in ROW128 else 512
    pk_ok = (2 <= n < 512 and n & (n - 1) == 0 and m * p > 1 and op not in ROW128 and
             (op not in WIDE or n >= 32))
    out = []
    if pk_ok:
        g = 512 // n
        j = np.arange(512); slot, col = j // n, j % n
        if m == 1:
            for p0 in range(0, p, g):
                k = slot < p - p0
                out.append((p0 + slot[k], np.zeros(k.sum(), int), col[k]))
        else:
            for pp in range(p):
                for m0 in range(0, m, g):
                    k = slot < m - m0
                    out.append((np.full(k.sum(), pp), m0 + slot[k], col[k]))
    else:
        for pp in range(p):
            for mm in range(m):
                for c0 in range(0, n, width):
                    col = np.arange(c0, min(n, c0 + width))
                    out.append((np.full(col.size, pp), np.full(col.size, mm), col))
    return out

def nbeats_of(op, n, m, p):
    width = 128 if op in ROW128 else 512
    if 2 <= n < 512 and n & (n - 1) == 0 and m * p > 1 and op not in ROW128 and (op not in WIDE or n >= 32):
        g = 512 // n
        return math.ceil(p / g) if m == 1 else p * math.ceil(m / g)
    return p * m * math.ceil(n / width)

def lines_of(o, off, idx, grouped=False):
    pp, mm, nn = idx
    if grouped:
        nn = nn // 32
    if o.sn == 0:
        nn = nn * 0
    a = o.base + off + pp * o.sp + mm * o.sm + nn * o.sn
    return a >> 6

LATB = {"EXP": 64, "SIGMOID": 118, "SILU": 126, "SOFTPLUS": 137, "RSQRT": 48, "RECIP": 24, "QUANT": 14, "QKV": 13}

def beat_cost(c):
    """Clocks beyond one per beat when beat i's reads share the clock with beat i - L's write (L: the
    command's pipeline latency, in beats at one beat per clock)."""
    d = c.d
    e, n, m, p = d.elems(c.rv[4] if d.override in ("n", "m", "p") else 0)
    if n * m * p == 0:
        return None
    bl = beats(c.op, n, m, p)
    use = bl[:64]
    L = LATB.get(c.op, 11)
    rds, wrs = [], []
    for idx in use:
        rd = set()
        for o, off, grp in ((d.a, c.rv[1], False), (d.b, c.rv[2], c.op == "DEQ"), (d.c, c.rv[3], False)):
            if o is None or (c.op in ("QUANT", "QKV") and o is d.c):
                continue
            rd.update(lines_of(o, off, idx, grp).tolist())
        wr = set()
        if d.dst is not None:
            if c.op in ("RSUM", "RSUMSQ", "RMAX"):
                wr.update(lines_of(d.dst, c.rv[0], (idx[0][:1], idx[1][:1], idx[2][:1] * 0)).tolist())
            else:
                wr.update(lines_of(d.dst, c.rv[0], idx).tolist())
        if c.op == "QUANT" and d.c is not None:
            wr.update(lines_of(d.c, c.rv[3], idx, True).tolist())
        rds.append(rd); wrs.append(wr)
    clocks = {k: len(use) for k in MAPS}
    for t in range(len(use) + L):
        rd = rds[t] if t < len(use) else set()
        wr = wrs[t - L] if 0 <= t - L < len(use) else set()
        if not rd and not wr:
            continue
        for k, (nb, sk) in MAPS.items():
            if sk == 9:
                need = math.ceil((len(rd) + len(wr)) / nb)
            elif sk == 5:
                # a write buffer: writes wait and fill banks the reads leave free, so a clock is set by its
                # reads (skewed), as long as the clock's total lines still fit the banks
                cnt = collections.Counter(bank(x, nb, 1) for x in rd)
                need = max(max(cnt.values()) if cnt else 1, math.ceil((len(rd) + len(wr)) / nb))
            else:
                cnt = collections.Counter(bank(x, nb, sk) for x in rd)
                cnt.update(bank(x, nb, sk) for x in wr)
                need = max(cnt.values())
            clocks[k] += max(need, 1) - 1
    return len(use), clocks, len(bl)

seen = collections.defaultdict(list)       # descriptor -> [(beats, clocks)]
tot_beats = 0; tot = {k: 0.0 for k in MAPS}; per_op = collections.defaultdict(lambda: collections.Counter())
MODE = None
class Probe(B.RtlSim):
    def _cost(self, c):
        super()._cost(c)
        global tot_beats
        if c.unit != "V" or c.op in SKIP:
            return
        key = id(c.d)
        s = seen[key]
        if len(s) < SAMPLES:
            r = beat_cost(c)
            if r is None:
                return
            s.append(r)
        nb_used = sum(x[0] for x in s)
        e, n, m, p = c.d.elems(c.rv[4] if c.d.override in ("n", "m", "p") else 0)
        nbeats = nbeats_of(c.op, n, m, p)
        tot_beats += nbeats
        for k in MAPS:
            ratio = sum(x[1][k] for x in s) / nb_used
            tot[k] += nbeats * ratio
            per_op[k][c.op] += nbeats * (ratio - 1)
        if MODE:
            ratio = sum(x[1][MODE] for x in s) / nb_used
            c.cycles += int(nbeats * (ratio - 1))

chip = B.SimChip(vec_lanes=512)
old = B.run_v1.Sim; B.run_v1.Sim = B.PR.Sim = Probe
pc = B.PR.chunk_time(16384, chip, fused=True, dn="lite")[0]
print(f"chunk {pc*1e3:.2f} ms at no bank cost; vector beats {tot_beats/1e6:.2f}M")
for k in MAPS:
    print(f"  {k:10s}: {tot[k]/1e6:.2f}M bank clocks ({tot[k]/tot_beats:.2f} per beat);  worst ops:",
          ", ".join(f"{o} +{v/1e3:.0f}k" for o, v in per_op[k].most_common(4)))
# prefill time with each mapping's extra clocks charged to its commands
for k in MAPS:
    MODE = k; seen.clear(); tot_beats = 0
    for kk in MAPS: tot[kk] = 0; per_op[kk].clear()
    pc = B.PR.chunk_time(16384, B.SimChip(vec_lanes=512), fused=True, dn="lite")[0]
    print(f"  with {k}: chunk {pc*1e3:.2f} ms, 32K prefill {pc*256:.2f} s", flush=True)
