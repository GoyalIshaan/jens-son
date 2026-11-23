"""9B timing with every vector command at the rate its RTL actually achieves (rtl/vec, note 32 part 4),
against the timing model's architectural rates. The SMTILE overlap rate assumes ready SRAM ports. Run: python3 isa/bench_rtl_rates.py

Per-command RTL costs in cycles (512-lane bank, 8-cycle FMA, rows launched one at a time):
  box ADD/SUB/MUL/FMA   beats + 8 (streaming, one beat per clock)        ROPE 2 x beats + 8   CONV4 4 x beats + 8
  CONV4 with a saved window (decode), COPY/MAX/CAUSAL: the one-beat-at-a-time path
  row ops (128 lanes), streamed across rows and planes (measured, test_vec_unit.py --rows-only):
      unary  rows x sub-beats per row + 8 + function latency + 3   (128 exp, 64 sigmoid/SiLU, 2 others per clock;
      a partial beat takes only the sub-beats its elements need)
      QUANT/DEQ rows x beats + 11
      QKV rows x (2 x beats + 1) + 11 (the next row's scan overlaps this row's drain), RMAX rows x (beats + 1) + 1,
      MATCH is one row by the ISA: beats + 2
  RSUM/RSUMSQ: CTX = 4 rows in flight on a 9-clock calendar (measured, test_vec_unit.py --red-only):
      ceil(rows / 4) x 9 x (ceil(n / 512) + 9) + RED_TAIL
  SMTILE: per row       199 at 1,024 positions, minus 2 per omitted 128-position beat
                       on the serial fallback (196 clocks at 1,024, 184 at 256 without stalls)
  SMTILE broadcast:    179 + 8 x (rows - 1) at 256, 211 + 17 x (rows - 1) at 1,024
                       at the decoded port, with SRAM writes immediately visible
  DN: measured controller cycles (sim_v1.DN_CYCLES)
"""
from itertools import combinations
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isa_v1 import SRAM_BYTES  # noqa: E402
import prefill_run as PR  # noqa: E402
import run_v1  # noqa: E402
import sim_v1  # noqa: E402
from sim_v1 import Sim, SimChip  # noqa: E402

LAT_F = dict(EXP=64, SIGMOID=118, SILU=126, SOFTPLUS=137, RSQRT=48, RECIP=24)   # 1 GHz: input-registered FMAs, glue stages
S_F = dict(EXP=1, SIGMOID=2, SILU=2, SOFTPLUS=64, RSQRT=64, RECIP=64)
SMTILE_ROW_1024 = 199
SMTILE_OVERLAP = {256: (165, 3), 1024: (214, 17)}   # 256: vec_smtile_p3 (measured 258 clocks for 32 rows)
RED_CTX = 4
RED_TAIL = 8          # (the four-context R1, before packing): 158 clocks for 1 row of 4,096, 620 for 16, 275 for 9 of 512
QL = 3                # the pipelined activation quantizer (act_quant32_pipe)
PACKED = "unpacked" not in os.environ.get("RTL_FIXED", "")


def groups(n, m, p):
    """Beats of short power-of-two rows packed several to a 512-lane beat (vec_box_walk); else rows x beats."""
    if PACKED and 2 <= n < 512 and n & (n - 1) == 0 and m * p > 1:
        g = 512 // n
        return math.ceil(p / g) if m == 1 else p * math.ceil(m / g), True
    return m * p * math.ceil(n / 512), False


def r1_cycles(n, m, p):
    """Two-phase R1 (vec_reduce_rows): A clocks read and accumulate, B clocks run the spatial tree.
    Measured: 4,096 x 1 in 142, x 16 in 340, 512 x 9 in 102, 64 planes of 128 in 100 (decoded boundary)."""
    beats, packed = groups(n, m, p)
    lout = int(math.log2(n)) if packed else 9
    if packed or n <= 512:
        last = 2 * (beats - 1)
    else:
        nb, rows = math.ceil(n / 512), m * p
        last = 2 * (((math.ceil(rows / 4) - 1) * nb + nb - 1) * 4 + (rows - 1) % 4)
    return last + 9 + 8 * lout + 5
FIX = set(os.environ.get("RTL_FIXED", "").split(","))      # command classes treated as already at model rates


def smtile_overlap_cycles(c, n, rows, planes):
    """Apply the same conservative broadcast layout checks as vec_unit."""
    if n not in SMTILE_OVERLAP or rows <= 1 or planes != 1:
        return None
    d = c.d
    r0, r1, r2, r3, _ = c.rv
    if (d.dst.dtype, d.a.dtype, d.b.dtype, d.c.dtype) != ("i8", "f32", "f16", "f32"):
        return None
    if (d.dst.sn, d.a.sn, d.b.sn, d.b.sm, d.b.sp, d.c.sm) != (1, 4, 2, 0, 0, 4):
        return None
    par = dict(d.par)
    exp_offset, exp_stride = par["e"]
    if d.dst.sm < n or d.a.sm < 4 * n or exp_stride < n // 32:
        return None

    def extent(base, size):
        return base, base + size

    def disjoint(a, b):
        return a[1] <= b[0] or b[1] <= a[0]

    code_base, score_base, scale_base, state_base = (
        d.dst.base + r0, d.a.base + r1, d.b.base + r2, d.c.base + r3,
    )
    inputs = [
        extent(score_base, (rows - 1) * d.a.sm + 4 * n),
        extent(scale_base, 2 * n),
        extent(scale_base + par["vs"], 2 * n),
    ]
    outputs = [
        extent(code_base, (rows - 1) * d.dst.sm + n),
        extent(code_base + exp_offset, (rows - 1) * exp_stride + n // 32),
        *(extent(state_base + bank * par["mlb"], 4 * rows) for bank in (0, 4, 5)),
    ]
    if any(low < 0 or high > SRAM_BYTES for low, high in inputs + outputs):
        return None
    if any(not disjoint(a, b) for a, b in combinations(outputs, 2)):
        return None
    if any(not disjoint(a, b) for a in outputs for b in inputs):
        return None
    first_row, interval = SMTILE_OVERLAP[n]
    return first_row + (rows - 1) * interval


def rtl_cycles(c):
    d = c.d
    e, n, m, p = d.elems(c.rv[4] if d.override in ("n", "m", "p") else 0)
    rows = m * p
    op = c.op
    b512 = math.ceil(n / 512) * rows
    b128 = math.ceil(n / 128)
    par = dict(d.par)
    beats = groups(n, m, p)[0]
    if op in ("ADD", "SUB", "MUL", "FMA"):
        return beats + 8 + 3                                  # measured: 16 packed beats in 27
    if op == "ROPE":
        return 2 * beats + 8 + 3
    if op == "CONV4":
        return (38 * b512 + 8) if ("window" in par and "conv_window" not in FIX) else 4 * beats + 8 + 3
    if op in ("COPY", "MAX", "CAUSAL"):
        return beats + 4                                      # the pass path: one beat per clock
    serial = "serial_rows" in FIX                             # the pre-streaming RTL, one row per launch
    rs = not serial                                           # unary, QUANT and DEQ stream rows as built
    scan_rs = rs
    if op in LAT_F:
        units = 128 // S_F[op]
        per_row = (n // 128) * S_F[op] + math.ceil((n % 128) / units)     # a partial beat takes only its sub-beats
        return (rows * per_row + 8 + LAT_F[op] + 3 if rs else rows * (per_row + 8 + LAT_F[op] + 3)) + 8
    qn = QL if op == "QUANT" else 0
    if op in ("QUANT", "DEQ"):
        if "narrow_q" not in FIX:                             # vec_rows_wide: 512 lanes, short rows packed
            return groups(n, m, p)[0] + 11 + qn               # measured: 16 beats in 27 (DEQ), 30 (QUANT)
        return (rows * b128 + 11 + QL if rs else rows * (b128 + 11 + QL)) + 8
    if op == "QKV":
        packed = n in (32, 64, 128, 256) and rows > 1
        if packed and "narrow_q" not in FIX:                  # scan beat, 5 clocks, stream beat (vec_rows_wide)
            return 2 * groups(n, m, p)[0] + 15                # measured: 16 beats in 47
        # 128 lanes: scan, 3 clocks of scale (vec_kv_scale_pipe), stream; the next scan overlaps the drain
        return (rows * (2 * b128 + 4) + 11 if scan_rs else rows * (2 * b128 + 15)) + 8
    if op in ("RMAX", "MATCH"):
        return (rows * (b128 + 1) + 1 if scan_rs else rows * (b128 + 2)) + 8
    if op in ("RSUM", "RSUMSQ"):
        if "ctx_reduce" not in FIX:
            return r1_cycles(n, m, p)
        if "reduce" in FIX:                                    # bank-limited ideal: one row per clock
            return 9 * math.ceil(n / 512) + 81 + rows + 8
        ctx = 1 if "serial_reduce" in FIX else 2 if "red2" in FIX else 8 if "red8" in FIX else RED_CTX
        return math.ceil(rows / ctx) * 9 * (math.ceil(n / 512) + 9) + RED_TAIL
    if op == "SMTILE":
        if "smtile_model" in FIX:
            return None
        if "smtile" in FIX:
            return rows * (b128 + 2) + 200
        if "smtile_serial" not in FIX:
            cycles = smtile_overlap_cycles(c, n, m, p)
            if cycles is not None:
                return cycles
        # Both passes issue one 128-position beat per cycle. Keep the same
        # three-clock allowance above the stall-free measurement at each width.
        return rows * (SMTILE_ROW_1024 - 2 * (8 - b128))
    return None


class RtlSim(Sim):
    def _cost(self, c):
        super()._cost(c)
        if c.unit == "V" and c.op not in ("NOP", "DN", "DNC"):
            cy = rtl_cycles(c)
            if cy is not None:
                c.cycles = cy


def measure(sim_cls):
    old = run_v1.Sim
    run_v1.Sim, PR.Sim = sim_cls, sim_cls
    try:
        chip = SimChip(vec_lanes=512)
        d, _ = run_v1.timed("decode", 32768, a=3, chip=chip, fused=True, dn_heads=4)
        s, _ = run_v1.timed("spec", 32768, a=3, chip=chip, fused=True, dn_heads=4)
        pc = PR.chunk_time(16384, chip, fused=True, dn="lite")[0]
    finally:
        run_v1.Sim, PR.Sim = old, old
    return dict(decode_tok_s=round(1e9 / d, 1), spec_a3_tok_s=round(4e9 / s, 1),
                prefill_chunk_ms=round(pc * 1e3, 2), prefill_32k_s=round(pc * 256, 1))


if __name__ == "__main__":
    only = os.environ.get("ONLY")
    out = {}
    for label, fixes in (("model rates", ""), ("RTL as built", ",".join(sorted(FIX))),
                         ("pre-overlap RTL", "smtile_serial,serial_rows,serial_reduce"),
                         ("SMTILE overlap only", "serial_rows,serial_reduce"),
                         ("+ row streaming, serial R1", "serial_reduce"),
                         ("as built, 2 R1 contexts (4 KiB less)", "red2"),
                         ("as built, 8 R1 contexts (8 KiB more)", "red8"),
                         # what is left, cumulative on top of the RTL as built
                         ("as built, QUANT/QKV/DEQ at 128 lanes", "narrow_q"),
                         ("+ CONV4 window streamed", "conv_window"),
                         ("+ R1 one row per clock", "conv_window,reduce"),
                         ("+ SMTILE at model rates", "conv_window,reduce,smtile_model")):
        if only and label not in only.split("|"):
            continue
        FIX.clear(); FIX.update(fixes.split(","))
        out[label] = measure(Sim if label == "model rates" else RtlSim)
        print(f"{label:45s} {out[label]}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtl", "vec", "results", "rtl_rates_timing.json")
    if only and os.path.exists(path):
        old = json.load(open(path)); old.update(out); out = old
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtl", "vec", "results",
                                     "rtl_rates_timing.json"), "w"), indent=1)
