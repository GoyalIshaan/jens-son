"""SRAM traffic per unit for the 9B programs (sizing the banked SRAM, note 33).

For every command: bytes read and written in SRAM, and the command's cycles in the timing model, so
bytes / cycles is the bandwidth that command needs while it runs. Reported: totals, the time-weighted average
over the run, and the commands that need the most bytes per clock.
  python3 isa/sram_traffic.py
"""
import math
from collections import defaultdict

import isa_v1 as I
import prefill_run as PR
import qwen_prefill_v1 as QP
import qwen_v1 as Q
from fsim_v1 import NP
from sim_v1 import Sim, SimChip

SZ = {k: v().itemsize for k, v in NP.items()}


def v_bytes(c):
    d = c.d
    e, n, m, p = d.elems(c.rv[4] if d.override in ("n", "m", "p") else 0)
    rd = sum(e * SZ[o.dtype] for o in (d.a, d.b, d.c) if o is not None and o.sn != 0)
    rd += sum(m * p * SZ[o.dtype] for o in (d.a, d.b, d.c) if o is not None and o.sn == 0)
    wr = (e if c.op not in ("RSUM", "RSUMSQ", "RMAX") else m * p) * SZ[d.dst.dtype] if d.dst is not None else 0
    if c.op == "DN":
        heads = dict(d.par).get("heads", 1)
        rd, wr = heads * 2 * 128 * 128 * 4, heads * 2 * 128 * 128 * 4      # state read twice, written twice
    return rd, wr


def m_bytes(c):
    d = c.d
    ov = c.rv[4]
    n_in = ov if (ov and d.override == "n_in") else d.n_in
    n_out = ov if (ov and d.override == "n_out") else d.n_out
    tiles = 1 if d.wfmt == "int8" else n_out // 64
    rd = tiles * d.rows * (n_in + n_in // 32)                  # activations reloaded per 64-output tile
    if c.op == "MMS":
        rd += n_in * n_out
    wr = d.rows * n_out * 4 if d.epi != "argmax" else d.rows * 4
    if d.epi == "acc":
        rd += wr
    return rd, wr


class Traffic(Sim):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.t = defaultdict(lambda: [0, 0, 0.0])
        self.peak = []

    def _cost(self, c):
        super()._cost(c)
        if c.op in ("NOP", "TAB"):
            return
        if c.unit == "V":
            rd, wr = v_bytes(c)
        elif c.unit == "M":
            rd, wr = m_bytes(c)
        elif c.unit == "R":
            rd, wr = 0, c.bytes
        elif c.unit == "X":
            rd, wr = c.bytes, 0
        elif c.unit == "L":
            rd, wr = c.bytes, c.bytes
        else:
            return
        cyc = c.cycles if c.unit in ("V", "L") else (c.fifo_bytes / max(c.beat_bytes, 1) * max(c.hold, 1)
                                                      if c.unit == "M" and c.op == "MM" else c.cycles or 1)
        cyc = max(cyc, 1)
        t = self.t[c.unit]
        t[0] += rd; t[1] += wr; t[2] += cyc
        self.peak.append(((rd + wr) / cyc, c.unit, c.op, c.note, rd, wr, cyc))


def report(name, prog, regs):
    s = Traffic(prog, prog.assemble(), SimChip(vec_lanes=512), regs=regs, oracle=lambda a: 1000)
    st = s.run()
    T = st["time_ns"]
    print(f"\n{name}: {T / 1e3:.0f} us")
    tot = 0
    for u, (rd, wr, cyc) in sorted(s.t.items()):
        tot += rd + wr
        print(f"  {u}: read {rd / 1e6:8.2f} MB, write {wr / 1e6:8.2f} MB, average over the run {(rd + wr) / T:7.0f} B/clock")
    print(f"  all units: {tot / T:.0f} B/clock on average")
    seen = set()
    print("  highest demand while running (bytes per clock):")
    for bpc, u, op, note, rd, wr, cyc in sorted(s.peak, reverse=True):
        if (u, op, note) in seen:
            continue
        seen.add((u, op, note))
        print(f"    {bpc:7.0f}  {u} {op:8s} {note[:40]:40s} read {rd:>9,} write {wr:>9,} in {cyc:,.0f} clocks")
        if len(seen) >= 10:
            break


if __name__ == "__main__":
    prog, _ = Q.build("decode", steps=1, lmax=32768, cfg=Q.Q9B, fused=True)
    report("9B decode, 32K context", prog, {Q.X_L: 32000})
    prog, _ = QP.build_prefill(chunks=1, lmax=40_000, fused=True, dn="lite")
    report("9B prefill chunk at 16K (lite DeltaNet)", prog, {1: 16384})
