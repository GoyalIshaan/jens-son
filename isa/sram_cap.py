"""Prefill and decode with every vector command held to an SRAM bandwidth (note 34).
  python3 isa/sram_cap.py
"""
# Prefill and decode at the RTL's as-built vector rates, with each vector command held to the SRAM's
# bandwidth: cycles = max(RTL cycles, bytes / (cap x efficiency)). Matrix-unit commands are not capped
# (their 1-2 KB per clock is listed separately).
import os, sys, math, collections
sys.path.insert(0, os.getcwd())
import bench_rtl_rates as B
import sram_traffic as T
B.FIX.clear()
CAP = None; EFF = 1.0
heavy = collections.Counter(); extra = collections.Counter()
class Capped(B.RtlSim):
    def _cost(self, c):
        super()._cost(c)
        if CAP and c.unit == "V" and c.op not in ("NOP", "DN", "DNC"):
            rd, wr = T.v_bytes(c)
            need = math.ceil((rd + wr) / (CAP * EFF)) + 11
            if need > c.cycles:
                extra[c.op] += need - c.cycles; c.cycles = need
for cap, eff in ((None, 1), (4096, 1), (4096, 0.8), (8192, 1), (8192, 0.8), (3072, 0.8), (2048, 0.8)):
    CAP, EFF = cap, eff; extra.clear()
    r = B.measure(Capped)
    top = ", ".join(f"{k} +{v/1e3:.0f}k" for k, v in extra.most_common(5))
    print(f"cap {cap or 'none'} eff {eff}: {r}  | extra clocks: {top}", flush=True)
