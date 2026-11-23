"""Per-command timing: the RTL's trace (results/<test>/trace.txt) against the timing simulator, unit by unit.

  python3 rtl/chip/compare_timing.py tiny|tinyprefill|smoke [tokens|chunks]

Commands of each unit run in order in both, so the i-th command of a unit is the same command. Prints each
unit's busy clocks in both, and the commands where the RTL spends the most clocks beyond the model.
"""
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_chip as R  # noqa: E402
from sim_v1 import Sim, SimChip  # noqa: E402

UNITS = ["W", "R", "X", "L", "M", "V"]


def rtl_trace(path):
    cmds = defaultdict(list)
    open_ = defaultdict(list)                  # W and R can have several commands in flight; dones are in order
    for line in open(path):
        f = line.split()
        t, kind, u = int(f[0]), f[1], int(f[2])
        if kind == "S":
            open_[u].append(t)
        elif open_[u]:
            cmds[UNITS[u]].append(t - open_[u].pop(0))
    return cmds


def model_class():
    """MODEL=rtl: the timing model with the vector unit's measured RTL rates (isa/bench_rtl_rates.py) and the
    D-029 matrix unit's load and epilogue clocks (isa/mxu_overhead.py, overlap256p). Default: the plain model."""
    if os.environ.get("MODEL") != "rtl":
        return Sim
    import bench_rtl_rates as B
    import mxu_overhead as MO
    B.FIX.clear()

    class Both(MO.Timed, B.RtlSim):
        design, E, packed = "overlap", 256, True
    return Both


def model(prog, regs):
    sim = model_class()(prog, prog.assemble(), SimChip(vec_lanes=512), regs=regs)
    log = defaultdict(list)
    orig_try, orig_done, orig_mend, orig_dend = sim._try_start, sim._done, sim._m_end, sim._dma_end

    def try_start(u):
        before = sim.cur[u]
        orig_try(u)
        c = sim.cur[u]
        if c is not None and c is not before:
            log[u].append([c, sim.now, None])

    def finish(u):
        if log[u] and log[u][-1][2] is None:
            log[u][-1][2] = sim.now
    sim._try_start = try_start
    sim._done = lambda u: (finish(u), orig_done(u))
    sim._m_end = lambda: (finish("M"), orig_mend())
    sim._dma_end = lambda u: (finish(u), orig_dend(u))
    st = sim.run()
    return log, st["time_ns"]


def main():
    test = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    prog, sram, dram, regs, info = (R.tiny(tokens) if test == "tiny" else R.tinyprefill(tokens)
                                    if test == "tinyprefill" else R.smoke())
    log, t_model = model(prog, regs)
    rtl = rtl_trace(os.path.join(HERE, "results", test + os.environ.get("TAG", ""), "trace.txt"))
    print(f"{test}: model {t_model:,.0f} clocks end to end")
    rows = []
    for u in UNITS:
        m = [(c, (e if e is not None else s) - s) for c, s, e in log[u]]
        r = rtl[u]
        n = min(len(m), len(r))
        print(f"  {u}: {len(r)} commands in the RTL, {len(m)} in the model; busy RTL {sum(r):,} vs model "
              f"{sum(d for _, d in m):,.0f}")
        for i in range(n):
            c, d = m[i]
            rows.append((r[i] - d, u, c.op, c.note, r[i], d))
    agg = defaultdict(lambda: [0, 0, 0])
    for ex, u, op, note, r, d in rows:
        a = agg[(u, op)]
        a[0] += 1
        a[1] += r
        a[2] += d
    print("  by opcode (count, RTL clocks, model clocks), largest excess first:")
    for (u, op), (k, r, d) in sorted(agg.items(), key=lambda kv: kv[1][2] - kv[1][1])[:14]:
        print(f"    {u} {op:8s} {k:6d}  RTL {r:9,}  model {d:9,.0f}  excess {r - d:9,.0f}")
    print("  largest excess (RTL - model clocks):")
    for ex, u, op, note, r, d in sorted(rows, reverse=True)[:15]:
        print(f"    {ex:8.0f}  {u} {op:7s} {note[:44]:44s} RTL {r:7,} model {d:9,.0f}")


if __name__ == "__main__":
    main()
