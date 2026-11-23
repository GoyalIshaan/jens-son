"""Build the Qwen3.5-9B program in ISA v1, check it, and time it.  Run: python3 isa/run_v1.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isa_v1 as I  # noqa: E402
import qwen_v1 as Q  # noqa: E402
from sim_v1 import Sim, SimChip  # noqa: E402


def static_report(mode, k=3):
    p, info = Q.build(mode, k=k, steps=1)
    b = p.assemble()
    words = I.encode(b)
    used = sum(x.used() for x in b)
    per = {s: sum(getattr(x, s) is not None for x in b) for s in "SDMV"}
    print(f"\n== {mode}: static program ==")
    print(f"  bundles {len(b):,} of {I.IMEM_BUNDLES:,} ({len(b) * 32 / 1024:.0f} KB), descriptors {len(p.descs)} "
          f"of {I.DMEM_DESCS}, semaphores {len(info['sems'].ids)} of {I.N_SEMS}")
    print(f"  slot use: {used / (4 * len(b)):.0%} overall; S {per['S']:,}  D {per['D']:,}  M {per['M']:,}  V {per['V']:,}")
    sa = info["sram"]
    print(f"  SRAM used {sa.total() / 2**20:.2f} MB of {I.SRAM_BYTES / 2**20:.2f} MB")
    print(f"  DRAM laid out {info['dram'].total() / 1e9:.2f} GB (lmax {info['lmax']:,})")
    assert len(b) <= I.IMEM_BUNDLES and len(p.descs) <= I.DMEM_DESCS and sa.total() <= I.SRAM_BYTES
    return p, b, info


def timed(mode, L, a=3, k=3, chip=SimChip(), steps=2, fused=False, dn_heads=4):
    """Steady-state time of one step: (time for `steps` steps - time for 1 step) / (steps - 1)."""
    out = []
    for n in (1, steps):
        p, info = Q.build(mode, k=k, steps=n, lmax=max(40_000, L + 64), fused=fused, dn_heads=dn_heads)
        b = p.assemble()
        acc_addr = None
        for w, addr, size in info["sram"].items:
            if w.startswith("argmax ids"):
                acc_addr = addr + 128
        oracle = lambda addr: a if addr == acc_addr else 1000
        regs = {Q.X_L: L, Q.X_A: a, Q.X_NC: (a + 1) if mode == "spec" else 1}
        sim = Sim(p, b, chip, regs=regs, oracle=oracle)
        sim.sem_names = {v: k for k, v in info["sems"].ids.items()}
        st = sim.run()
        out.append(st)
    dt = (out[1]["time_ns"] - out[0]["time_ns"]) / (steps - 1)
    return dt, out[1]


def acceptance_mix():
    """P(a drafts accepted), a = 0..3, from the measured windowed acceptance (perf model, note 21)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "design", "calc"))
    import perf_model as PM
    acc = PM.window_acc(PM.DRAFT_WINDOW)
    p, run = [], 1.0
    for j in range(3):
        p.append(run * (1 - acc[j]))
        run *= acc[j]
    return p + [run], PM


def spec_tok_s(L, chip, fused=False):
    p, _ = acceptance_mix()
    dts = [timed("spec", L, a=a, chip=chip, fused=fused)[0] for a in range(4)]
    return sum(pp * (a + 1) for a, pp in enumerate(p)) * 1e9 / sum(pp * d for pp, d in zip(p, dts))


def main():
    from dataclasses import replace
    t0 = time.time()
    static_report("decode")
    _, _, info = static_report("spec")
    print("\n== SRAM layout, speculative program (T = 4) ==")
    print(info["sram"].report())
    p, PM = acceptance_mix()
    pc = replace(PM.Chip(), rows=64, exp_per_cycle=128)
    print(f"\n== timing (simulated, 1 GHz, 512 GB/s x 0.92; acceptance mix a=0..3 {[round(x, 3) for x in p]}) ==")
    print("  weight FIFO size vs DRAM latency, plain decode at 4K (tok/s):")
    for fifo in (96, 128, 192, 256):
        row = [f"{lat} ns {1e9 / timed('decode', 4096, chip=SimChip(fifo=fifo * 1024, latency_ns=lat))[0]:6.1f}"
               for lat in (100, 150, 200, 300)]
        print(f"    {fifo:3d} KB: " + " | ".join(row))
    for lanes in (256, 512):
        ch = SimChip(vec_lanes=lanes)
        for L in (4096, 32768):
            dec = 1e9 / timed("decode", L, chip=ch)[0]
            sp = spec_tok_s(L, ch)
            print(f"  {lanes} vector lanes, {L:6d} context: decode {dec:6.1f} tok/s (perf model "
                  f"{PM.decode(L, pc)['tok_s']:6.1f}), speculative k=3 {sp:6.1f} tok/s (perf model "
                  f"{PM.spec(L, 3, pc, PM.window_acc(4096), window=4096)['tok_s']:6.1f})")
    ch = SimChip(vec_lanes=512)
    dt, st = timed("spec", 32768, a=3, chip=ch)
    print("\n  speculative step at 32K (a = 3, 512 lanes), two steps simulated:")
    print("    DRAM idle (us per step) by what was running:",
          {k: round(v / 2e3) for k, v in sorted(st["idle_dram"].items(), key=lambda x: -x[1])})
    print("    vector busy (us per step) by op:",
          {k[2:]: round(v / 2e3) for k, v in sorted(st["notes"].items(), key=lambda x: -x[1]) if k[0] == "V" and v > 2e4})
    print(f"\n(simulation took {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
