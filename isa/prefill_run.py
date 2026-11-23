"""Time the prefill program (one 128-token chunk at a given start position) and integrate over a prompt."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_prefill_v1 as QP  # noqa: E402
from sim_v1 import Sim, SimChip  # noqa: E402


def chunk_time(L, chip=SimChip(vec_lanes=512), fused=False, dn="seq"):
    ts = []
    for n in (1, 2):
        p, info = QP.build_prefill(chunks=n, lmax=max(40_000, L + 512), fused=fused, dn=dn)
        b = p.assemble()
        sim = Sim(p, b, chip, regs={1: L}, oracle=lambda a: 1000)
        sim.sem_names = {v: k for k, v in info["sems"].ids.items()}
        ts.append(sim.run())
    st = ts[1]
    return (ts[1]["time_ns"] - ts[0]["time_ns"]) / 1e9, st, ts[0]


def prompt_time(n_tokens, chip=SimChip(vec_lanes=512), fused=False, points=5, dn="seq"):
    """Chunk time is close to linear in its start position: simulate a few starts and integrate."""
    starts = [round(i * (n_tokens - 128) / (points - 1) / 128) * 128 for i in range(points)]
    tt = [chunk_time(L, chip, fused, dn)[0] for L in starts]
    n_chunks = n_tokens // 128
    total = 0.0
    for c in range(n_chunks):
        L = c * 128
        for i in range(points - 1):
            if starts[i] <= L <= starts[i + 1]:
                f = (L - starts[i]) / max(1, starts[i + 1] - starts[i])
                total += tt[i] + f * (tt[i + 1] - tt[i])
                break
    return total, list(zip(starts, tt))


if __name__ == "__main__":
    t0 = time.time()
    for L in (0, 16384):
        dt, st, s0 = chunk_time(L)
        busy = {k: round((st["busy"][k] - s0["busy"][k]) / 1e3) for k in st["busy"]}
        idle = {k: round(v / 2e3) for k, v in sorted(st["idle_dram"].items(), key=lambda x: -x[1])}
        print(f"chunk at {L}: {dt * 1e3:.2f} ms; unit busy us {busy}; DRAM idle us/chunk (approx) {idle}")
        print("   vector busy by op (us, 2 chunks):",
              {k[2:]: round(v / 1e3) for k, v in sorted(st["notes"].items(), key=lambda x: -x[1]) if k[0] in "VM" and v > 5e4})
    print(f"({time.time() - t0:.0f} s)")
