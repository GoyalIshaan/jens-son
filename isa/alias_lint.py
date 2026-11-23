"""D-027 lint: a vector command's output must equal one of its inputs exactly (in place) or overlap none of them.

Checked on every V command as it is issued (addresses only, so it runs inside the timing simulator):
outputs are dst, plus c for QUANT and QKV; inputs are a, b and c. DN, CONV4, SMTILE, MATCH and DNC have their
own operand rules and are skipped here (their in-place state updates are by design).

  python3 isa/alias_lint.py        lint the 9B decode, speculative and prefill programs
"""
import sys

import isa_v1 as I
from fsim_v1 import NP

SKIP = {"DN", "CONV4", "SMTILE", "MATCH", "DNC", "NOP", "CAUSAL"}
OFFSET_OF = {"dst": 0, "a": 1, "b": 2, "c": 3}


def extent(o, off, shape):
    """Byte range [lo, hi) of an operand box; e8 operands read elementwise use one byte per 32 along n."""
    p, m, n = shape
    if o.dtype == "e8":
        n = -(-n // 32)
    size = NP[o.dtype]().itemsize
    lo = hi = o.base + off
    for count, stride in ((p, o.sp), (m, o.sm), (n, o.sn)):
        span = (count - 1) * stride
        lo, hi = (lo + span, hi) if span < 0 else (lo, hi + span)
    return lo, hi + size


def check(c):
    """Violations for one issued command (c.d is a Vec, c.rv its registers)."""
    if c.unit != "V" or c.op in SKIP or c.d is None:
        return []
    d, rv = c.d, c.rv
    _, n, m, p = d.elems(rv[4] if d.override in ("n", "m", "p") else 0)
    reduce = c.op in ("RSUM", "RSUMSQ", "RMAX")
    outs = [("dst", d.dst, (p, m, 1) if reduce else (p, m, n))]
    ins = [(k, getattr(d, k), (p, m, n)) for k in ("a", "b", "c") if getattr(d, k) is not None]
    if c.op in ("QUANT", "QKV"):
        outs.append(("c", d.c, (p, m, n // 32) if c.op == "QUANT" else (p, m, 1)))
        ins = [x for x in ins if x[0] != "c"]
    if c.op == "DEQ":
        ins = [(k, o, (p, m, n // 32) if k == "b" else s) for k, o, s in ins]
    bad = []
    for ok_, oo, os_ in outs:
        if oo is None:
            continue
        olo, ohi = extent(oo, rv[OFFSET_OF[ok_]], os_)
        for ik, io, is_ in ins:
            ilo, ihi = extent(io, rv[OFFSET_OF[ik]], is_)
            if ohi <= ilo or ihi <= olo:
                continue
            same = (oo.base + rv[OFFSET_OF[ok_]] == io.base + rv[OFFSET_OF[ik]] and
                    (oo.sn, oo.sm, oo.sp, oo.dtype) == (io.sn, io.sm, io.sp, io.dtype) and os_ == is_)
            if not same:
                bad.append(f"{c.op} {c.note!r}: {ok_} [{olo}, {ohi}) partly overlaps {ik} [{ilo}, {ihi})")
    return bad


def install(sim):
    """Wrap a simulator's _enqueue so every issued V command is checked; violations collect in sim.alias."""
    sim.alias = []
    orig = sim._enqueue

    def enq(o):
        orig(o)
        for u in ("V",):
            if sim.q[u]:
                sim.alias.extend(check(sim.q[u][-1]))
    sim._enqueue = enq
    return sim


if __name__ == "__main__":
    import prefill_run as PR
    import qwen_prefill_v1 as QP
    import qwen_v1 as Q
    from sim_v1 import Sim, SimChip
    chip = SimChip(vec_lanes=512)
    runs = []
    for mode in ("decode", "spec"):
        prog, info = Q.build(mode, steps=1, lmax=32768, cfg=Q.Q9B, fused=True)
        runs.append((f"9B {mode}", prog, {Q.X_L: 32000}))
    prog, info = QP.build_prefill(chunks=2, lmax=40_000, fused=True, dn="lite")
    runs.append(("9B prefill (lite DeltaNet, 2 chunks at 16K)", prog, {1: 16384}))
    # self-test: a shifted in-place ADD must be caught
    from isa_v1 import Opnd, Prog, Vec
    t = Prog()
    t.V("ADD", Vec(64, 2, dst=Opnd(4096, 4, 256), a=Opnd(4096 + 256, 4, 256), b=Opnd(0, 4, 256)))
    t.V("ADD", Vec(64, 2, dst=Opnd(8192, 4, 256), a=Opnd(8192, 4, 256), b=Opnd(0, 4, 256)))
    t.S("HALT")
    runs.append(("self-test (expect 1 violation)", t, {}))
    for name, prog, regs in runs:
        s = install(Sim(prog, prog.assemble(), chip, regs=regs, oracle=lambda a: 1000))
        s.run()
        uniq = sorted(set(s.alias))
        print(f"{name}: {len(s.alias)} violations ({len(uniq)} distinct)")
        for v in uniq[:20]:
            print("   ", v)
    sys.exit(0)
