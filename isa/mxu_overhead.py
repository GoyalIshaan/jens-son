"""What the matrix unit's activation loads and epilogue cost on the 9B programs (note 33, part 5).

The timing simulator charges an MM only its beats. The RTL also loads each chunk's activations into the
stationary buffer and drains the accumulators through the epilogue. This adds those clocks to every M command,
under three designs, and reruns the 9B decode and prefill:

  serial    the RTL as first built: per chunk, 2 clocks per row (activations, then exponents); per 64-output
            tile (lut4) or per command (int8), the epilogue at E outputs per clock after the last beat
  overlap   activations double-buffered (the next chunk loads during this chunk's beats, exponents on their own
            port, 1 clock per row; a 1-chunk lut4 MM loads once); accumulators double-buffered per lut4 tile
            (a tile drains during the next tile's beats); a one-segment int8 MM streams each beat's outputs to
            the epilogue as soon as they are final
  the epilogue width E, in outputs per clock: 16 (one 64-byte line) up to 256 (16 lines)
  a "p" suffix packs a seg-4 command's activation load: rows of 256 bytes at a 256-byte stride, 4 per read

  python3 isa/mxu_overhead.py
"""
import math
import sys

import isa_v1 as I
import qwen_prefill_v1 as QP
import qwen_v1 as Q
from sim_v1 import Sim, SimChip


def extra_clocks(c, design, E, RP=I.ROWS, packed=False):
    """Clocks the RTL spends on M command c beyond the simulator's beats."""
    d = c.d
    ov = c.rv[4]
    n_in = ov if (ov and d.override == "n_in") else d.n_in
    n_out = ov if (ov and d.override == "n_out") else d.n_out
    rows = d.rows
    h = math.ceil(rows / RP)
    drain = lambda outs: math.ceil(outs * rows / E) if d.epi != "argmax" else math.ceil(outs * rows / E)
    if d.wfmt == "lut4":
        tiles, chunks = n_out // 64, n_in // 1024
        tile_beats = chunks * 64 * h
        if design == "serial":
            return tiles * (chunks * 2 * rows + drain(64))
        load = rows if chunks == 1 else rows + tiles * chunks * max(0, rows - 64 * h)
        return load + drain(64) + (tiles - 1) * max(0, drain(64) - tile_beats)
    segw = d.seg
    segs = n_in // (256 if segw == 4 else 1024)
    beats = n_out // segw
    if design == "serial":
        return segs * 2 * rows + drain(n_out)
    lrows = math.ceil(rows / 4) if (packed and segw == 4 and d.act_stride == 256) else rows
    load = lrows + (segs - 1) * max(0, lrows - beats * h)
    if segs == 1:
        return load + max(0, drain(n_out) - beats * h) + math.ceil(16 * rows / E)
    return load + drain(n_out)


class Timed(Sim):
    design, E, packed = None, 16, False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.drained = None          # the command whose extra clocks are already scheduled (held, so not reused)

    def _m_end(self):
        c = self.cur["M"]
        if self.design and c.op in ("MM", "MMS") and c is not self.drained:
            self.drained = c
            x = extra_clocks(c, self.design, self.E, packed=self.packed)
            self.stats.setdefault("m_extra", 0.0)
            self.stats["m_extra"] += x * self.ns
            if x > 0:
                self.at(self.now + x * self.ns, self._m_end)
                return
        super()._m_end()


def run(prog, regs, design, E, packed=False):
    s = Timed(prog, prog.assemble(), SimChip(vec_lanes=512), regs=regs, oracle=lambda a: 1000)
    s.design, s.E, s.packed = design, E, packed
    st = s.run()
    return st["time_ns"], st.get("m_extra", 0.0)


if __name__ == "__main__":
    cases = [("9B decode, 32K context", Q.build("decode", steps=1, lmax=32768, cfg=Q.Q9B, fused=True)[0],
              {Q.X_L: 32000}),
             ("9B prefill chunk at 16K (lite DeltaNet)",
              QP.build_prefill(chunks=1, lmax=40_000, fused=True, dn="lite")[0], {1: 16384})]
    which = sys.argv[1:] or ["base", "serial16", "overlap16", "overlap64", "overlap128", "overlap256",
                             "overlap128p", "overlap256p"]
    for name, prog, regs in cases:
        print(name)
        for w in which:
            packed = w.endswith("p")
            w0 = w.rstrip("p") if packed else w
            design, E = (None, 16) if w0 == "base" else (w0.rstrip("0123456789"), int(w0.lstrip("serialovp")))
            t, x = run(prog, regs, design, E, packed)
            print(f"  {w:11s} {t / 1e6:8.3f} ms   (matrix-unit extra {x / 1e6:.3f} ms)", flush=True)
