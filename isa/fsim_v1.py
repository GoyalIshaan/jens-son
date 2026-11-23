"""Functional simulator for ISA v1 (progression step 6a). It runs a real program on real data.

It is the timing simulator (sim_v1.Sim) plus values: every command's semantics, computed with golden v2's
bit-exact primitives (golden/bitexact.py for the matrix unit, golden/bitexact_vec.py and golden/chip_sfu.py
for the vector unit). The timing simulator already runs the scalar core, loops, queues and semaphores, so the
order in which commands touch memory is the one the hardware would produce.

Memory rules (what makes a missing semaphore visible):
  - a command reads all its inputs when it starts, and its results land when it signals (V, L: at completion;
    M: at its last beat; R: when its last byte arrives; X: when DRAM accepts the writes). A consumer that
    starts before its producer has signalled reads stale data, so the output comes out wrong.
  - W pushes DRAM bytes into the weight FIFO when it starts; TAB / MM pop them in program order.
  - LW / SW read and write 32-bit words of SRAM at issue time.

Operand boxes: element (p, m, n) of an operand is at base + reg + p*sp + m*sm + n*sn bytes, typed f32, f16,
i8, u32 or e8 (a signed exponent byte). numpy checks every box against the memory bounds.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden"))
import bitexact as B  # noqa: E402
import bitexact_vec as BV  # noqa: E402
import chip_sfu as SFU  # noqa: E402
import isa_v1 as I  # noqa: E402
from sim_v1 import Sim, SimChip  # noqa: E402

NP = {"f32": np.float32, "f16": np.float16, "i8": np.int8, "u32": np.uint32, "e8": np.int8}
SFUN = {"EXP": SFU.exp, "SIGMOID": SFU.sigmoid, "SILU": SFU.silu, "SOFTPLUS": SFU.softplus, "RSQRT": SFU.rsqrt,
        "RECIP": SFU.recip}
_f = BV._f


def box(mem, o, off, shape):
    """A numpy view of an operand box (writable, no copy)."""
    p, m, n = shape
    return np.ndarray((p, m, n), dtype=NP[o.dtype], buffer=mem, offset=o.base + off,
                      strides=(o.sp, o.sm, o.sn))


def t32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def rne_i8(x):
    """fp32 -> int8 value: round half to even, saturate to +-127 (the vector unit's int8 conversions)."""
    return torch.clamp(torch.round(x), -127, 127)


class FSim(Sim):
    def __init__(self, prog, bundles, sram, dram, chip=SimChip(vec_lanes=512), regs=None, on_argmax=None,
                 trace=False):
        super().__init__(prog, bundles, chip, regs=regs, trace=trace)
        self.sram, self.dram = sram, dram
        self.fifo = bytearray()
        self.table = np.zeros(16, np.int8)
        self.tscale = np.float32(1.0)
        self.on_argmax = on_argmax
        self.on_land = None
        self.host_out = []
        self.counts = {}

    # ---------------------------------------------------------------- scalar memory: SRAM words
    def _scalar(self, s, next_pc):
        if s.op == "LW":
            a = self.x[s.rs1] + s.imm
            self.x[s.rd] = int(self.sram[a:a + 4].view(np.uint32)[0]) if s.rd else 0
            return next_pc
        if s.op == "SW":
            a = self.x[s.rs1] + s.imm
            self.sram[a:a + 4].view(np.uint32)[0] = self.x[s.rs2] & 0xFFFFFFFF
            return next_pc
        return super()._scalar(s, next_pc)

    # ---------------------------------------------------------------- hooks: execute at start, land at signal
    def _try_start(self, u):
        before = self.cur[u]
        super()._try_start(u)
        c = self.cur[u]
        if c is not None and c is not before:
            self.counts[(u, c.op)] = self.counts.get((u, c.op), 0) + 1
            c.pending = self._execute(c)

    def _land(self, c):
        w = getattr(c, "pending", None)
        if w:
            for fn in w:
                fn()
        c.pending = None
        if self.on_land is not None:
            self.on_land(self, c)

    def _done(self, u):
        self._land(self.cur[u])
        super()._done(u)

    def _m_end(self):
        self._land(self.cur["M"])
        super()._m_end()

    def _dma_end(self, u):
        self._land(self.cur[u])
        super()._dma_end(u)

    def _deliver(self, pick, c, n):
        if pick != "W" and c.bytes - n <= 1e-9:
            self._land(c)
        super()._deliver(pick, c, n)

    # zero-byte DMA commands finish through _done; NOP commands have nothing to land

    # ---------------------------------------------------------------- semantics
    def _execute(self, c):
        if c.op == "NOP":
            return None
        if c.unit in ("W", "R", "X", "L"):
            return self._dma(c)
        if c.unit == "M":
            return self._mxu(c)
        return self._vec(c)

    # DMA ---------------------------------------------------------------
    def _dma(self, c):
        d, rv = c.d, c.rv
        n1 = rv[2] or d.n1
        n2 = rv[3] or d.n2
        src_mem = {"W": self.dram, "R": self.dram, "X": self.sram, "L": self.sram}[c.unit]
        if c.op == "H2S":
            raise NotImplementedError("host input")
        src = np.ndarray((n2, n1, d.n0), np.uint8, buffer=src_mem, offset=d.src + rv[0],
                         strides=(d.s2_src, d.s1_src, 1)).copy()
        if c.unit == "W":
            self.fifo += src.tobytes()
            return None
        if c.op == "S2H":
            return [lambda: self.host_out.append(src.tobytes())]
        dst_mem = self.dram if c.unit == "X" else self.sram

        def land():
            np.ndarray((n2, n1, d.n0), np.uint8, buffer=dst_mem, offset=d.dst + rv[1],
                       strides=(d.s2_dst, d.s1_dst, 1))[...] = src
        return [land]

    # matrix unit -------------------------------------------------------
    def _pop(self, n):
        assert len(self.fifo) >= n, f"weight FIFO underflow: need {n}, have {len(self.fifo)}"
        b = bytes(self.fifo[:n])
        del self.fifo[:n]
        return b

    def _mxu(self, c):
        if c.op == "TAB":
            def tab():
                h = np.frombuffer(self._pop(32), np.uint8)
                self.table = h[:16].view(np.int8).copy()
                self.tscale = h[16:20].view(np.float32)[0]
            return [tab]
        d, rv = c.d, c.rv
        ov = rv[4]
        n_in = ov if (ov and d.override == "n_in") else d.n_in
        n_out = ov if (ov and d.override == "n_out") else d.n_out
        R = d.rows
        q = np.ndarray((R, n_in), np.int8, buffer=self.sram, offset=d.act + rv[0], strides=(d.act_stride, 1))
        e = np.ndarray((R, n_in // 32), np.int8, buffer=self.sram, offset=d.exps + rv[1],
                       strides=(d.exps_stride, 1))
        q = torch.from_numpy(q.astype(np.float32))
        e = torch.from_numpy(e.astype(np.int32))
        wsnap = None
        if c.op == "MMS":                                   # SRAM weights: read at start like any input
            wsnap = np.ndarray(n_in * n_out, np.int8, buffer=self.sram, offset=d.wsrc + rv[3]).copy()

        def land():
            # FIFO weights stream in while the command runs: they are all through only at its end
            y = self._mm_compute(d, q, e, n_in, n_out, wsnap)
            self._mm_epi(d, rv, R, n_out, y)
        return [land]

    def _mm_compute(self, d, q, e, n_in, n_out, wsnap):
        if d.wfmt == "lut4":
            assert wsnap is None and d.seg == 1
            raw = np.frombuffer(self._pop(n_in * n_out * 9 // 16), np.uint8)
            tiles, chunks = n_out // 64, n_in // 1024
            r = raw.reshape(tiles, chunks, 64, 576)
            codes = r[..., :512]
            nib = np.stack([codes & 15, codes >> 4], -1).reshape(tiles, chunks, 64, 1024)
            k = self.table[nib].astype(np.float32).transpose(0, 2, 1, 3).reshape(n_out, n_in)
            s8 = r[..., 512:].copy().view(np.uint8).transpose(0, 2, 1, 3).reshape(n_out, n_in // 16)
            s8 = torch.from_numpy(s8.copy()).view(torch.float8_e4m3fn).float()
            M, E = B.e4m3_split(s8)
            ts = torch.tensor(float(self.tscale) * d.tscale, dtype=torch.float32)
            y = B.mm_exact(q, e, torch.from_numpy(k), M, E, ts)
        else:
            # int8 weights, segment-major: [n_in / seg_w segments][n_out outputs][seg_w inputs]
            sw = I.LANES // d.seg
            nseg = n_in // sw
            nb = n_in * n_out
            raw = np.frombuffer(self._pop(nb), np.int8) if wsnap is None else wsnap
            k = raw.reshape(nseg, n_out, sw).transpose(1, 0, 2).reshape(n_out, n_in).astype(np.float32)
            G = n_in // 16
            ts = torch.tensor(d.tscale, dtype=torch.float32)
            y = B.mm_exact(q, e, torch.from_numpy(k), torch.ones(n_out, G), torch.zeros(n_out, G), ts, beat=sw)
        return y.numpy()

    def _mm_epi(self, d, rv, R, n_out, y):
        if True:
            if d.epi == "argmax":
                idx = (np.argmax(y, axis=1) + d.id_base).astype(np.uint32)          # ties: lowest index
                for t in range(R):
                    self.sram[d.dst + rv[2] + t * d.dst_stride:][:4].view(np.uint32)[0] = idx[t]
                if self.on_argmax:
                    self.on_argmax(self, idx, y)
                return
            dst = np.ndarray((R, n_out), np.float32, buffer=self.sram, offset=d.dst + rv[2], strides=(d.dst_stride, 4))
            if d.epi == "acc":
                dst[...] = _f(t32(dst) + torch.from_numpy(y)).numpy()
            else:
                dst[...] = y

    # vector unit -------------------------------------------------------
    def _vec(self, c):
        d, rv, op = c.d, c.rv, c.op
        par = dict(d.par)
        e_, n, m, p = d.elems(rv[4] if d.override in ("n", "m", "p") else 0)
        S = self.sram
        if op == "DN":
            heads = par.get("heads", 1)
            if not (1 <= heads <= 4 and rv[1] + heads <= par["lv"] and (heads == 1 or par["commit_spec"])):
                raise ValueError("invalid resident DeltaNet head group")
            return [self._dn(c, par, slot) for slot in range(heads)]
        if op == "CONV4":
            return self._conv4(c, par, n, m, p)
        if op == "CAUSAL":
            a = t32(box(S, d.a, rv[1], (p, m, n)))
            pos = rv[2] + torch.arange(n)[None, None, :]
            lim = rv[3] + torch.arange(p)[:, None, None] * par.get("step", 1)
            y = torch.where(pos > lim, torch.tensor(float("-inf")), a)
            return self._put(d.dst, rv[0], (p, m, n), y)
        if op == "SMTILE":
            return self._smtile(c, par, n, m, p)
        if op == "MATCH":
            a = box(S, d.a, rv[1], (1, 1, n))[0, 0]
            b = box(S, d.b, rv[2], (1, 1, n))[0, 0]
            k_ = 0
            while k_ < n and a[k_] == b[k_]:
                k_ += 1
            return self._put(d.dst, rv[0], (1, 1, 1), torch.tensor([[[k_]]]), raw=np.uint32)
        a = t32(box(S, d.a, rv[1], (p, m, n))) if d.a is not None else None
        if par.get("neg"):
            a = -a
        if op in ("RSUM", "RSUMSQ", "RMAX"):
            y = a.amax(-1) if op == "RMAX" else BV.reduce_sum(a, square=(op == "RSUMSQ"))
            return self._put(d.dst, rv[0], (p, m, 1), y[..., None])
        if op == "QUANT":
            x = a
            if d.b is not None:
                x = _f(x * t32(box(S, d.b, rv[2], (p, m, n))))
            q, ex = B.quant_act_exact(x)
            ex = ex + par.get("exp_bias", 0)
            w1 = self._put(d.dst, rv[0], (p, m, n), q)
            w2 = self._put(d.c, rv[3], (p, m, n // 32), ex.float())
            return w1 + w2
        if op == "QKV":
            amax = a.abs().amax(-1)
            s16 = (amax / 127).half()
            sf = s16.float()
            r = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))           # one exact reciprocal per row
            q = rne_i8(_f(a * r[..., None]))
            w1 = self._put(d.dst, rv[0], (p, m, n), q)
            w2 = self._put(d.c, rv[3], (p, m, 1), sf[..., None])
            return w1 + w2
        if op == "DEQ":
            s_ = t32(box(S, d.b, rv[2], (p, m, n // 32)))
            y = _f(a * s_.repeat_interleave(32, -1))
            return self._put(d.dst, rv[0], (p, m, n), y)
        if op == "ROPE":
            rot = par["rot"]
            cs = t32(box(S, d.b, rv[2], (p, 1, 64)))[:, 0]                   # per token: 32 cos | 32 sin
            y = BV.rope(a.reshape(p, m, n), cs[:, :rot // 2], cs[:, 32:32 + rot // 2], rot)
            return self._put(d.dst, rv[0], (p, m, n), y)
        b = self._rd(d.b, rv[2], (p, m, n)) if d.b is not None else None
        cc = self._rd(d.c, rv[3], (p, m, n)) if d.c is not None else None
        if op == "ADD":
            y = _f(a + b)
        elif op == "SUB":
            y = _f(a - b)
        elif op == "MUL":
            y = _f(a * b)
        elif op == "MAX":
            y = torch.maximum(a, b)
        elif op == "FMA":
            y = BV.fma(a, b, cc)
        elif op == "COPY":
            y = rne_i8(a) if d.dst.dtype == "i8" else a
        elif op in SFUN:
            if b is not None and cc is not None:
                x = BV.fma(a, b, cc)
            elif b is not None:
                x = _f(a * b)
            elif cc is not None:
                x = _f(a + cc)
            else:
                x = a
            y = SFUN[op](x)
        else:
            raise NotImplementedError(op)
        return self._put(d.dst, rv[0], (p, m, n), y)

    def _rd(self, o, off, shape):
        """An input operand as fp32. An e8 operand read by an elementwise op is 2^e, one byte per 32 along n
        (its sn is the stride from one 32-group to the next)."""
        if o.dtype != "e8":
            return t32(box(self.sram, o, off, shape))
        p, m, n = shape
        g = box(self.sram, o, off, (p, m, -(-n // 32)))
        e = torch.from_numpy(g.astype(np.int32))
        return torch.ldexp(torch.ones(e.shape), e).repeat_interleave(32, -1)[..., :n]

    def _smtile(self, c, par, n, m, p):
        """R7 over a (p, m, n) box of scores: rows (p, m), n positions. No mask operand: an input of -inf (written by
        CAUSAL) stays -inf. b: K scale per position, V scales `vs` bytes later; c: m per row, alpha at c + 4 mlb,
        l at c + 5 mlb; dst: P int8, exponents at dst + e[0], e[1] bytes per row."""
        d, rv, S = c.d, c.rv, self.sram
        from isa_v1 import Opnd
        s = t32(box(S, d.a, rv[1], (p, m, n)))
        ks = t32(box(S, d.b, rv[2], (p, m, n)))
        vs = t32(box(S, Opnd(d.b.base + par["vs"], d.b.sn, d.b.sm, d.b.sp, d.b.dtype), rv[2], (p, m, n)))
        mlb = par["mlb"]
        Oal = Opnd(d.c.base + 4 * mlb, d.c.sn, d.c.sm, d.c.sp, "f32")
        Ol = Opnd(d.c.base + 5 * mlb, d.c.sn, d.c.sm, d.c.sp, "f32")
        mo = t32(box(S, d.c, rv[3], (p, m, 1)))[..., 0]
        lo = t32(box(S, Ol, rv[3], (p, m, 1)))[..., 0]
        sc = torch.where(torch.isneginf(s), torch.full_like(s, float("-inf")), _f(s * ks))
        mn = torch.maximum(mo, sc.amax(-1))
        pe = SFU.exp(sc - mn[..., None])
        lt = BV.reduce_sum(pe)
        al = SFU.exp(mo - mn)
        ln = BV.fma(lo, al, lt)
        assert n % 32 == 0
        q, e = B.quant_act_exact(_f(pe * vs))
        eo, er = par["e"]
        Oe = Opnd(d.dst.base + eo, 1, er, 0, "e8")
        return (self._put(d.dst, rv[0], (p, m, n), q) + self._put(Oe, rv[0], (p, m, n // 32), e.float())
                + self._put(d.c, rv[3], (p, m, 1), mn[..., None]) + self._put(Oal, rv[3], (p, m, 1), al[..., None])
                + self._put(Ol, rv[3], (p, m, 1), ln[..., None]))

    def _put(self, o, off, shape, y, raw=None):
        arr = y.numpy() if isinstance(y, torch.Tensor) else y
        if o.dtype in ("i8", "e8"):
            arr = arr.astype(np.int8)
        elif o.dtype == "u32" or raw is np.uint32:
            arr = arr.astype(np.uint32)
        else:
            arr = arr.astype(NP[o.dtype])
        sram = self.sram

        def land():
            box(sram, o, off, shape)[...] = arr
        return [land]

    def _conv4(self, c, par, n, m, p):
        """R5. Channels are the (m, n) box, token rows p. b: 4 fp32 taps per channel (oldest first); c: the 3
        history rows (row stride c.sp). par 'window': also write the (3 + p)-row window, row stride n*m*4."""
        d, rv, S = c.d, c.rv, self.sram
        x = t32(box(S, d.a, rv[1], (p, m, n)))
        w = np.ndarray((m, n, 4), np.float32, buffer=S, offset=d.b.base + rv[2], strides=(d.b.sm, d.b.sn, 4))
        w = t32(w).reshape(m * n, 4)
        h = np.ndarray((3, m, n), np.float32, buffer=S, offset=d.c.base + rv[3], strides=(d.c.sp, d.c.sm, d.c.sn))
        h = t32(h).reshape(3, m * n)
        y, _ = BV.conv4(x.reshape(p, m * n), h, w)
        out = self._put(d.dst, rv[0], (p, m, n), y.reshape(p, m, n))
        if "window" in par:
            win = torch.cat([h, x.reshape(p, m * n)]).numpy()
            base = par["window"]

            def land():
                S[base:base + win.nbytes].view(np.float32)[:] = win.reshape(-1)
            out.append(land)
        return out

    def _dn(self, c, par, slot):
        """One slot of a resident DeltaNet group (R4). a: the conv-output rows (q | k | v | beta | decay per token);
        b: the replay log rows (same layout); c: the head's [128 x 128] state (+ r3); r1 = first head index;
        r4 = logged tokens to replay first (committed). commit_spec 1: run T tokens in place; 0: on a copy."""
        d, rv, S = c.d, c.rv, self.sram
        LQK, LV, LD = par["lqk"], par["lv"], 128
        T, h, nrep = par["T"], rv[1] + slot, rv[4]
        kh = h // (LV // LQK)
        QKV = 2 * LQK * LD + LV * LD
        width = QKV + 2 * LV
        st = np.ndarray((LD, LD), np.float32, buffer=S, offset=d.c.base + rv[3] + slot * LD * d.c.sm, strides=(d.c.sm, 4))
        Sm = t32(st)

        def row(o, t):
            r = t32(np.ndarray(width, np.float32, buffer=S, offset=o.base + t * o.sp))
            return (r[kh * LD:(kh + 1) * LD], r[LQK * LD + kh * LD:LQK * LD + (kh + 1) * LD],
                    r[2 * LQK * LD + h * LD:2 * LQK * LD + (h + 1) * LD], r[QKV + h], r[QKV + LV + h])
        for t in range(nrep):
            q, k, v, beta, dd = row(d.b, t)
            Sm, _ = BV.dn_token(Sm, k, q, v, beta, None, out=False, d=dd)
        work = Sm if par["commit_spec"] else Sm.clone()
        outs = []
        for t in range(T):
            q, k, v, beta, dd = row(d.a, t)
            work, o = BV.dn_token(work, k, q, v, beta, None, d=dd)
            outs.append(o)
        committed = work if par["commit_spec"] else Sm
        dst_base = d.dst.base + rv[0]

        def land():
            st[...] = committed.numpy()
            for t in range(T):
                S[dst_base + t * d.dst.sp + h * LD * 4:][:LD * 4].view(np.float32)[:] = outs[t].numpy()
        return land
