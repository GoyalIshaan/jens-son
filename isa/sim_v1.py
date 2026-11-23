"""Timing simulator for ISA v1 (note 28). Not functional: it runs the real program (scalar control, loops,
registers, queues, semaphores) and times every command, but it does not compute values. Loads of values
the units produce (argmax token ids, the accepted-draft count) come from an `oracle`.

Model
- Issue: one bundle per cycle, in order. Stalls on a full unit queue (16) or on WAITS.
- DRAM: one shared server at bw x util, served in chunks (16 KB). Reads arrive `latency` after their
  chunk is served (pipelined). Writes cost `write_penalty` x their bytes in DRAM time and are batched:
  served when nothing else is waiting, or first once `x_force` bytes are pending.
  Priority otherwise: R (state, KV scales, embeddings) > W (weights, K, V) > X.
- Weight FIFO: W chunks are requested only while the FIFO has room; the matrix unit consumes FIFO bytes in
  program order at beat_bytes per (ceil(rows / 64)) cycles. TAB pops the 32-byte header.
- Matrix unit: the next command may start as soon as the previous one has consumed its last weights; each
  command's results (and semaphore) are ready `mxu_drain` cycles after that.
- Vector unit: one command at a time, `vec_overhead` + elements / rate cycles (rates from D-020).
- L DMA (SRAM->SRAM, host): bytes / 64 + overhead cycles, no DRAM.
"""
import heapq
import math
from collections import deque
from dataclasses import dataclass

import isa_v1 as I

UNITS = ("W", "R", "X", "L", "M", "V")


# One 128-wide R4 token on the 512-lane RTL with ready SRAM, from launch to done, measured on the
# dependency-scheduled controller (rtl/vec/vec_dn_head.sv, vec_dn_group.sv; bench tb_dn_interleave.sv, no stalls).
# 1-3 heads are bound by the S.k chain's latency (32 dependent FMAs); 4 heads by bank issue slots.
DN_CYCLES = {False: {1: 497, 2: 500, 3: 503, 4: 617}, True: {1: 496, 2: 498, 3: 500, 4: 502}}


def dn_group_cycles(heads, replay):
    """Fetch and launch, the controller, retire, then write each output row (replay writes none)."""
    return heads + 1 + DN_CYCLES[bool(replay)][heads] + 1 + (0 if replay else heads)



@dataclass
class SimChip:
    bw: float = 512e9
    util: float = 0.92
    clk: float = I.CLK
    write_penalty: float = 1.15
    latency_ns: float = 150.0
    chunk: int = 16384
    fifo: int = I.WFIFO_BYTES
    x_force: int = 64 * 1024
    rows: int = I.ROWS
    vec_lanes: int = I.VEC_LANES
    sfu: tuple = tuple(sorted(I.SFU_RATE.items()))
    vec_overhead: int = 8
    mxu_drain: int = 24
    l_overhead: int = 16
    queue_depth: int = I.QUEUE_DEPTH


class Cmd:
    __slots__ = ("unit", "op", "d", "rv", "wait", "sig", "note", "t_issue", "t_start", "t_end",
                 "remaining", "fifo_bytes", "beat_bytes", "hold", "cycles", "bytes", "unit_nop", "pending")


class Sim:
    def __init__(self, prog, bundles, chip=SimChip(), regs=None, oracle=None, trace=False):
        self.p, self.b, self.c = prog, bundles, chip
        self.ns = 1e9 / chip.clk                    # ns per cycle
        self.bw = chip.bw * chip.util / 1e9         # bytes per ns
        self.x = [0] * I.N_REGS
        for k, v in (regs or {}).items():
            self.x[k] = v
        self.mem = {}
        self.oracle = oracle or (lambda addr: 0)
        self.sem = [0] * (I.N_SEMS + 1)
        self.q = {u: deque() for u in UNITS}
        self.cur = {u: None for u in UNITS}
        self.loops = []
        self.pc = 0
        self.halted = False
        self.ev = []
        self.seq = 0
        self.now = 0.0
        self.last_issue = -1.0
        self.issue_sched = False
        self.issue_block = None
        self.fifo_level = self.fifo_infl = self.fifo_busy = 0
        self.dram_free_at = 0.0
        self.dram_sched = False
        self.m_busy_until = 0.0
        self.m_piece = False
        self.trace = trace
        self.stats = dict(bundles=0, slots=0, cmds={u: 0 for u in UNITS}, busy={u: 0.0 for u in UNITS},
                          dram_busy=dict(R=0.0, W=0.0, X=0.0), dram_bytes=dict(R=0, W=0, X=0),
                          issue_stall=dict(queue=0.0, waits=0.0), idle_dram=dict(), notes={})
        self._idle_since = 0.0
        self.ver = 0                                # bumps on any semaphore signal or queue pop
        self.block_ver = -1
        self._w_consumed = 0
        self._w_requested = 0

    # ---------------------------------------------------------------- events
    def at(self, t, fn, *a):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, fn, a))

    def run(self, max_events=50_000_000):
        self.at(0.0, self._issue)
        self.issue_sched = True
        n = 0
        while self.ev:
            t, _, fn, a = heapq.heappop(self.ev)
            self.now = t
            fn(*a)
            self._poke()
            n += 1
            assert n < max_events, "event limit"
        busy_units = [u for u in UNITS if self.cur[u] or self.q[u]]
        if not (self.halted and not busy_units):
            self.diagnose()
        assert self.halted and not busy_units, f"deadlock at pc {self.pc}: busy {busy_units}, sems {self._sems()}"
        assert self._w_requested == self._w_consumed, \
            f"weight stream mismatch: DMA sent {self._w_requested} B, matrix unit used {self._w_consumed} B"
        self.stats["time_ns"] = self.now
        return self.stats

    def diagnose(self, names=None):
        names = names or getattr(self, "sem_names", {})
        nm = lambda s: names.get(s, s)
        print(f"t={self.now:.0f} pc={self.pc} issue_block={self.issue_block}")
        b = self.b[self.pc]
        print("  bundle:", [(o.slot, o.op, o.note, [nm(w) for w in o.wait]) for o in (b.S, b.D, b.M, b.V) if o])
        for u in UNITS:
            c = self.cur[u]
            h = self.q[u][0] if self.q[u] else None
            print(f"  {u}: cur={c.op + ' ' + c.note if c else None}"
                  f"{' rem=' + str(c.remaining) if c else ''} | head={h.op + ' ' + h.note if h else None} "
                  f"waits={[nm(w) for w in h.wait] if h else ''} qlen={len(self.q[u])}")
        print("  sems:", {nm(i): v for i, v in enumerate(self.sem) if v})
        print("  fifo level/infl/busy", self.fifo_level, self.fifo_infl, self.fifo_busy)

    def _why(self):
        """What the machine is doing when DRAM goes idle."""
        v, m = self.cur["V"], self.cur["M"]
        if m is not None and m.op != "NOP" and m.fifo_bytes and self.fifo_level < 1:
            return "waiting on DRAM latency"
        if v is not None:
            return "vector " + v.op + (" (" + v.note + ")" if v.note else "")
        if m is not None:
            return "matrix " + m.op + (" (" + m.note + ")" if m.note else "")
        hm = self.q["M"][0] if self.q["M"] else None
        if hm is not None:
            nm = getattr(self, "sem_names", {})
            return "matrix waits " + ",".join(str(nm.get(w, w)) for w in hm.wait)
        return "nothing queued (issue)"

    def _sig(self, s):
        self.sem[s] += 1
        self.ver += 1

    def _sems(self):
        return {i: v for i, v in enumerate(self.sem) if v}

    def _poke(self):
        for u in UNITS:
            if self.cur[u] is None:
                self._try_start(u)
        if self.cur["M"] is not None and not self.m_piece:
            self._m_next()
        self._dram_kick()
        if self.issue_block is not None and not self.issue_sched and self.ver != self.block_ver:
            self.issue_sched = True
            self.at(max(self.now, self.last_issue + self.ns), self._issue)

    # ---------------------------------------------------------------- issue (scalar core)
    def _issue(self):
        self.issue_sched = False
        if self.halted:
            return
        bd = self.b[self.pc]
        units = [o for o in (bd.D, bd.M, bd.V) if o is not None]
        s = bd.S
        reason = None
        if s is not None and s.op == "WAITS" and self.sem[s.imm] <= 0:
            reason = "waits"
        for o in units:
            u = self._unit_of(o)
            if len(self.q[u]) >= self.c.queue_depth:
                reason = "queue"
        if reason:
            if self.issue_block is None:
                self.issue_block = (reason, self.now)
            self.block_ver = self.ver
            return
        if self.issue_block is not None:
            r, t0 = self.issue_block
            self.stats["issue_stall"][r] += self.now - t0
            self.issue_block = None
        self.last_issue = self.now
        self.stats["bundles"] += 1
        self.stats["slots"] += bd.used()
        for o in units:                                # registers are read before the scalar op writes
            self._enqueue(o)
        next_pc = self.pc + 1
        if s is not None:
            next_pc = self._scalar(s, next_pc)
        # hardware loops: at the end of a body, branch back while iterations remain
        while self.loops and next_pc == self.loops[-1][1]:
            start, end, left = self.loops[-1]
            if left > 1:
                self.loops[-1][2] = left - 1
                next_pc = start
                break
            self.loops.pop()
        self.pc = next_pc
        if not self.halted:
            self.issue_sched = True
            self.at(self.now + self.ns, self._issue)

    def _scalar(self, s, next_pc):
        x, op = self.x, s.op
        a, b = x[s.rs1], x[s.rs2]
        v = None
        if op == "ADD":
            v = a + b
        elif op == "SUB":
            v = a - b
        elif op == "MUL":
            v = a * b
        elif op == "AND":
            v = a & b
        elif op == "OR":
            v = a | b
        elif op == "XOR":
            v = a ^ b
        elif op == "MIN":
            v = min(a, b)
        elif op == "MAX":
            v = max(a, b)
        elif op == "ADDI":
            v = a + s.imm
        elif op == "ANDI":
            v = a & s.imm
        elif op == "SLLI":
            v = a << s.imm
        elif op == "SRLI":
            v = a >> s.imm
        elif op == "LI":
            v = s.imm
        elif op == "LW":
            addr = a + s.imm
            v = self.mem[addr] if addr in self.mem else self.oracle(addr)
        elif op == "SW":
            self.mem[a + s.imm] = b
        elif op == "LOOP":
            n = x[s.rs1]
            if n <= 0:
                return next_pc + s.imm
            assert len(self.loops) < I.LOOP_DEPTH, "loop nesting too deep"
            self.loops.append([next_pc, next_pc + s.imm, n])
        elif op == "WAITS":
            self.sem[s.imm] -= 1
        elif op == "SIGS":
            self.sem[s.imm] += 1
        elif op == "HALT":
            self.halted = True
        elif op in ("BEQ", "BNE", "BLT", "J"):
            take = op == "J" or (op == "BEQ" and a == b) or (op == "BNE" and a != b) or (op == "BLT" and a < b)
            if take:
                return next_pc + s.imm
        if v is not None and s.rd:
            x[s.rd] = v & 0xFFFFFFFFFFFF
        return next_pc

    @staticmethod
    def _unit_of(o):
        return I.UNIT_OF_DOP[o.op] if o.slot == "D" else o.slot

    def _enqueue(self, o):
        c = Cmd()
        c.unit = self._unit_of(o)
        c.op = "NOP" if o.op.startswith("NOP") else o.op
        c.d = None if (c.op == "NOP" or c.op == "TAB") else self.p.descs[o.desc]
        c.wait, c.sig, c.note = o.wait, o.sig, o.note
        c.rv = tuple(self.x[r] for r in o.regs) + (0,) * (5 - len(o.regs))
        c.t_issue = self.now
        self._cost(c)
        self.q[c.unit].append(c)
        self.stats["cmds"][c.unit] += 1

    # ---------------------------------------------------------------- command costs
    def _cost(self, c):
        cy = 0.0
        c.fifo_bytes = c.beat_bytes = c.hold = c.bytes = 0
        if c.op == "NOP":
            c.cycles, c.remaining = 1, 0
            c.bytes = c.fifo_bytes = 0
            return
        if c.unit in ("W", "R", "X", "L"):
            d = c.d
            n1 = c.rv[2] or d.n1
            n2 = c.rv[3] or d.n2
            c.bytes = d.n0 * n1 * n2
            if c.unit == "L":
                # a byte transpose (n0 = 1, source along i1, destination along i2) goes through the L engine's
                # 64 x 64 tile: 64 clocks of row reads, then 64 of column writes (rtl/chip/chip_dma.sv)
                tp = d.n0 == 1 and d.s1_src == 1 and d.s2_dst == 1
                cy = self.c.l_overhead + c.bytes / (32 if tp else 64)
            if c.unit == "W":
                self._w_requested += c.bytes
        elif c.unit == "M":
            if c.op == "TAB":
                c.fifo_bytes, c.beat_bytes, c.hold, cy = 32, 32, 1, 1      # header: table + tscale
            else:
                d = c.d
                ov = c.rv[4]
                n_in = ov if (ov and d.override == "n_in") else d.n_in
                n_out = ov if (ov and d.override == "n_out") else d.n_out
                vals = n_in * n_out
                bpv = 9 / 16 if d.wfmt == "lut4" else 1.0
                c.beat_bytes = I.LANES * bpv
                c.hold = math.ceil(d.rows / self.c.rows)
                c.fifo_bytes = vals * bpv if c.op == "MM" else 0
                # beats: seg outputs per beat, each from up to 1,024/seg inputs (a short n_in leaves lanes idle)
                cy = math.ceil(n_out / d.seg) * math.ceil(n_in * d.seg / I.LANES) * c.hold
        elif c.unit == "V":
            d = c.d
            e, n, m, p = d.elems(c.rv[4] if d.override in ("n", "m", "p") else 0)
            sfu = dict(self.c.sfu)
            if c.op in I.SFU_OF:
                rate = sfu[I.SFU_OF[c.op]]
            else:
                rate = self.c.vec_lanes
            if c.op == "CONV4":
                cy = e * 4 / self.c.vec_lanes
            elif c.op == "DN":
                par = dict(d.par)
                nc = c.rv[4]                               # tokens replayed from the log (committed)
                if "heads" in par:
                    heads = par["heads"]
                    cy = nc * dn_group_cycles(heads, True) + par["T"] * dn_group_cycles(heads, False)
                else:  # Legacy descriptors, including the sequential prefill cost model.
                    cy = (2 * nc + 3 * par["T"]) * 128 * 128 / self.c.vec_lanes
            elif c.op == "DNC":
                # chunked DeltaNet, vector part, one head x one 64-token chunk (note 30); d = 128, C = 64.
                # elementwise work counted in element-ops at full width, exps at the exp rate, and the
                # triangular solve as C(C-1)/2 * d fmas plus a latency tail of 16 steps x 4 cycles.
                par = dict(d.par)
                L_ = self.c.vec_lanes
                ph = par["phase"]
                if ph == 0:      # quantize q, k (both sides), the state copy, Ktil; decay matrix
                    cy = (49152 + 49152 + 98304 + 4096 + 8192 + 49152) / L_ + 4160 / min(sfu["exp"], L_)
                elif ph == 1:    # A, r, the solve, Delta and P quantized (hi + lo), state pre-scaled by gamma_C
                    cy = (8192 + 24576 + 49152 + 4096 + 24576 + 16384) / L_ + 64 * 63 // 2 * 128 / L_ + 64
                else:            # o = gamma * (S^T q) + P Delta
                    cy = 16384 / L_
            elif c.op == "MATCH":
                cy = n
            elif c.op == "SMTILE":
                # pass 1: x K scale, mask, row max; pass 2: exp, row sum, x V scale, round to int8 (exp-limited)
                cy = e / self.c.vec_lanes + e / min(sfu["exp"], self.c.vec_lanes)
            else:
                cy = e / rate
            cy += self.c.vec_overhead
        c.cycles = cy
        c.remaining = c.fifo_bytes if c.unit == "M" else c.bytes

    # ---------------------------------------------------------------- units
    def _try_start(self, u):
        if not self.q[u]:
            return
        c = self.q[u][0]
        if any(self.sem[s] <= 0 for s in c.wait):
            return
        if len(c.wait) == 2 and c.wait[0] == c.wait[1] and self.sem[c.wait[0]] < 2:
            return
        for s in c.wait:
            self.sem[s] -= 1
        self.q[u].popleft()
        self.ver += 1
        self.cur[u] = c
        c.t_start = self.now
        if u != "M" and (c.op == "NOP" or (u in ("R", "X", "W") and c.bytes == 0)):
            self.at(self.now + self.ns, self._done, u)
        elif u == "V" or u == "L":
            self.at(self.now + c.cycles * self.ns, self._done, u)
        elif u == "M":
            self.m_piece = False
        # W/R/X progress through the DRAM server

    def _done(self, u):
        c = self.cur[u]
        c.t_end = self.now
        self.stats["busy"][u] += c.t_end - c.t_start
        key = f"{u} {c.op}"
        self.stats["notes"][key] = self.stats["notes"].get(key, 0.0) + c.t_end - c.t_start
        bn = self.stats.setdefault("bynote", {})
        k2 = (u, c.note or c.op)
        bn[k2] = bn.get(k2, 0.0) + c.t_end - c.t_start
        if u == "M" and c.note in ("out-projection", "FFN down"):
            self.stats.setdefault("marks", []).append((c.note, c.t_end))
        for s in c.sig:
            self._sig(s)
        if self.trace:
            print(f"{self.now:12.0f} {u} {c.op:7s} {c.note}")
        self.cur[u] = None

    def _signal_later(self, t, c):
        def f():
            for s in c.sig:
                self._sig(s)
            if self.trace:
                print(f"{self.now:12.0f} M {c.op:7s} {c.note}")
        self.at(t, f)

    # matrix unit: consume the FIFO piece by piece
    def _m_next(self):
        c = self.cur["M"]
        if c.op == "NOP":
            self.m_piece = True
            self.at(self.now + self.ns, self._m_end)
            return
        if c.op == "MMS" or c.fifo_bytes == 0:
            self.m_piece = True
            self.at(self.now + c.cycles * self.ns, self._m_end)
            return
        if c.remaining <= 0:
            return
        piece = min(c.remaining, self.c.chunk)
        if self.fifo_level + 1e-9 < piece:
            return
        self.fifo_level -= piece
        self.fifo_busy += piece
        c.remaining -= piece
        self._w_consumed += piece
        self.m_piece = True
        dur = piece / c.beat_bytes * c.hold * self.ns if c.op != "TAB" else self.ns
        self.at(self.now + dur, self._m_piece_end, piece)

    def _m_piece_end(self, piece):
        self.fifo_busy -= piece
        self.m_piece = False
        c = self.cur["M"]
        if c.remaining <= 1e-9:
            self._m_end()

    def _m_end(self):
        c = self.cur["M"]
        c.t_end = self.now
        self.stats["busy"]["M"] += c.t_end - c.t_start
        bn = self.stats.setdefault("bynote", {})
        k2 = ("M", c.note or c.op)
        bn[k2] = bn.get(k2, 0.0) + c.t_end - c.t_start
        if c.note in ("out-projection", "o-projection", "FFN down"):
            self.stats.setdefault("marks", []).append((c.note, c.t_end))
        self.cur["M"] = None
        self.m_piece = False
        self._signal_later(self.now + self.c.mxu_drain * self.ns, c)

    # ---------------------------------------------------------------- DRAM
    def _dram_kick(self):
        if self.dram_sched or self.now < self.dram_free_at - 1e-9:
            return
        pick = None
        xr, rr, wr = self.cur["X"], self.cur["R"], self.cur["W"]
        w_ok = wr is not None and wr.remaining > 0 and \
            self.fifo_level + self.fifo_infl + self.fifo_busy + min(wr.remaining, self.c.chunk) <= self.c.fifo
        r_ok = rr is not None and rr.remaining > 0
        x_ok = xr is not None and xr.remaining > 0
        if x_ok and xr.remaining >= self.c.x_force:
            pick = "X"
        elif r_ok:
            pick = "R"
        elif w_ok:
            pick = "W"
        elif x_ok:
            pick = "X"
        if pick is None:
            if self._idle_since is None:
                self._idle_since = self.now
                self._idle_why = self._why()
            return
        if self._idle_since is not None and self.now > self._idle_since:
            gap = self.now - self._idle_since
            why = getattr(self, "_idle_why", "start")
            self.stats["idle_dram"][why] = self.stats["idle_dram"].get(why, 0.0) + gap
        self._idle_since = None
        c = self.cur[pick]
        n = min(c.remaining, self.c.chunk)
        c.remaining -= n
        dt = n / self.bw * (self.c.write_penalty if pick == "X" else 1.0)
        self.stats["dram_busy"][pick] += dt
        self.stats["dram_bytes"][pick] += n
        if pick == "W":
            self.fifo_infl += n
        self.dram_free_at = self.now + dt
        self.dram_sched = True
        self.at(self.dram_free_at, self._dram_served, pick, c, n)

    def _dram_served(self, pick, c, n):
        self.dram_sched = False
        if pick == "X":
            if c.remaining <= 0:
                self._dma_end("X")
            return
        self.at(self.now + self.c.latency_ns, self._deliver, pick, c, n)
        if c.remaining <= 0:                        # fully requested: the next command may start requesting
            c.t_end = self.now
            self.stats["busy"][pick] += c.t_end - c.t_start
            self.cur[pick] = None
            c.remaining = -1

    def _deliver(self, pick, c, n):
        if pick == "W":
            self.fifo_infl -= n
            self.fifo_level += n
        c.bytes -= n
        if c.bytes <= 1e-9:
            for s in c.sig:
                self._sig(s)
            if self.trace:
                print(f"{self.now:12.0f} {pick} {c.op:7s} {c.note}")

    def _dma_end(self, u):
        c = self.cur[u]
        c.t_end = self.now
        self.stats["busy"][u] += c.t_end - c.t_start
        key = f"{u} {c.op}"
        self.stats["notes"][key] = self.stats["notes"].get(key, 0.0) + c.t_end - c.t_start
        for s in c.sig:
            self._sig(s)
        self.cur[u] = None
