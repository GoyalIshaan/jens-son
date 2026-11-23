"""ISA v1 for the Qwen3.5-9B chip (progression step 3, note 28).

The machine, in one paragraph
- One 256-bit bundle issues per cycle, in order, from instruction memory. A bundle has four 64-bit slots:
  S (scalar / control), D (DMA), M (matrix unit), V (vector unit).
- The S slot runs in the issue stage (one cycle). The D, M and V slots don't run there: they push one
  *command* into that unit's queue (16 deep). Each unit runs its queue in order. A command may wait on up
  to two counting semaphores before it starts and signal up to two when it finishes. A full queue stalls
  issue; so does the scalar WAITS instruction. This is the only dynamic part: compute order is fixed by
  the program, and DRAM's variable timing is absorbed by queues and semaphores (note 05: "static compute,
  queued DMA").
- Commands are coarse: one command is a whole matrix, a whole vector operation over up to three
  dimensions, a whole DeltaNet head, a whole 2-D/3-D copy. Their shapes live in a descriptor table loaded
  with the program; registers add dynamic offsets (layer, position, buffer parity) at issue time.
- The weights never pass through SRAM: DMA queue W streams them from DRAM into a 96 KB weight FIFO that
  only the matrix unit reads, in program order. No semaphores are needed for weights.

Everything here is a v1 proposal (D-022, proposed). Units and numbers come from D-005, D-012, D-015,
D-019, D-020, D-021 and notes 24-27.
"""
from dataclasses import dataclass, field, fields
from typing import Optional

# ------------------------------------------------------------------ the machine (decided or proposed elsewhere)
CLK = 1.0e9
LANES = 1024                 # weight lanes = 64 groups of 16 (D-015)
ROWS = 64                    # token rows (D-015, leaning 64)
ACC_WORDS = 8192             # fp32 accumulators per half, two halves (D-029): 64 outputs x 128 rows
VEC_LANES = 256              # fp32 vector lanes, shared with the DeltaNet head step (note 18)
SFU_RATE = dict(exp=128, sigmoid=64, silu=64, softplus=2, rsqrt=2, recip=2)   # results per cycle (D-020; SiLU/sigmoid 64 by D-023)
SRAM_BYTES = int(6.5 * 2**20)  # D-012
SRAM_BANKS, BANK_BYTES = 64, 64  # 64 banks x 64 B lines, low-order interleaved: 4 KB/cycle peak
WFIFO_BYTES = 192 * 1024     # weight FIFO, outside SRAM_BYTES. 96 KB collapses at 150 ns DRAM latency (sim, note 28)
IMEM_BUNDLES = 8192          # 8,192 x 32 B = 256 KB instruction memory
DMEM_DESCS = 1024            # 1,024 x 64 B = 64 KB descriptor memory
N_REGS, N_SEMS = 32, 63      # x0 = 0; semaphore 0 means "none"
QUEUE_DEPTH = 16
LOOP_DEPTH = 4               # nested zero-overhead hardware loops
DRAM_ADDR_BITS, SRAM_ADDR_BITS = 36, 23

# ------------------------------------------------------------------ opcodes
S_OPS = ["NOP", "ADD", "SUB", "MUL", "AND", "OR", "XOR", "MIN", "MAX", "ADDI", "ANDI", "SLLI", "SRLI", "LI",
         "LW", "SW", "BEQ", "BNE", "BLT", "J", "LOOP", "WAITS", "SIGS", "HALT"]
D_OPS = ["NOP", "W", "R", "X", "L", "H2S", "S2H", "NOP_W", "NOP_R", "NOP_X", "NOP_L"]     # W: DRAM->weight FIFO, R: DRAM->SRAM, X: SRAM->DRAM,
#                                                        L: SRAM->SRAM (2-D/3-D, byte granularity), host in/out
M_OPS = ["NOP", "TAB", "MM", "MMS"]                   # TAB: pop the 32-byte matrix header (16-entry int8 table,
#                                                        fp32 tscale, 12 bytes pad) from the FIFO;
#                                                        MM: weights from the FIFO; MMS: weights from SRAM
V_OPS = ["NOP", "ADD", "SUB", "MUL", "MAX", "FMA", "COPY",
         "EXP", "SIGMOID", "SILU", "SOFTPLUS", "RSQRT", "RECIP",   # f(a*b + c); b, c optional
         "RSUM", "RSUMSQ", "RMAX",                                 # reduce the innermost dimension
         "QUANT", "QKV", "DEQ",                                    # int8 + 2^e per 32 (D-019); int8 + fp16 per row (KV)
         "ROPE", "CAUSAL", "CONV4", "DN", "MATCH",
         "SMTILE",                                                 # fused online-softmax step (D-023)
         "DNC"]                                                    # chunked DeltaNet, vector part (note 30)
UNIT_OF_DOP = {"W": "W", "R": "R", "X": "X", "L": "L", "H2S": "L", "S2H": "L", "NOP_W": "W", "NOP_R": "R",
               "NOP_X": "X", "NOP_L": "L"}
SFU_OF = {"EXP": "exp", "SIGMOID": "sigmoid", "SILU": "silu", "SOFTPLUS": "softplus", "RSQRT": "rsqrt",
          "RECIP": "recip"}


# ------------------------------------------------------------------ descriptors (64 bytes each in DMEM)
def _check(name, v, bits, signed=False):
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    assert lo <= v <= hi, f"{name}={v} does not fit {bits} bits"


@dataclass(frozen=True)
class Dma:
    """Up to 3-D copy: n0 contiguous bytes, repeated n1 times (strides s1), repeated n2 times (strides s2).
    src/dst are DRAM, SRAM or FIFO addresses depending on the opcode. 36+23 bit bases, 20-bit counts."""
    src: int
    dst: int
    n0: int
    n1: int = 1
    s1_src: int = 0
    s1_dst: int = 0
    n2: int = 1
    s2_src: int = 0
    s2_dst: int = 0

    def check(self):
        _check("src", self.src, DRAM_ADDR_BITS)
        _check("dst", self.dst, DRAM_ADDR_BITS)
        for f in ("n0", "n1", "n2"):
            _check(f, getattr(self, f), 24)
        for f in ("s1_src", "s1_dst", "s2_src", "s2_dst"):
            _check(f, getattr(self, f), 32, signed=True)       # 36+36+3x24+4x32 = 272 bits

    @property
    def bytes(self):
        return self.n0 * self.n1 * self.n2


@dataclass(frozen=True)
class Mm:
    """One matrix through the lanes. Activation int8 rows (data + one exponent byte per 32) x weights.
    wfmt: 'lut4' (4-bit code + E4M3 per 16, table from TAB) or 'int8' (K and V, no table, no scale).
    seg: 1 = one output per beat (1,024 inputs), 4 = four outputs per beat (256 inputs each; attention).
    epi: 'f32' write fp32 outputs, 'acc' add into fp32 dst (residual), 'argmax' running (best, index) per row.
    override: which count register r4 replaces ('n_in' or 'n_out')."""
    n_in: int
    n_out: int
    rows: int
    act: int
    act_stride: int
    exps: int
    exps_stride: int
    dst: int
    dst_stride: int
    wfmt: str = "lut4"
    seg: int = 1
    epi: str = "f32"
    wsrc: int = 0              # MMS: SRAM address of the weights
    id_base: int = 0           # argmax: added to the winning index
    tscale: float = 1.0        # per-tensor fp32 output multiply, gmax / 127 (note 29)
    override: str = ""

    def check(self):
        for f in ("n_in", "n_out"):
            _check(f, getattr(self, f), 20)
        _check("rows", self.rows, 10)
        for f in ("act", "exps", "dst", "wsrc"):
            _check(f, getattr(self, f), SRAM_ADDR_BITS)
        for f in ("act_stride", "exps_stride", "dst_stride"):
            _check(f, getattr(self, f), 20)
        assert self.wfmt in ("lut4", "int8") and self.seg in (1, 4) and self.epi in ("f32", "acc", "argmax")
        # n_in must fill whole beats: no lane mask (D-025 zero-pads short operands in SRAM instead, note 30)
        assert self.n_in % (LANES // self.seg) == 0 or self.override == "n_in", "n_in must fill whole beats"

    def values(self, n_in=None, n_out=None):
        return (n_in or self.n_in) * (n_out or self.n_out)


@dataclass(frozen=True)
class Opnd:
    """A strided operand: element (p, m, n) is at base + p*sp + m*sm + n*sn (bytes). dtype f32/f16/i8/u32/e8."""
    base: int
    sn: int = 4
    sm: int = 0
    sp: int = 0
    dtype: str = "f32"


@dataclass(frozen=True)
class Vec:
    """A vector-unit command over a (p, m, n) box. Reductions reduce n. QUANT/QKV/DEQ treat each (p, m) as
    one row of n values. DN, CONV4, ROPE, CAUSAL, MATCH use `par` for their own fields."""
    n: int
    m: int = 1
    p: int = 1
    dst: Optional[Opnd] = None
    a: Optional[Opnd] = None
    b: Optional[Opnd] = None
    c: Optional[Opnd] = None
    par: tuple = ()            # op-specific: ("exp_bias", -4), ("rows", 4), ...
    override: str = ""         # which dimension r4 replaces ('n', 'm' or 'p')

    def check(self):
        for f in ("n", "m", "p"):
            _check(f, getattr(self, f), 16)
        for o in (self.dst, self.a, self.b, self.c):
            if o is not None:
                _check("base", o.base, SRAM_ADDR_BITS)
                for s in ("sn", "sm", "sp"):
                    _check(s, getattr(o, s), 20, signed=True)          # 4 x (23 + 3x20 + 3) + 48 + 64 = 456 bits

    def elems(self, override_val=0):
        n, m, p = self.n, self.m, self.p
        if override_val:
            if self.override == "n":
                n = override_val
            elif self.override == "m":
                m = override_val
            elif self.override == "p":
                p = override_val
        return n * m * p, n, m, p


# ------------------------------------------------------------------ program items
@dataclass
class Op:
    slot: str                  # S, D, M, V
    op: str
    desc: int = 0              # D/M/V: descriptor index
    regs: tuple = ()           # D/M/V: up to 5 registers added to (or replacing) descriptor fields
    wait: tuple = ()           # up to 2 semaphores
    sig: tuple = ()            # up to 2 semaphores
    rd: int = 0                # S fields
    rs1: int = 0
    rs2: int = 0
    imm: int = 0
    note: str = ""

    def reads(self):
        if self.slot == "S":
            return {self.rs1, self.rs2} - {0}
        return set(self.regs) - {0}

    def writes(self):
        return {self.rd} - {0} if self.slot == "S" and self.op not in ("SW", "BEQ", "BNE", "BLT", "J", "LOOP",
                                                                         "WAITS", "SIGS", "HALT", "NOP") else set()


@dataclass
class Loop:
    count_reg: int
    body: list = field(default_factory=list)
    note: str = ""


@dataclass
class Label:
    """A branch target. It starts a new bundle; `assemble()` resolves branch offsets to it."""
    name: str


@dataclass
class Bundle:
    S: Optional[Op] = None
    D: Optional[Op] = None
    M: Optional[Op] = None
    V: Optional[Op] = None

    def used(self):
        return sum(x is not None for x in (self.S, self.D, self.M, self.V))


class Prog:
    """Builder: emit ops in program order; `assemble()` packs them into bundles."""

    def __init__(self):
        self.items = []
        self._stack = [self.items]
        self.descs = []
        self._dindex = {}

    # descriptors are de-duplicated: the same shape used in every layer is one entry
    def desc(self, d):
        d.check()
        if d not in self._dindex:
            self._dindex[d] = len(self.descs)
            self.descs.append(d)
        return self._dindex[d]

    def _emit(self, op):
        assert len(op.wait) <= 2 and len(op.sig) <= 2 and len(op.regs) <= 5
        for s in op.wait + op.sig:
            assert 1 <= s <= N_SEMS
        self._stack[-1].append(op)
        return op

    def S(self, op, rd=0, rs1=0, rs2=0, imm=0, note=""):
        assert op in S_OPS
        return self._emit(Op("S", op, rd=rd, rs1=rs1, rs2=rs2, imm=imm, note=note))

    def D(self, op, d, regs=(), wait=(), sig=(), note=""):
        assert op in D_OPS
        return self._emit(Op("D", op, self.desc(d), tuple(regs), tuple(wait), tuple(sig), note=note))

    def M(self, op, d=None, regs=(), wait=(), sig=(), note=""):
        assert op in M_OPS
        return self._emit(Op("M", op, self.desc(d) if d is not None else 0, tuple(regs), tuple(wait), tuple(sig),
                             note=note))

    def V(self, op, d=None, regs=(), wait=(), sig=(), note=""):
        assert op in V_OPS
        return self._emit(Op("V", op, self.desc(d) if d is not None else 0, tuple(regs), tuple(wait), tuple(sig),
                             note=note))

    def fence(self, slot_op, wait=(), sig=(), note=""):
        """A NOP command in one unit's queue that only waits and/or signals (cost: one cycle)."""
        slot, op = {"W": ("D", "W"), "R": ("D", "R"), "X": ("D", "X"), "L": ("D", "L"), "M": ("M", "NOP"),
                    "V": ("V", "NOP")}[slot_op]
        o = Op(slot, "NOP" if slot != "D" else op, 0, (), tuple(wait), tuple(sig), note="fence " + note)
        if slot == "D":
            o.op = "NOP_" + op
        return self._emit(o)

    def label(self, name):
        assert all(not (isinstance(it, Label) and it.name == name) for it in self._all_items()), f"label {name} twice"
        self._stack[-1].append(Label(name))

    def branch(self, op, rs1, rs2, target, note=""):
        """BEQ / BNE / BLT to a label (or J with rs1 = rs2 = 0). The offset is filled in by `assemble()`.
        A taken branch must not jump into or out of a hardware loop body (checked at assembly)."""
        assert op in ("BEQ", "BNE", "BLT", "J")
        o = self._emit(Op("S", op, rs1=rs1, rs2=rs2, note=note))
        o.target = target
        return o

    def _all_items(self):
        def walk(items):
            for it in items:
                yield it
                if isinstance(it, Loop):
                    yield from walk(it.body)
        return walk(self.items)

    def loop(self, count_reg, note=""):
        prog = self

        class _Ctx:
            def __enter__(self_):
                lp = Loop(count_reg, note=note)
                prog._stack[-1].append(lp)
                prog._stack.append(lp.body)
                return lp

            def __exit__(self_, *a):
                prog._stack.pop()
        return _Ctx()

    # ---------------------------------------------------------------- packing
    def assemble(self):
        """Greedy in-order packing. An op joins the open bundle if its slot is free and it doesn't read a
        register written by the bundle's scalar op (all slots read registers before the bundle writes).
        WAITS, LOOP and HALT close the bundle after themselves. A loop body is packed on its own."""
        out = []
        labels, branches = {}, []           # name -> (bundle index, loop path); [(op, bundle index, loop path)]
        path = []                           # ids of the enclosing hardware loops

        def pack(items):
            cur = Bundle()
            for it in items:
                if isinstance(it, Label):
                    if cur.used():
                        out.append(cur)
                        cur = Bundle()
                    labels[it.name] = (len(out), tuple(path))
                    continue
                if isinstance(it, Loop):
                    if cur.S is not None:
                        out.append(cur)
                        cur = Bundle()
                    lp = Op("S", "LOOP", rs1=it.count_reg, note=it.note)
                    cur.S = lp
                    out.append(cur)
                    start = len(out)
                    path.append(id(it))
                    pack(it.body)
                    path.pop()
                    lp.imm = len(out) - start
                    assert lp.imm >= 1, "empty loop body"
                    cur = Bundle()
                    continue
                slot_free = getattr(cur, it.slot) is None
                hazard = cur.S is not None and (cur.S.writes() & it.reads())
                if it.slot == "S" and it.op == "WAITS" and cur.used():
                    slot_free = False                          # a wait starts its own bundle
                if not slot_free or hazard:
                    out.append(cur)
                    cur = Bundle()
                setattr(cur, it.slot, it)
                if it.slot == "S" and getattr(it, "target", None) is not None:
                    branches.append((it, len(out), tuple(path)))
                if it.slot == "S" and (it.op in ("WAITS", "HALT") or getattr(it, "target", None) is not None):
                    out.append(cur)                            # a branch ends its bundle
                    cur = Bundle()
            if cur.used():
                out.append(cur)
        pack(self.items)
        for op, at, lpath in branches:
            assert op.target in labels, f"undefined label {op.target}"
            tgt, tpath = labels[op.target]
            assert tpath == lpath, f"branch to {op.target} crosses a hardware-loop boundary"
            op.imm = tgt - (at + 1)                            # relative to the next bundle, as the S unit adds it
        return out


# ------------------------------------------------------------------ binary encoding (checks the field widths)
def encode_slot(op):
    """64-bit slot. S: op5 rd5 rs1 5 rs2 5 imm32 (52 bits). D/M/V: op5 desc10 r0-r4 5x5 w1 w2 s1 s2 4x6 (64 bits)."""
    if op is None:
        return 0
    if op.slot == "S":
        code = S_OPS.index(op.op)
        _check("op", code, 5)
        _check("imm", op.imm, 32, signed=True)
        return code | op.rd << 5 | op.rs1 << 10 | op.rs2 << 15 | (op.imm & 0xFFFFFFFF) << 20
    table = {"D": D_OPS, "M": M_OPS, "V": V_OPS}[op.slot]
    code = table.index(op.op)
    _check("op", code, 5)
    _check("desc", op.desc, 10)
    word = code | op.desc << 5
    regs = list(op.regs) + [0] * (5 - len(op.regs))
    for i, r in enumerate(regs):
        word |= r << (15 + 5 * i)
    ws = list(op.wait) + [0] * (2 - len(op.wait))
    ss = list(op.sig) + [0] * (2 - len(op.sig))
    for i, s in enumerate(ws + ss):
        word |= s << (40 + 6 * i)
    assert word < 1 << 64
    return word


def encode(bundles):
    return [encode_slot(b.S) | encode_slot(b.D) << 64 | encode_slot(b.M) << 128 | encode_slot(b.V) << 192
            for b in bundles]
