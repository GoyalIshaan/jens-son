"""Bit-exact Python reference for mxu_group.v, and test-vector generator.

    python3 ref_group.py R MODE N OUTDIR

Writes OUTDIR/{stat,beat,exp}.hex. Each stationary block (table, activations, activation scales)
is followed by BEATS weight beats streamed back to back, so the test also checks the pipeline.
Includes worst-case blocks (all +/-127) that hit the 19-bit adder-tree limit.
"""
import os
import sys

import numpy as np

BEATS = 8


def e4m3_parts(code):
    """E4M3 (fn variant) -> (sign, M, E) with value = (-1)^sign * M * 2^E exactly."""
    s, e, m = code >> 7, (code >> 3) & 15, code & 7
    if e == 0:
        return s, m, -6 - 3
    return s, 8 + m, e - 7 - 3


def f32_bits(x):
    """fp32 bits with the chip's conventions: below 2^-126 (incl. zero) -> +0."""
    x = np.float32(x)
    if x == 0 or abs(float(x)) < 2.0 ** -126:
        return 0
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def group_ref(tbl, act, as_, codes, ws, mode):
    """tbl[16] int, act[R][16] int, as_[R] (fp32 float, or int k for mode 2), codes[16], ws code."""
    s, M, E = e4m3_parts(ws)
    w = [tbl[c] for c in codes]
    out = []
    for r in range(len(act)):
        S = sum(int(a) * int(b) for a, b in zip(act[r], w))
        assert -2 ** 18 <= S < 2 ** 18
        if mode == 0:
            wsf = np.float32((-1) ** s * M * 2.0 ** E)                 # exact
            t = np.float32(wsf) * np.float32(as_[r])                    # fp32 multiply, RNE
            y = np.float32(S) * t                                       # fp32 multiply, RNE
            out.append(f32_bits(y))
        elif mode in (1, 3):
            P = (-1) ** s * S * M
            y = np.float32(float(P) * 2.0 ** E * float(np.float32(as_[r])))   # exact in double, one RNE
            out.append(f32_bits(y))
        else:
            P = (-1) ** s * S * M
            out.append((P, E + int(as_[r])))
    return out


def pack(vals, width):
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & ((1 << width) - 1)) << (i * width)
    return v


def hexw(v, bits):
    return f"{v:0{(bits + 3) // 4}x}"


def main():
    R, mode, n_blocks, outdir = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.default_rng(1234 + 10 * R + mode)
    os.makedirs(outdir, exist_ok=True)
    valid_ws = [c for c in range(256) if c not in (0x7F, 0xFF)]
    stat, beat, exp = [], [], []
    for b in range(n_blocks):
        kind = b % 5
        if kind == 0:      # worst case: everything +127 -> S = 16 * 127 * 127 = 258,064
            tbl = np.full(16, 127)
            act = np.full((R, 16), 127)
        elif kind == 1:    # worst negative
            tbl = np.full(16, 127)
            act = np.full((R, 16), -127)
        else:
            tbl = rng.integers(-127, 128, 16)
            act = rng.integers(-127, 128, (R, 16))
        if mode == 2:
            as_ = rng.integers(-20, 5, R)
            as_bits = [int(k) & 0xFFFFFFFF for k in as_]
        elif mode == 3:                          # bf16-valued scales: low 16 bits of the fp32 pattern are zero
            as_ = (rng.uniform(1, 2, R) * 2.0 ** rng.integers(-20, 5, R)).astype(np.float32)
            as_ = (as_.view(np.uint32) & 0xFFFF0000).view(np.float32)
            as_bits = [int(x) for x in as_.view(np.uint32)]
        else:
            as_ = (rng.uniform(1, 2, R) * 2.0 ** rng.integers(-20, 5, R)).astype(np.float32)
            as_bits = [int(x) for x in as_.view(np.uint32)]
        stat.append(hexw(pack(tbl, 8), 128) + " " + hexw(pack(act.flatten(), 8), R * 128) + " "
                    + hexw(pack(as_bits, 32), R * 32))
        for _ in range(BEATS):
            codes = rng.integers(0, 16, 16)
            ws = int(rng.choice(valid_ws))
            if kind < 2:
                codes = np.zeros(16, dtype=int)
                ws = 0x38                         # 1.0
            y = group_ref(tbl, act, as_, codes, ws, mode)
            beat.append(hexw(pack(codes, 4), 64) + " " + hexw(ws, 8))
            if mode == 2:
                exp.append(hexw(pack([p for p, _ in y], 24), R * 24) + " " + hexw(pack([e for _, e in y], 10), R * 10))
            else:
                exp.append(hexw(pack(y, 32), R * 32))
    # $readmemh wants one value per word: split the space-separated fields into separate files
    def write(name, rows):
        cols = list(zip(*[r.split() for r in rows]))
        for j, col in enumerate(cols):
            with open(os.path.join(outdir, f"{name}{j}.hex"), "w") as f:
                f.write("\n".join(col) + "\n")
    write("stat", stat)
    write("beat", beat)
    write("exp", exp)
    print(f"R={R} MODE={mode}: {n_blocks} blocks x {BEATS} beats = {n_blocks * BEATS} vectors -> {outdir}")


if __name__ == "__main__":
    main()
