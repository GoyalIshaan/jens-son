"""Bit-exact reference and vectors for beat_reducer.v.

    python3 ref_reducer.py add N OUTDIR      # fp32_add: a, b -> y
    python3 ref_reducer.py red N OUTDIR      # reducer_p2 (64 groups, F=16) + 4-beat accumulation

Edge cases are mixed in on purpose: exact cancellation, near-cancellation, big exponent gaps (> 27),
ties, negative terms that floor when shifted out, all-zero beats.
"""
import os
import sys

import numpy as np

F, N, W = 16, 64, 46


def f32(x):
    """The chip's fp32 rounding of an exact double: RNE, below 2^-126 -> +0, -0 -> +0."""
    v = np.float32(x)
    if v == 0 or abs(float(v)) < 2.0 ** -126:
        return np.float32(0.0)
    return v


def bits(v):
    return int(np.array([v], dtype=np.float32).view(np.uint32)[0])


def fadd(a, b):
    return f32(float(np.float32(a)) + float(np.float32(b)))     # exact in double for normals, then one RNE


def rand_f32(rng, lo=-30, hi=30):
    m = rng.uniform(1, 2)
    return np.float32((-1) ** rng.integers(2) * m * 2.0 ** int(rng.integers(lo, hi)))


def reducer_ref(P, e):
    emax = max(e)
    S = 0
    for p, x in zip(P, e):
        sh = emax - x
        t = p << F
        t = (-1 if t < 0 else 0) if sh > W - 1 else t >> sh          # Python >> floors, like >>>
        S += t
    assert -2 ** (W - 1) <= S < 2 ** (W - 1)
    return f32(float(S) * 2.0 ** (emax - F))                          # |S| < 2^46: exact in double


def main():
    what, n, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(7)
    if what == "add":
        A, Bv, Y = [], [], []
        for i in range(n):
            k = i % 6
            a = rand_f32(rng)
            if k == 0:
                b = np.float32(-a)                                           # exact cancellation
            elif k == 1:
                b = np.float32(-float(a) * (1 + rng.integers(1, 64) * 2.0 ** -23))   # near cancellation
            elif k == 2:
                b = rand_f32(rng, -60, -40)                                  # huge gap
            elif k == 3:
                b = np.float32(float(a) * 2.0 ** -24 * (1 + 2.0 ** -23 * int(rng.integers(0, 3))))  # ties region
            else:
                b = rand_f32(rng)
            if i % 50 == 0:
                b = np.float32(0.0)
            A.append(bits(a)); Bv.append(bits(b)); Y.append(bits(fadd(a, b)))
        for name, col in (("a", A), ("b", Bv), ("y", Y)):
            open(os.path.join(outdir, f"add_{name}.hex"), "w").write("\n".join(f"{x:08x}" for x in col) + "\n")
    else:
        Pf, Ef, V, ACC = [], [], [], []
        for i in range(n):
            acc = np.float32(0.0)
            for beat in range(4):
                kind = (i + beat) % 5
                P = [int(x) for x in rng.integers(-(2 ** 23), 2 ** 23, N)]
                spread = [2, 8, 20, 40, 3][kind]                              # exponent spread inside the beat
                base = int(rng.integers(-60, 30))
                e = [base + int(x) for x in rng.integers(0, spread + 1, N)]
                if kind == 4:
                    P = [0] * N if beat == 0 else P                           # an all-zero beat
                if kind == 3:
                    P[0] = -P[1] if P[1] else 1                               # some cancellation
                    e[0] = e[1]
                v = reducer_ref(P, e)
                acc = fadd(acc, v)
                Pf.append("".join(f"{p & 0xFFFFFF:06x}" for p in reversed(P)))
                Ef.append("".join(f"{x & 0x3FF:03x}" for x in reversed(e)))
                V.append(f"{bits(v):08x}")
                ACC.append(f"{bits(acc):08x}")
        # e is packed 10 bits per group: write it as a bit-string-free hex by repacking
        def pack10(row):
            vals = [int(row[3 * j:3 * j + 3], 16) for j in range(N)][::-1]  # back to group order
            x = 0
            for j, v in enumerate(vals):
                x |= v << (10 * j)
            return f"{x:0{(N * 10 + 3) // 4}x}"
        open(os.path.join(outdir, "red_P.hex"), "w").write("\n".join(Pf) + "\n")
        open(os.path.join(outdir, "red_e.hex"), "w").write("\n".join(pack10(r) for r in Ef) + "\n")
        open(os.path.join(outdir, "red_v.hex"), "w").write("\n".join(V) + "\n")
        open(os.path.join(outdir, "red_acc.hex"), "w").write("\n".join(ACC) + "\n")
    print(what, n, "->", outdir)


if __name__ == "__main__":
    main()
