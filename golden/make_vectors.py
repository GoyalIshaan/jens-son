"""Golden v2 test vectors (progression step 4). Run: python3 golden/make_vectors.py
Writes golden/vectors/: one line per vector, hex fp32 bit patterns.
  sfu_<f>.hex   input -> output, for exp, recip, rsqrt, sigmoid, softplus (chip_sfu.py), incl. edge cases
  act_quant.hex 32 fp32 inputs -> 32 int8 values and the exponent (bitexact.quant_act_exact)
  mm_beats.hex  one output over 4 beats: 4,096 int8 activations + 128 exponents, 4,096 table entries,
                256 E4M3 scales, the tensor scale -> fp32 result (bitexact.mm_exact)
  dn_token.hex, smtile_row.hex   rules R4, R7 (bitexact_vec.py)
  conv4.hex     R5: per case 3 history rows + 8 new rows + weights [C, 4] (C = 64) -> 8 outputs + new history
  rope.hex      R6: position | 2 heads x 256 inputs | 32 c, 32 s | 2 heads x 256 outputs (theta 1e7)
  rope_tab.hex  R6's two tables: 1,024 rows x (cos, sin of p*1024*f) then 1,024 rows x (cos, sin of p*f), 32 each
  solve_r8.hex  R8: An [64, 64] (negated, strictly lower) | r [64, 128] -> Delta [64, 128]
"""
import os

import numpy as np
import torch

import bitexact as B
import chip_sfu as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")


def h32(t):
    return [f"{int(v):08x}" for v in t.float().contiguous().view(torch.int32).numpy().astype(np.uint32)]


def main(n=4096, seed=7):
    os.makedirs(OUT, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g) * 2 - 1
    edges = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0 ** -126, 88.0, -87.0, -104.0, 1e-8, -1e-8, 30.0, -30.0])
    ins = {"exp": torch.cat([edges, u * 90]), "recip": torch.cat([edges[2:], torch.exp(u * 40) * torch.sign(u)]),
           "rsqrt": torch.cat([edges[2:5], torch.exp(u * 40)]), "sigmoid": torch.cat([edges, u * 30]),
           "softplus": torch.cat([edges, u * 30])}
    for f, x in ins.items():
        y = getattr(S, f)(x)
        with open(os.path.join(OUT, f"sfu_{f}.hex"), "w") as fh:
            fh.write("\n".join(f"{a} {b}" for a, b in zip(h32(x), h32(y))) + "\n")
    # activation quantization
    x = torch.randn(n // 32, 32, generator=g) * torch.exp(torch.randn(n // 32, 1, generator=g) * 4)
    q, e = B.quant_act_exact(x)
    with open(os.path.join(OUT, "act_quant.hex"), "w") as fh:
        for i in range(x.shape[0]):
            fh.write(" ".join(h32(x[i])) + " | " + " ".join(f"{int(v) & 0xFF:02x}" for v in q[i]) +
                     f" | {int(e[i]) & 0x3FF:03x}\n")
    # matrix beats through the real weight format
    import formats
    w = torch.randn(64, 4096, generator=g) * 0.02
    qt = formats.quantize_weight(w, "lut4_g16_fp8_t8")
    k, M, E, ts = B.weight_parts(qt)
    xa = torch.randn(8, 4096, generator=g)
    xa[:, ::61] *= 40
    qa, ea = B.quant_act_exact(xa)
    y = B.mm_exact(qa, ea, k, M, E, ts)
    s8 = (qt.scales / qt.gmax).to(torch.float8_e4m3fn).view(torch.uint8)
    with open(os.path.join(OUT, "mm_beats.hex"), "w") as fh:
        for t in range(8):
            for o in range(64):
                fh.write("".join(f"{int(v) & 0xFF:02x}" for v in qa[t]) + " " +
                         "".join(f"{int(v) & 0x3FF:03x}" for v in ea[t]) + " " +
                         "".join(f"{int(v) & 0xFF:02x}" for v in k[o]) + " " +
                         "".join(f"{int(v):02x}" for v in s8[o].flatten()) + " " +
                         h32(ts.reshape(1))[0] + " " + h32(y[t, o].reshape(1))[0] + "\n")
    # DeltaNet head step (one token) and fused softmax tile rows, in the order fixed by bitexact_vec.py
    import bitexact_vec as BV
    with open(os.path.join(OUT, "dn_token.hex"), "w") as fh:
        for case in range(8):
            Sm = torch.randn(128, 128, generator=g) * 0.1
            k = torch.nn.functional.normalize(torch.randn(128, generator=g), dim=0)
            q = torch.nn.functional.normalize(torch.randn(128, generator=g), dim=0) * 0.0884
            v = torch.randn(128, generator=g)
            beta = torch.rand(1, generator=g)[0]
            gg = -torch.rand(1, generator=g)[0] * 0.5
            Sn, o = BV.dn_token(Sm, k, q, v, beta, gg)
            fh.write(" ".join(h32(Sm.flatten())) + " | " + " ".join(h32(torch.cat([k, q, v, beta.reshape(1), gg.reshape(1)])))
                     + " | " + " ".join(h32(Sn.flatten())) + " | " + " ".join(h32(o)) + "\n")
    with open(os.path.join(OUT, "smtile_row.hex"), "w") as fh:
        for case in range(64):
            sc = torch.randn(1024, generator=g) * 4
            ks = torch.rand(1024, generator=g) * 0.02 + 0.001
            vs = torch.rand(1024, generator=g) * 0.02 + 0.001
            m = torch.tensor(float("-inf")) if case % 4 == 0 else torch.randn(1, generator=g)[0] * 0.1
            l = torch.tensor(0.0) if case % 4 == 0 else torch.rand(1, generator=g)[0] * 50
            mask = torch.arange(1024) > (1023 - case * 8)
            qv, e, mn, ln, al = BV.smtile_row(sc, ks, vs, m, l, mask)
            fh.write(" ".join(h32(torch.cat([sc, ks, vs, m.reshape(1), l.reshape(1)]))) + f" | {1023 - case * 8} | "
                     + "".join(f"{int(x) & 0xFF:02x}" for x in qv) + " " + "".join(f"{int(x) & 0x3FF:03x}" for x in e)
                     + " | " + " ".join(h32(torch.stack([mn, ln, al]))) + "\n")
    # R5 conv, R6 RoPE, R8 solve (added 2026-11-21; after the others so their bits don't change)
    with open(os.path.join(OUT, "conv4.hex"), "w") as fh:
        for case in range(16):
            hist = torch.randn(3, 64, generator=g) * 2
            x = torch.randn(8, 64, generator=g) * 2
            w = torch.randn(64, 4, generator=g) * 0.4
            y, hn = BV.conv4(x, hist, w)
            fh.write(" ".join(h32(hist.flatten())) + " | " + " ".join(h32(x.flatten())) + " | " +
                     " ".join(h32(w.flatten())) + " | " + " ".join(h32(y.flatten())) + " | " +
                     " ".join(h32(hn.flatten())) + "\n")
    tb = BV.rope_tables()
    with open(os.path.join(OUT, "rope_tab.hex"), "w") as fh:
        for a, b in ((tb[0], tb[1]), (tb[2], tb[3])):
            for i in range(1024):
                fh.write(" ".join(h32(a[i])) + " " + " ".join(h32(b[i])) + "\n")
    pos = torch.cat([torch.tensor([0, 1, 1023, 1024, 1025, 32767, 32768, 262143]),
                     torch.randint(0, 262144, (56,), generator=g)])
    c, s_ = BV.rope_cs(pos, tb)
    with open(os.path.join(OUT, "rope.hex"), "w") as fh:
        for i in range(pos.shape[0]):
            x = torch.randn(1, 2, 256, generator=g) * 3
            y = BV.rope(x, c[i:i + 1], s_[i:i + 1])
            fh.write(f"{int(pos[i]):05x} | " + " ".join(h32(x.flatten())) + " | " + " ".join(h32(c[i])) + " " +
                     " ".join(h32(s_[i])) + " | " + " ".join(h32(y.flatten())) + "\n")
    with open(os.path.join(OUT, "solve_r8.hex"), "w") as fh:
        for case in range(4):
            An = -torch.tril(torch.randn(64, 64, generator=g) * 0.08, -1)
            r = torch.randn(64, 128, generator=g)
            d = BV.solve_r8(An, r)
            fh.write(" ".join(h32(An.flatten())) + " | " + " ".join(h32(r.flatten())) + " | " +
                     " ".join(h32(d.flatten())) + "\n")
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
