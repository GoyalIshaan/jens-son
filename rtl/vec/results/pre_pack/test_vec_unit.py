"""Compare decoded vector commands at the byte-addressed SRAM port against FSim.
Generate disposable fixtures so expected values follow the actual golden arithmetic.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "isa"))
from fsim_v1 import FSim, NP
from isa_v1 import Opnd, Vec, V_OPS

TYPES = ["f32", "f16", "i8", "u32", "e8"]
SIZE = 512 * 1024


def fixtures(folder, dn_only=False, tile_only=False, rows_only=False, red_only=False):
    rng = np.random.default_rng(19)
    mem = np.zeros(SIZE, np.uint8)
    cursor = 1024
    checks = []

    def buffer(arr):
        nonlocal cursor
        raw = np.ascontiguousarray(arr).reshape(-1).view(np.uint8)
        addr = cursor
        mem[addr:addr + len(raw)] = raw
        cursor += (len(raw) + 63) // 64 * 64
        return addr

    def operand(a, dtype="f32"):
        a = np.asarray(a, dtype=NP[dtype])
        n = a.shape[-1]
        m = a.shape[-2] if a.ndim >= 2 else 1
        stride = a.itemsize
        return Opnd(buffer(a), stride, stride * n, stride * m * n, dtype)

    def case(op, d, rv=(0, 0, 0, 0, 0)):
        checks.append((op, d, rv))

    x = operand(rng.uniform(0.25, 3, (2, 2, 64)))
    b = operand(rng.normal(size=(2, 2, 64)))
    c = operand(rng.normal(size=(2, 2, 64)))
    dst = operand(np.zeros((2, 2, 64)))
    for op in ["ADD", "SUB", "MUL", "MAX", "FMA", "COPY", "EXP", "SIGMOID",
               "SILU", "SOFTPLUS", "RSQRT", "RECIP"]:
        # Positive rsqrt arguments; other SFUs exercise affine pre-processing.
        y, z = (None, None) if op in ("COPY", "RSQRT") else (b, c)
        columns = 33 if op in ("EXP", "SIGMOID", "SILU", "SOFTPLUS", "RSQRT", "RECIP") else 64
        rows = 1 if columns == 33 else 2
        case(op, Vec(columns, rows, rows, dst, x, y, z))
    # One row spans two physical 512-lane beats and ends with only one valid lane.
    a = operand(rng.normal(size=(2, 513)))
    res = operand(np.zeros((2, 1)))
    for op in ("RSUM", "RSUMSQ", "RMAX"):
        case(op, Vec(513, 2, 1, res, a))
    qi = operand(np.zeros((2, 2, 64)), "i8")
    ex = operand(np.zeros((2, 2, 2)), "e8")
    scale = operand(np.zeros((2, 2, 1)), "f16")
    case("QUANT", Vec(64, 2, 2, qi, x, b, ex, (("exp_bias", 2),)))
    case("MUL", Vec(64, 2, 2, dst, x, ex))
    case("QKV", Vec(64, 2, 2, qi, x, None, scale))
    half = operand(rng.normal(size=(2, 2, 2)), "f16")
    case("DEQ", Vec(64, 2, 2, dst, qi, half))
    u = operand(np.arange(17), "u32")
    t = np.arange(17); t[11] = 999
    case("MATCH", Vec(17, dst=operand([0], "u32"), a=u, b=operand(t, "u32")))
    case("CAUSAL", Vec(64, 2, 2, dst, x, par=(("step", 3),)), (0, 0, 20, 34, 0))
    angle = operand(rng.normal(size=(2,1,64)))
    case("ROPE",Vec(64,2,2,dst,dst,angle,par=(("rot",64),)))
    hist=operand(rng.normal(size=(3,2,64)))
    weights=Opnd(buffer(rng.normal(size=(2,64,4)).astype(np.float32)),16,64*16,0)
    win=buffer(np.zeros((5,2,64),np.float32))
    case("CONV4",Vec(64,2,2,dst,x,weights,hist,(("window",win),)))
    # R7 on two rows, with a full beat and a partial one. Check the write
    # column and exponent addresses on both beats, including -inf * zero.
    ns, nr = 192, 2
    sc = rng.normal(size=(nr, ns)); sc[:, -10:] = -np.inf
    ks = rng.uniform(.01, .1, size=ns);ks[-10:] = 0
    ksbase = buffer(np.r_[ks, rng.uniform(.01, .1, size=ns)].astype(np.float16))
    scales = Opnd(ksbase, 2, 0, 0, "f16")
    running = np.zeros((6, nr),np.float32);running[0] = [-np.inf,1.];running[-1] = [0,50.]
    old = Opnd(buffer(running), 0, 4, 0)
    prob = operand(np.zeros((nr,ns)), "i8")
    exp_stride = ns // 32 + 2
    exp_dst=buffer(np.zeros((nr,exp_stride),np.int8))
    case("SMTILE", Vec(ns,nr,1,prob,operand(sc),scales,old,(("vs",2*ns),("mlb",4*nr),("e",(exp_dst-prob.base,exp_stride)))))
    # Raw signed exponents, uint32 rounding, and negated reductions.
    raw_exp = operand([-128, -126, -3, 0, 3, 126, 127], "e8")
    case("COPY", Vec(7, dst=dst, a=raw_exp))
    case("COPY", Vec(7, dst=raw_exp, a=dst))
    ids = operand([0, 127, 2**24+1, 2**24+3, 2**31+256], "u32")
    case("COPY", Vec(5, dst=dst, a=ids))
    case("COPY", Vec(5, dst=ids, a=dst))
    case("RSUM", Vec(513, 2, 1, res, a, par=(("neg",1),)))
    case("RMAX", Vec(513, 2, 1, res, a, par=(("neg",1),)))
    # R4 with non-zero value-head index, replay, then two tentative tokens.
    dim, heads = 128, 2
    w = 2*dim+heads*dim+2*heads
    def token_rows(t):
        vals = rng.normal(0, .05, (t,w)).astype(np.float32)
        vals[:, -2*heads:] = rng.uniform(.1,.9,(t,2*heads))
        return Opnd(buffer(vals),4,0,w*4)
    rows,logs=token_rows(2),token_rows(1)
    state=operand(rng.normal(0,.1,(dim,dim)))
    work=buffer(np.zeros((dim,dim),np.float32))
    out=Opnd(buffer(np.zeros((2,heads,dim),np.float32)),4,0,4*heads*dim)
    case("DN", Vec(dim,1,2,out,rows,logs,state,(("T",2),("work",work),("commit_spec",0),("lv",heads),("lqk",1))),
         (0,1,0,0,1))
    # Four different resident states, including a group starting at a nonzero
    # value head that crosses shared-key boundaries. Compare each command
    # with four ordinary one-head golden updates and the grouped FSim.
    heads, keys = 8, 4
    w = (2*keys+heads)*dim+2*heads
    rows, logs = token_rows(2), token_rows(1)
    state = operand(rng.normal(0, .1, (4, dim, dim)))
    out = operand(np.zeros((2, heads, dim)))
    params = (("heads",4),("work",0),("commit_spec",1),("lv",heads),("lqk",keys))
    for tokens, replay in ((0,1), (2,0)):
        case("DN", Vec(dim,dim,1,out,rows,logs,state,params+(("T",tokens),)), (0,1,0,0,replay))
    if dn_only:
        checks = [item for item in checks if item[0]=="DN"]
    if tile_only:
        checks.clear()
        cursor = 1024
        mem.fill(0)
        profiles = []
        for width, rows, planes, ready in ((256, 28, 1, 0), (256, 32, 1, 1),
                                            (1024, 4, 1, 1), (1024, 15, 1, 0),
                                            (192, 3, 2, 0)):
            fast = planes == 1
            scores = rng.normal(0, 4, (planes, rows, width)).astype(np.float32)
            k = rng.uniform(0.02, 1.8, width)
            v = rng.uniform(0.01, 0.7, width)
            k[::17] = 0
            scores[..., ::17] = -np.inf
            layout = operand(scores)
            if not fast:
                layout = replace(layout, base=layout.base+4*(width-1), sn=-4)
            scales = Opnd(buffer(np.concatenate((k, v)).astype(np.float16)), 2, 0, 0, "f16")
            row_slots = rows*planes + int(rng.integers(0, 4))
            stats = np.zeros((6, row_slots), dtype=np.float32)
            stats[0] = rng.normal(0, 4, row_slots)
            stats[0, ::4] = -np.inf
            stats[5] = rng.uniform(0, 40, row_slots)
            stats[5, ::4] = 0
            old = Opnd(buffer(stats), 0, 4, 4*rows)
            code_stride = width + int(rng.integers(0, 16))
            codes = Opnd(buffer(np.zeros((planes, rows, code_stride), dtype=np.int8)),
                         1, code_stride, rows*code_stride, "i8")
            exp_stride = width//32 + int(rng.integers(0, 4))
            exp_addr = buffer(np.zeros((rows, exp_stride), dtype=np.int8))
            case("SMTILE", Vec(width, rows, planes, codes, layout, scales, old,
                               (("vs", 2*width), ("mlb", 4*row_slots),
                                ("e", (exp_addr-codes.base, exp_stride)))))
            profiles.append(f"{int(fast)} {ready}\n")
        (folder / "controls.txt").write_text("".join(profiles))
    if rows_only:
        # Row ops streamed across rows and planes: short rows (one sub-beat), rows that end in a partial
        # beat, and full-width rows, each under random read/write stalls; then ready-port rate cases.
        checks.clear()
        cursor = 1024
        mem.fill(0)
        profiles = []
        def rowcase(op, n, rows, planes, ready):
            x = rng.uniform(0.25, 3, (planes, rows, n))
            if op == "DEQ":
                qa = operand(rng.integers(-127, 128, (planes, rows, n)), "i8")
                sc = operand(rng.uniform(0.01, 2, (planes, rows, n // 32)), "f16")
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, n))), qa, sc))
            elif op == "QKV":
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, n)), "i8"), operand(x * rng.normal(size=x.shape)),
                             None, operand(np.zeros((planes, rows, 1)), "f16")))
            elif op == "RMAX":
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, 1))), operand(rng.normal(size=x.shape)),
                             par=(("neg", 1),)))
            elif op == "MATCH":
                u = rng.integers(0, 5, (planes, rows, n)); t = u.copy()
                for r in range(rows): t[0, r, (7 * r + 3) % n] += 1 if r % 3 else 0
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, 1)), "u32"), operand(u, "u32"),
                             operand(t, "u32")))
            elif op == "QUANT":
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, n)), "i8"), operand(x),
                             operand(rng.normal(size=(planes, rows, n))),
                             operand(np.zeros((planes, rows, n // 32)), "e8"), (("exp_bias", 2),)))
            else:
                y, z = (None, None) if op == "RSQRT" else (operand(rng.normal(size=(planes, rows, n))),
                                                           operand(rng.normal(size=(planes, rows, n))))
                case(op, Vec(n, rows, planes, operand(np.zeros((planes, rows, n))), operand(x), y, z))
            profiles.append(f"0 {ready}\n")
        for op, n in (("EXP", 300), ("SIGMOID", 200), ("SILU", 129), ("SOFTPLUS", 5), ("RSQRT", 3),
                      ("RECIP", 7), ("QUANT", 256), ("DEQ", 96)):
            rowcase(op, n, 3, 2, 0)
        for op, n, rows, planes in (("QKV", 256, 3, 2), ("RMAX", 300, 3, 2), ("MATCH", 40, 1, 1)):
            rowcase(op, n, rows, planes, 0)
        rowcase("QKV", 256, 16, 1, 1)
        for op, n, rows in (("EXP", 1024, 8), ("SILU", 1024, 8), ("RSQRT", 2, 32), ("QUANT", 1024, 8),
                            ("DEQ", 1024, 8)):
            rowcase(op, n, rows, 1, 1)
        (folder / "controls.txt").write_text("".join(profiles))
    if red_only:
        # R1 with several rows in flight: ragged widths (one valid lane in the last beat), two planes,
        # negated sums, random stalls; then ready-port rate cases at the 9B hidden width.
        checks.clear()
        cursor = 1024
        mem.fill(0)
        profiles = []
        def redcase(op, n, rows, planes, ready, neg=0):
            a = operand(rng.normal(size=(planes, rows, n)))
            res = operand(np.zeros((planes, rows, 1)))
            case(op, Vec(n, rows, planes, res, a, par=(("neg", neg),) if neg else ()))
            profiles.append(f"0 {ready}\n")
        redcase("RSUMSQ", 513, 5, 2, 0)
        redcase("RSUM", 100, 7, 1, 0, 1)
        redcase("RSUMSQ", 1024, 3, 1, 0)
        redcase("RSUMSQ", 4096, 1, 1, 1)
        redcase("RSUMSQ", 4096, 16, 1, 1)
        redcase("RSUM", 512, 9, 1, 1)
        (folder / "controls.txt").write_text("".join(profiles))
    assert cursor <= SIZE
    np.savetxt(folder/"start.hex",mem[:cursor],fmt="%02x")
    sim=FSim.__new__(FSim);sim.sram=mem
    with (folder/"cases.txt").open("w") as cases, (folder/"expected.txt").open("w") as expected:
        for op,d,rv in checks:
            before=mem.copy()
            par=dict(d.par)
            n,m,p=d.n,d.m,d.p
            fields=[V_OPS.index(op),n,m,p,int(d.b is not None),int(d.c is not None),par.get("neg",0),
                    par.get("exp_bias",0),par.get("rot",0),rv[2],rv[3],par.get("step",1),
                    rv[1],rv[1]//(par.get("lv",1)//par.get("lqk",1)),par.get("lqk",1),par.get("lv",1),
                    rv[4],par.get("T",1),par.get("commit_spec",1),par.get("work",0),
                    par.get("vs",0),*par.get("e",(0,0)),par.get("mlb",0)]
            for idx,operand_ in enumerate([d.dst,d.a,d.b,d.c]):
                o=operand_ or Opnd(0)
                off=rv[idx] if op!="DN" or idx!=1 else 0
                if op=="CAUSAL" and idx in (2,3): off=0
                fields += [o.base+off,o.sn,o.sm,o.sp,TYPES.index(o.dtype)]
            for land in sim._vec(SimpleNamespace(d=d,rv=rv,op=op)):land()
            if par.get("heads", 1)>1:
                serial=FSim.__new__(FSim)
                serial.sram=before.copy()
                serial_par=tuple((k,v) for k,v in d.par if k!="heads")
                for slot in range(par["heads"]):
                    offsets=(rv[0],rv[1]+slot,rv[2],rv[3]+slot*dim*d.c.sm,rv[4])
                    cmd=SimpleNamespace(d=replace(d,par=serial_par),rv=offsets,op=op)
                    for land in serial._vec(cmd): land()
                np.testing.assert_array_equal(mem,serial.sram)
            if op=="DN" and not par["commit_spec"] and par["T"]:
                # The scalar oracle clones its tentative state internally. Run
                # the same math in-place on a private image to check SRAM scratch.
                shadow=FSim.__new__(FSim)
                shadow.sram=before.copy()
                inplace=replace(d,par=tuple(dict(par,commit_spec=1).items()))
                cmd=SimpleNamespace(d=inplace,rv=rv,op=op)
                for land in shadow._vec(cmd): land()
                layout=dict(shape=(dim,dim),dtype=np.float32,strides=(d.c.sm,4))
                source=np.ndarray(buffer=shadow.sram,offset=d.c.base+rv[3],**layout)
                np.ndarray(buffer=mem,offset=par["work"],**layout)[:]=source
            fields += [int("window" in par),par.get("window",0),par.get("heads",1)-1]
            print(*fields,file=cases)
            changed=np.where(before!=mem)[0]
            print(len(changed),file=expected)
            for addr in changed: print(addr,int(mem[addr]),file=expected)
    return cursor,len(checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument("--dn-only",action="store_true")
    selection.add_argument("--smtile-only",action="store_true")
    selection.add_argument("--rows-only",action="store_true")
    selection.add_argument("--red-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="vec_unit_") as tmp:
        folder=Path(tmp)
        size,count=fixtures(folder,args.dn_only,args.smtile_only,args.rows_only,args.red_only)
        sources=[*HERE.glob("vec_*.sv"),HERE/"vector_exec.sv",HERE/"sfu.v",HERE/"act_quant32.v",*HERE.glob("fp32_fma*.v")]
        options=["-Ptb_vec_unit.BUFFERED=1"] if args.smtile_only else []
        subprocess.run(["iverilog","-g2012","-s","tb_vec_unit",*options,"-o",str(folder/"sim"),
                        "tb_vec_unit.sv",*map(str,sources)],cwd=HERE,check=True)
        args_vvp=["+tile_checks"] if args.smtile_only or args.rows_only or args.red_only else []
        subprocess.run(["vvp",str(folder/"sim"),f"+bytes={size}",f"+cases={count}",*args_vvp],
                       cwd=folder,check=True)

if __name__=="__main__":
    main()
