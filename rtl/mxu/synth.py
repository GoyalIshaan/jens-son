"""Synthesize mxu_group and its pieces with Yosys + ABC on the SkyWater 130 nm HD library (typical corner).

    python3 synth.py            # writes results/synth.json and prints a summary

For each target: cell area (um^2), NAND2-equivalent gates (area / 3.7536 um^2, the nand2_1 cell),
flip-flop share, and ABC's critical-path delay (no wire load: this is synthesis, not place and route).
Combinational pieces are mapped twice: area-oriented (map -D 100000) and delay-oriented (map -D 1).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, "..", "third_party")
LIBS = {
    # SkyWater 130 nm high-density, typical corner (25 C, 1.8 V)
    "sky130": dict(comb=[os.path.join(TP, "sky130hd", "sky130_fd_sc_hd__tt_025C_1v80.lib")],
                   seq=os.path.join(TP, "sky130hd", "sky130_fd_sc_hd__tt_025C_1v80.lib"),
                   nand2=3.7536, dont_use=["*lpflow*", "*probe*"]),
    # ASAP7 predictive 7 nm FinFET, RVT, typical corner (25 C, 0.7 V); area of NAND2xp33 = 0.08748 um^2
    "asap7": dict(comb=[os.path.join(TP, "asap7", f) for f in (
                      "asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib", "asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib",
                      "asap7sc7p5t_AO_RVT_TT_nldm_211120.lib", "asap7sc7p5t_OA_RVT_TT_nldm_211120.lib")],
                  seq=os.path.join(TP, "asap7", "asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib"),
                  nand2=0.08748, dont_use=["*x1p*_ASAP7*", "*ICG*", "*SDF*"]),
}
LIBNAME = os.environ.get("SYN_LIB", "sky130")
LIB = LIBS[LIBNAME]
WORK = os.environ.get("SYN_WORK", "/tmp/mxu_syn") + "_" + LIBNAME   # ABC cannot handle spaces in paths
NAND2 = LIB["nand2"]


def run(top, params=None, D=100000, files=("mxu_group.v", "syn_wrappers.v"), post=""):
    os.makedirs(WORK, exist_ok=True)
    for src in LIB["comb"] + [LIB["seq"]]:
        dst = os.path.join(WORK, os.path.basename(src))
        if not os.path.exists(dst):
            subprocess.run(["cp", src, dst], check=True)
    comb = " ".join(f"-liberty {os.path.basename(x)}" for x in LIB["comb"])
    seq = os.path.basename(LIB["seq"])
    du = " ".join(f"-dont_use {x}" for x in LIB["dont_use"])
    for f in files:
        subprocess.run(["cp", os.path.join(HERE, f), WORK], check=True)
    chp = " ".join(f"-set {k} {v}" for k, v in (params or {}).items())
    script = (f"read_verilog {' '.join(files)}; chparam {chp} {top}; " if params else f"read_verilog {' '.join(files)}; ") + (
        f"hierarchy -top {top}; synth -top {top} -flatten -booth; "
        f"dfflibmap -liberty {seq} {du}; "
        f"abc {comb} {du} -script +strash;dch,-f;map,-D,{D};topo;{post}stime,-p; "
        f"opt_clean; stat {comb} -liberty {seq}")
    log = os.path.join(WORK, f"{top}_{'_'.join(f'{k}{v}' for k, v in (params or {}).items())}_D{D}"
                             f"{'_buf' if post else ''}.log")
    r = subprocess.run(["yosys", "-l", log, "-p", script], cwd=WORK, capture_output=True, text=True)
    txt = open(log).read()
    if r.returncode != 0:
        raise RuntimeError(f"yosys failed for {top} {params}: see {log}")
    area = float(re.findall(r"Chip area for module .*?: ([0-9.]+)", txt)[-1])
    seq = re.findall(r"of which used for sequential elements: ([0-9.]+)", txt)
    delays = [float(x) for x in re.findall(r"Delay =\s*([0-9.]+) ps", txt)]
    cells = re.findall(r"^\s+(\d+)\s+[0-9.E+]+\s+cells$", txt, re.M)
    return dict(top=top, params=params or {}, D=D, area_um2=area, nand2_eq=area / NAND2,
                seq_um2=float(seq[-1]) if seq else 0.0, delay_ps=max(delays) if delays else None,
                cells=int(cells[-1]) if cells else None, log=log)


def main():
    out = []
    pieces = ["w_mult8", "w_lookup16", "adder_tree16", "w_scale_a", "w_scale_c", "w_scale_b"]
    for p in pieces:
        for D in (100000, 1):
            r = run(p, D=D)
            out.append(r)
            print(f"{p:14s} D={D:6d}: {r['area_um2']:9.1f} um2 = {r['nand2_eq']:7.0f} NAND2  delay {r['delay_ps']:7.0f} ps",
                  flush=True)
    rows = [int(x) for x in sys.argv[1:]] or [4, 8, 32, 64]
    for R in rows:
        for mode in ((1,) if R not in (4, 32) else (0, 1, 2)):
            r = run("mxu_group", {"R": R, "MODE": mode})
            out.append(r)
            print(f"mxu_group R={R:2d} MODE={mode}: {r['area_um2']:10.0f} um2 = {r['nand2_eq']:8.0f} NAND2, "
                  f"flip-flops {100 * r['seq_um2'] / r['area_um2']:4.1f}%, worst stage {r['delay_ps']:6.0f} ps, "
                  f"{r['cells']} cells", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    tag = "_".join(str(r) for r in rows)
    json.dump(dict(lib=LIBNAME, nand2_um2=NAND2, results=out),
              open(os.path.join(HERE, "results", f"synth_{LIBNAME}_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
