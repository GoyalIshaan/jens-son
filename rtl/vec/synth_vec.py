"""Synthesize the vector-unit blocks with Yosys + ABC on ASAP7 (7 nm predictive, RVT, TT), as rtl/mxu/synth.py
does for the matrix unit. Combinational blocks; area-mapped (D 100000) and delay-mapped (D 1).

    python3 rtl/vec/synth_vec.py [top ...]      # writes rtl/vec/results/synth_asap7_vec.json
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mxu"))
os.environ.setdefault("SYN_LIB", "asap7")
import synth as SY  # noqa: E402

LIB = SY.LIBS["asap7"]
WORK = "/tmp/vec_syn_asap7"
FILES = ["fp32_fma.v", "fp32_fma_pipe.v", "fp32_fma_pipe8.v", "vec_alu.v", "act_quant32.v", "sfu.v", "reduce_tree.v"]
TOPS = ["fp32_fma", "vec_alu", "act_quant32", "sfu_exp", "sfu_recip", "sfu_rsqrt", "sfu_sigmoid", "sfu_softplus"]


def run(top, D):
    os.makedirs(WORK, exist_ok=True)
    for src in LIB["comb"] + [LIB["seq"]]:
        dst = os.path.join(WORK, os.path.basename(src))
        if not os.path.exists(dst):
            subprocess.run(["cp", src, dst], check=True)
    for f in FILES + ["sfu_tables.vh", "vec_delay.vh"]:
        subprocess.run(["cp", os.path.join(HERE, f), WORK], check=True)
    comb = " ".join(f"-liberty {os.path.basename(x)}" for x in LIB["comb"])
    du = " ".join(f"-dont_use {x}" for x in LIB["dont_use"])
    if os.environ.get("SYN_KS"):            # Kogge-Stone carry networks instead of Yosys' default ripple carry
        flow = (f"synth -top {top} -flatten -booth -run :fine; "
                f"techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top {top} -run fine:; ")
    elif top == "act_quant32":
        # All 32 shifts are needed together. SAT-based resource sharing
        # spends minutes proving lane pairs cannot share hardware.
        flow = f"synth -top {top} -flatten -booth -noshare; "
    else:
        flow = f"synth -top {top} -flatten -booth; "
    script = (f"read_verilog -sv {' '.join(FILES)}; hierarchy -top {top}; {flow}"
              f"dfflibmap -liberty {os.path.basename(LIB['seq'])} {du}; "
              f"abc {comb} {du} -script +strash;dch,-f;map,-D,{D};topo;stime,-p; opt_clean; "
              f"stat {comb} -liberty {os.path.basename(LIB['seq'])}")
    log = os.path.join(WORK, f"{top}_D{D}.log")
    r = subprocess.run(["yosys", "-q", "-l", log, "-p", script], cwd=WORK, capture_output=True, text=True)
    txt = open(log).read()
    if r.returncode != 0:
        raise RuntimeError(f"yosys failed for {top}: {r.stderr[-2000:]}")
    area = float(re.findall(r"Chip area for module .*?: ([0-9.]+)", txt)[-1])
    delays = [float(x) for x in re.findall(r"Delay =\s*([0-9.]+) ps", txt)]
    return dict(area_um2=area, nand2_eq=round(area / LIB["nand2"]), delay_ps=max(delays) if delays else None)


if __name__ == "__main__":
    tops = sys.argv[1:] or TOPS
    out_p = os.path.join(HERE, "results", os.environ.get("SYN_OUT", "synth_asap7_vec.json"))
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for t in tops:
        for D in (100000, 1):
            r = run(t, D)
            res[f"{t} D{D}"] = r
            print(t, "area-mapped" if D > 1 else "delay-mapped", r, flush=True)
            json.dump(res, open(out_p, "w"), indent=1)
