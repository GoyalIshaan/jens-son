"""1 GHz timing survey: synthesize each chip unit with Yosys + ABC on ASAP7 (7 nm predictive, RVT, TT) and
report its longest combinational path between flip-flops (and from inputs / to outputs), delay-mapped, with
Kogge-Stone carry networks (Yosys' default ripple carry is slower than any production flow) and ABC buffering
and gate sizing.
Lane counts are cut down (the per-lane logic, which sets the path, is the same at full width). No wires:
pre-layout numbers, as in notes 24 and 32. A unit fits 1 GHz if its worst path is under 1,000 ps.

    python3 rtl/synth_timing.py [name ...]      # writes rtl/results_timing.json and prints a line per unit
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mxu"))
os.environ.setdefault("SYN_LIB", "asap7")
import synth as SY  # noqa: E402

LIB = SY.LIBS["asap7"]
WORK = "/tmp/chip_timing_asap7"                    # ABC cannot handle spaces in paths
V, C, M = "vec", "chip", "mxu"
FMA = [f"{V}/fp32_fma_pipe8_ce.v", f"{V}/fp32_fma_pipe.v", f"{V}/fp32_fma.v", f"{V}/fp32_fma_pipe8.v"]
SFU = [f"{V}/sfu.v"] + FMA
UNITS = {
    # name: (top, files, params)
    "fma8_ce":        ("fp32_fma_pipe8_ce", FMA, {}),
    "exp_pool_lane":  ("sfu_exp_core_ce", SFU, {"LAT": 8}),
    "sigmoid_lane":   ("sfu_sigmoid_core_ce", SFU, {"LAT": 8}),
    "rsqrt_lane":     ("sfu_rsqrt_core_ce", SFU, {"LAT": 8}),
    "sigmoid9_lane":  ("sfu_sigmoid_core_ce", SFU, {"LAT": 9}),
    "rsqrt9_lane":    ("sfu_rsqrt_core_ce", SFU, {"LAT": 9}),
    "softplus9_lane": ("sfu_softplus_core_ce", SFU, {"LAT": 9}),
    "recip_lane":     ("sfu_recip_core_ce", SFU, {"LAT": 8}),
    "softplus_lane":  ("sfu_softplus_core_ce", SFU, {"LAT": 8}),
    "act_quant32":    ("act_quant32", [f"{V}/act_quant32.v"], {}),
    "act_quant32_pipe": ("act_quant32_pipe", [f"{V}/act_quant32_pipe.v"], {}),
    "box_addr":       ("vec_box_addr", [f"{V}/vec_box.sv"], {}),
    "kv_scale":       ("vec_kv_scale", [f"{V}/vec_formats.sv"], {}),        # QKV: one clock, smax -> scale
    "kv_scale_pipe":  ("vec_kv_scale_pipe", [f"{V}/vec_formats.sv"], {}),   # 3 stages
    "rows_wide":      ("vec_rows_wide", [f"{V}/vec_row_ops.sv", f"{V}/vec_formats.sv", f"{V}/act_quant32_pipe.v"] + FMA, {}),
    "smtile_p3":      ("vec_smtile_p3", [f"{V}/vec_smtile_p3.sv", f"{V}/vec_row_ops.sv", f"{V}/act_quant32_pipe.v"] + FMA, {}),
    "smtile_overlap": ("vec_smtile_overlap", [f"{V}/vec_smtile_overlap.sv", f"{V}/vec_row_ops.sv", f"{V}/act_quant32_pipe.v"] + FMA, {}),
    # the matrix unit's control and pipeline at small memories (in silicon the accumulator and activation
    # buffers are SRAM macros; as flip-flop arrays at full size they don't synthesize)
    "mxu_small":      ("chip_mxu", [f"{C}/chip_mxu.sv", f"{C}/mxu_row_pipe.sv", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"] + FMA,
                       {"RP": 4, "MAXR": 16, "ACCW": 256, "ER": 2}),
    "ctrl":           ("chip_ctrl", [f"{C}/chip_ctrl.sv"], {"HIN": 64, "PB": 64}),
    "perf":           ("chip_perf", [f"{C}/chip_perf.sv"], {}),
    "wfifo_small":    ("chip_wfifo", [f"{C}/chip_wfifo.sv"], {"SIZE": 4096, "PB": 64}),
    "reduce_rows16":  ("vec_reduce_rows", [f"{V}/vec_reduce_rows.sv", f"{V}/vec_row_ops.sv", f"{V}/vec_formats.sv"] + FMA,
                       {"LANES": 16}),
    "box_command16":  ("vec_box_command", [f"{V}/vec_box_command.sv", f"{V}/vec_box.sv"] + FMA,
                       {"LANES": 16, "LOCAL_BANK": 0}),
    "dma_dram_in":    ("chip_dma", [f"{C}/chip_dma.sv"], {"KIND": 1, "OUTST": 16, "PB": 64}),
    "dma_sram_out":   ("chip_dma", [f"{C}/chip_dma.sv"], {"KIND": 2, "OUTST": 16, "PB": 64}),
    "dma_local":      ("chip_dma", [f"{C}/chip_dma.sv"], {"KIND": 3, "OUTST": 16, "PB": 64}),
    "core":           ("chip_core", [f"{C}/chip_core.sv"], {"IMEM": 64, "DMEM": 64, "QD": 4}),
    "mxu_row":        ("mxu_row", [f"{C}/chip_mxu.sv", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"] + FMA, {}),
    "mxu_row_pipe":   ("mxu_row_pipe", [f"{C}/mxu_row_pipe.sv", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_w2f":      ("probe_w2f", [f"{C}/mxu_stage_probe.v", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_grp":      ("probe_grp", [f"{C}/mxu_stage_probe.v", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_align": ("probe_align", [f"{C}/mxu_stage_probe.v", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_add3": ("probe_add3", [f"{C}/mxu_stage_probe.v", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_sm": ("probe_sm", [f"{C}/mxu_stage_probe.v", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_row_nce": ("probe_row_nce", [f"{C}/mxu_stage_probe.v", f"{C}/mxu_row_pipe.sv", f"{M}/mxu_group.v", f"{M}/beat_reducer.v"], {}),
    "probe_max": ("probe_max", [f"{C}/mxu_stage_probe.v"], {}),
    "mxu_group_ref":  ("mxu_group", [f"{M}/mxu_group.v"], {}),
}


# units whose SystemVerilog (unpacked array ports) Yosys can't read: converted to Verilog-2005 with sv2v first
SV2V = {"box_command16", "core"}


def run(name):
    top, files, params = UNITS[name]
    work = os.path.join(WORK, name)
    os.makedirs(work, exist_ok=True)
    for src in LIB["comb"] + [LIB["seq"]]:
        dst = os.path.join(work, os.path.basename(src))
        if not os.path.exists(dst):
            subprocess.run(["cp", src, dst], check=True)
    for f in files + [f"{V}/sfu_tables.vh", f"{V}/vec_delay.vh", f"{C}/desc_fields.vh"]:
        if os.path.exists(os.path.join(HERE, f)):
            subprocess.run(["cp", os.path.join(HERE, f), work], check=True)
    names = [os.path.basename(f) for f in files]
    if name in SV2V:
        subprocess.run(["sv2v", "-DSYNTHESIS", f"-I{work}", "-w", os.path.join(work, "sv2v_out.v")]
                       + [os.path.join(work, n) for n in names], check=True)
        names = ["sv2v_out.v"]
    comb = " ".join(f"-liberty {os.path.basename(x)}" for x in LIB["comb"])
    seq = os.path.basename(LIB["seq"])
    du = " ".join(f"-dont_use {x}" for x in LIB["dont_use"])
    chp = " ".join(f"-set {k} {v}" for k, v in params.items())
    script = (f"read_verilog -sv -DSYNTHESIS {' '.join(names)}; "
              + (f"chparam {chp} {top}; " if params else "")
              + f"hierarchy -top {top}; synth -top {top} -flatten -booth -noshare -run :fine; "
              f"techmap -map +/techmap.v -map +/choices/kogge-stone.v; synth -top {top} -noshare -run fine:; "
              f"dfflibmap -liberty {seq} {du}; "
              f"abc {comb} {du} -script +strash;dch,-f;map,-D,1;topo;buffer,-p;upsize,-D,1;dnsize,-D,1;stime,-p; opt_clean; "
              f"stat {comb} -liberty {seq}")
    log = os.path.join(work, "yosys.log")
    r = subprocess.run(["yosys", "-q", "-l", log, "-p", script], cwd=work, capture_output=True, text=True)
    txt = open(log).read() if os.path.exists(log) else ""
    if r.returncode != 0:
        return dict(error=(r.stderr or txt)[-600:])
    area = re.findall(r"Chip area for module .*?: ([0-9.]+)", txt)
    delays = [float(x) for x in re.findall(r"Delay =\s*([0-9.]+) ps", txt)]
    ffs = re.findall(r"of which used for sequential elements: ([0-9.]+)", txt)
    return dict(delay_ps=max(delays) if delays else None, area_um2=float(area[-1]) if area else None,
                seq_um2=float(ffs[-1]) if ffs else None, params=params)


if __name__ == "__main__":
    names = sys.argv[1:] or list(UNITS)
    out_p = os.path.join(HERE, "results_timing.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    with ThreadPoolExecutor(int(os.environ.get("JOBS", "3"))) as ex:
        for name, r in zip(names, ex.map(run, names)):
            res[name] = r
            d = r.get("delay_ps")
            flag = "" if d is None else ("  fits 1 GHz" if d < 1000 else "  OVER 1 ns")
            if "error" in r:
                print(f"{name:16s} error: {r['error'][-300:]}", flush=True)
            else:
                print(f"{name:16s} {d:8.0f} ps  area {r['area_um2']:.0f} um2{flag}", flush=True)
            json.dump(res, open(out_p, "w"), indent=1)
