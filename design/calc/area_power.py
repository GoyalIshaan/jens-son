"""Die area, power and unit cost of the chip, from the synthesized units (rtl/results_timing.json) and cited
figures. Rough, pre-layout: it answers "does the design fit the 80 mm2 die and the $150 card the goal assumed?"
(D-010, note 07), not "what will the layout be".

    python3 design/calc/area_power.py

Method (note 24): synthesized ASAP7 cell area / 0.70 (placed: 70% of the area holds cells, assumed) / 1.18 (N6 has
18% higher logic density than N7, TSMC via AnandTech), with ASAP7 standing in for N7. Every unit's area is its
synthesized per-lane (or per-row) area times the chip's count.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "..", "..", "rtl", "results_timing.json")))
PLACE, N6 = 0.70, 1.18
UM2_PER_MM2 = 1e6


def a(name):
    return T[name]["area_um2"] if name in T and T[name].get("area_um2") else None


# ---- logic: (what, unit in results_timing.json, count at chip size, note)
LOGIC = [
    ("matrix unit: 64 rows x 1,024 int8 lanes", "mxu_row_pipe", 64, "one row = 64 groups of 16 lanes + reducer"),
    ("vector FMA bank, 512 fp32 lanes", "fma8_ce", 512, ""),
    ("exp pool, 128 units", "exp_pool_lane", 128, ""),
    ("sigmoid pool, 64 units", "sigmoid9_lane", 64, ""),
    ("SiLU multiply, 64 FMAs", "fma8_ce", 64, ""),
    ("softplus, rsqrt, recip, 2 each", None, 1, "summed below"),
    ("QUANT/QKV/DEQ wide (512 lanes, 16 quantizers, 16 scale units)", "rows_wide", 1, "synthesized whole"),
    ("softmax tile (period 3)", "smtile_p3", 1, "control and tree around the shared bank"),
    ("softmax tile (1,024 positions)", "smtile_overlap", 1, ""),
    ("R1 reducer (16-lane cut, x32)", "reduce_rows16", 32, "lane count scaled"),
    ("DMA engines W, R, X, L", "dma_dram_in", 2, "W and R"),
    ("", "dma_sram_out", 1, "X"),
    ("", "dma_local", 1, "L (with the transpose tile)"),
    ("control, counters", "ctrl", 1, ""),
    ("", "perf", 1, ""),
]
SMALL_SFU = [("softplus9_lane", 2), ("rsqrt9_lane", 2), ("recip_lane", 2)]

# ---- on-chip memories (bits); in silicon they're SRAM macros
KB = 1024 * 8
MEMS = [
    ("main SRAM, 64 banks (D-012, D-028)", 6.5 * 1024 * KB),
    ("weight FIFO (D-023)", 192 * KB),
    ("DeltaNet state banks, 4 x 64 KiB (D-023)", 256 * KB),
    ("instruction memory, 8,192 x 32 B", 256 * KB),
    ("descriptor memory, 1,024 x 64 B", 64 * KB),
    ("matrix accumulator, 2 x 8,192 fp32 (D-029)", 2 * 8192 * 32),
    ("matrix activation buffers, 2 x 128 rows x 1 KB, + exponents", 2 * 128 * 8192 + 2 * 128 * 256),
]
BITCELL_UM2 = 0.027        # TSMC N7 high-density SRAM cell (note 27)
ARRAY_EFF = (0.60, 0.70)   # share of an SRAM macro that's bit cells (note 27; to be checked in layout)

# ---- hard IP (no public area; indirect estimates)
#   GDDR PHY: a 1,024-bit HBM2e PHY is about 10.45 mm2 at 7 nm (LLMCompass, arXiv 2312.03134, parameter table),
#   and a GDDR6 PHY needs 1.5-1.75x the area of HBM2 at the same bandwidth (Rambus, "Memory Systems for AI: Part 6").
#   HBM2e at 3.6 Gb/s x 1,024 = 460 GB/s; ours is 512 GB/s, so x 512/460.
GDDR_PHY = tuple(10.45 * k * 512 / 460 for k in (1.5, 1.75))
PCIE_PHY = (1.0, 3.0)       # PCIe Gen4 x4 PHY + controller: no public figure found; a guess, flagged as such
IO_MISC = (5.0, 10.0)       # pads, ESD, PLLs, sensors, test: a guess, flagged as such


def main():
    print("== Logic (synthesized on ASAP7, x count; mm2 of cells -> placed -> N6) ==")
    cells = 0.0
    for what, unit, n, note in LOGIC:
        if unit is None:
            ar = sum(a(u) * k for u, k in SMALL_SFU if a(u))
        else:
            ar = (a(unit) or 0) * n
        cells += ar
        flag = "" if (unit is None or a(unit)) else "  (not synthesized yet)"
        print(f"  {what or '':58s} {ar / UM2_PER_MM2:6.3f} mm2{flag}  {note}")
    placed = cells / PLACE
    logic_n6 = placed / N6
    print(f"  cells {cells / UM2_PER_MM2:.2f} mm2, placed {placed / UM2_PER_MM2:.2f} mm2, at N6 {logic_n6 / UM2_PER_MM2:.2f} mm2")

    print("\n== On-chip memories (N7 HD bit cell 0.027 um2, 60-70% of a macro is bit cells) ==")
    bits = sum(b for _, b in MEMS)
    for what, b in MEMS:
        print(f"  {what:58s} {b / 8 / 1024:8.0f} KB")
    mem = tuple(bits * BITCELL_UM2 / e / UM2_PER_MM2 for e in reversed(ARRAY_EFF))
    print(f"  total {bits / 8 / 2**20:.2f} MB: {mem[0]:.1f}-{mem[1]:.1f} mm2")

    print("\n== Hard IP and the rest (indirect estimates and guesses, see comments) ==")
    print(f"  GDDR7 PHY, 128-bit, 512 GB/s        {GDDR_PHY[0]:.1f}-{GDDR_PHY[1]:.1f} mm2  (indirect: HBM2e area x Rambus ratio)")
    print(f"  PCIe Gen4 x4                        {PCIE_PHY[0]:.1f}-{PCIE_PHY[1]:.1f} mm2  (guess)")
    print(f"  pads, ESD, PLLs, sensors, test      {IO_MISC[0]:.1f}-{IO_MISC[1]:.1f} mm2  (guess)")
    lo = logic_n6 / UM2_PER_MM2 + mem[0] + GDDR_PHY[0] + PCIE_PHY[0] + IO_MISC[0]
    hi = logic_n6 / UM2_PER_MM2 + mem[1] + GDDR_PHY[1] + PCIE_PHY[1] + IO_MISC[1]
    print(f"\n== Die: {lo:.0f}-{hi:.0f} mm2 at N6, against the 80 mm2 the cost model assumed ==")

    # ---- cost: the budget script's own model (wafer price, Murphy yield, 8 GB GDDR7 card)
    import sys
    sys.path.insert(0, HERE)
    import qwen35_budget as B
    for area in (lo, hi):
        die, dpw, yld = B.die_cost(area, "TSMC N7/N6")
        print(f"  {area:5.0f} mm2: {dpw} dies per wafer, yield {yld * 100:.0f}%, ${die:.2f} per good die")
    for mem_per_gb in (8.0, 12.0, 15.0):
        dies = [B.die_cost(x, "TSMC N7/N6")[0] for x in (lo, hi)]
        units = [8 * mem_per_gb + d + 15.0 + 35.0 for d in dies]
        print(f"  card at ${mem_per_gb:.0f}/GB GDDR7 (8 GB), package $15, board $35: ${units[0]:.0f}-{units[1]:.0f} per unit")

    # ---- power: energy per operation (Horowitz ISSCC 2014 at 45 nm, scaled), and the DRAM interface
    print("\n== Power (rough: energy per op x rate; leakage and clock tree +30%) ==")
    mac = (0.05, 0.08)              # pJ per int8 multiply-add at 6 nm (note 27, from Horowitz's 0.2 + 0.03 pJ at 45 nm)
    fma32 = (1.0, 1.5)              # pJ per fp32 FMA at 7 nm (Horowitz: 3.7 + 0.9 pJ at 45 nm, /3 to /4.5)
    sram_bit = (0.03, 0.06)         # pJ per SRAM bit read (Horowitz: 10 pJ per 64-bit read of 8 KB at 45 nm, /3 to /5)
    dram_bit = 7.5                  # pJ per bit, GDDR6 device level (Micron, via Tom's Hardware); GDDR7 similar or less
    macs = 64 * 1024 * 1e9          # matrix unit at full rate
    fmas = 512 * 1e9
    sram_bps = 1766 * 8 * 1e9       # prefill's measured average SRAM traffic (note 33, part 3)
    for name, util_m, util_v, dram_gbs in (("prefill (matrix busy ~70%)", 0.7, 0.5, 100), ("decode (memory bound)", 0.05, 0.05, 470)):
        chip = [1.3 * (macs * util_m * mac[i] + fmas * util_v * fma32[i] + sram_bps * sram_bit[i]) * 1e-12 for i in (0, 1)]
        dram = dram_gbs * 8e9 * dram_bit * 1e-12
        print(f"  {name:28s} chip {chip[0]:.1f}-{chip[1]:.1f} W, DRAM interface and devices about {dram:.0f} W")


if __name__ == "__main__":
    main()
