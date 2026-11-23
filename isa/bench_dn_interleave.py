"""Compare one and four resident heads at fixed 9B workload and queue settings.

Run python3 isa/bench_dn_interleave.py to print the timing JSON.
"""
import json

from qwen_v1 import build
from run_v1 import timed
from sim_v1 import SimChip, dn_group_cycles


def main():
    context = 32768
    chip = SimChip(vec_lanes=512)
    widths = {}
    for heads in (1, 4):
        row = {
            "normal_token_ready_sram": dn_group_cycles(heads, False),
            "replay_token_ready_sram": dn_group_cycles(heads, True),
        }
        for mode, emitted in (("decode", 1), ("spec", 4)):
            time_ns, _ = timed(mode, context, a=3, chip=chip, fused=True, dn_heads=heads)
            row[mode + "_tok_s"] = emitted * 1e9 / time_ns
        widths[heads] = row
    memory = {}
    for mode in ("decode", "spec"):
        _, info = build(mode, fused=True)
        memory[mode] = info["sram"].total()
    print(json.dumps(dict(context=context, spec_accepted=3, fused=True, chip_clock_hz=chip.clk,
                          vec_lanes=chip.vec_lanes, sram_bytes=memory, by_group_width=widths), indent=2))


if __name__ == "__main__":
    main()
