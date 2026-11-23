#!/bin/sh
# Run from any directory. Keep each invocation's binaries and logs isolated.
set -eu
cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
python3 gen_tables.py >/dev/null

testbench() {
    top=$1
    shift
    iverilog -g2012 -s "$top" -o "$work/sim" "$@"
    if ! vvp "$work/sim" >"$work/log" 2>&1; then
        cat "$work/log"
        return 1
    fi
    cat "$work/log"
    if grep -Eq 'FAIL|MISMATCH|FATAL|ERROR' "$work/log"; then return 1; fi
    grep -q PASS "$work/log"
}
testbench tb_core tb_core.v fp32_fma.v act_quant32.v
testbench tb_alu tb_alu.v vec_alu.v sfu.v fp32_fma*.v
testbench tb_sfu tb_sfu.v sfu.v fp32_fma*.v
for stages in 4 8; do
    testbench tb_fma_pipe -Ptb_fma_pipe.LAT="$stages" tb_fma_pipe.v fp32_fma_pipe*.v
done
for func in 0 1 2 3 4 5; do
    testbench tb_sfu_pipe -Ptb_sfu_pipe.FUNC="$func" tb_sfu_pipe.v sfu.v fp32_fma*.v
done
testbench tb_reduce tb_reduce.v reduce_tree.v fp32_fma.v
testbench tb_reduce_pipe tb_reduce_pipe.sv vec_reduce_pipe.sv fp32_fma_pipe*.v
testbench tb_box_command tb_box_command.sv vec_box*.sv fp32_fma_pipe*.v
testbench tb_vector_exec tb_vector_exec.sv vector_exec.sv vec_*.sv act_quant32.v act_quant32_pipe.v sfu.v fp32_fma*.v
testbench tb_dn_interleave tb_dn_interleave.sv vec_dn_group.sv vec_dn_head.sv
testbench tb_dn_head tb_dn_head.sv vec_dn_head.sv sfu.v fp32_fma*.v
# The 64 full softmax tiles also run in Icarus. Set VEC_FAST_SIM=1 to
# compile that test with Verilator on a configured C++ toolchain.
if [ "${VEC_FAST_SIM:-0}" = 1 ]; then
    verilator --binary --timing -j 3 -Wno-fatal --Mdir "$work/verilator" \
        --top-module tb_smtile tb_smtile.sv vec_smtile.sv sfu.v act_quant32.v act_quant32_pipe.v fp32_fma*.v
    "$work/verilator/Vtb_smtile"
else
    testbench tb_smtile tb_smtile.sv vec_smtile.sv sfu.v act_quant32.v act_quant32_pipe.v fp32_fma*.v
fi
python3 test_vec_unit.py
