// Tests for fp32_fma (vec/fma.hex, from golden bitexact_vec.fma) and act_quant32 (golden/vectors/act_quant.hex).
`timescale 1ns/1ps
module tb_core;
    parameter NF = 20000, NQ = 128;
    reg  [31:0] fv [0:4*NF-1];
    reg  [31:0] a, b, c;
    wire [31:0] y;
    fp32_fma dut (.a(a), .b(b), .c(c), .y(y));
    reg  [31:0] qv [0:NQ*65-1];
    reg  [32*32-1:0] qx;
    wire [32*8-1:0] qq;
    wire [9:0] qe;
    act_quant32 dq (.x(qx), .q(qq), .e(qe));
    integer i, j, bad = 0, badq = 0;
    function isnan; input [31:0] v; isnan = (v[30:23] == 8'hff) && (v[22:0] != 0); endfunction
    initial begin
        $readmemh("vec/fma.hex", fv);
        for (i = 0; i < NF; i = i + 1) begin
            a = fv[4*i]; b = fv[4*i+1]; c = fv[4*i+2]; #1;
            if (!(y === fv[4*i+3] || (isnan(y) && isnan(fv[4*i+3])))) begin
                bad = bad + 1;
                if (bad <= 8) $display("FMA MISMATCH %0d: %h*%h+%h = %h, want %h", i, a, b, c, y, fv[4*i+3]);
            end
        end
        $display("fp32_fma: %0d vectors, %0d mismatches -> %s", NF, bad, bad == 0 ? "PASS" : "FAIL");
        $readmemh("vec/act_quant.hex", qv);
        for (i = 0; i < NQ; i = i + 1) begin
            for (j = 0; j < 32; j = j + 1) qx[32*j +: 32] = qv[65*i + j];
            #1;
            for (j = 0; j < 32; j = j + 1)
                if (qq[8*j +: 8] !== qv[65*i + 32 + j][7:0]) begin
                    badq = badq + 1;
                    if (badq <= 5) $display("QUANT MISMATCH group %0d lane %0d: %h want %h", i, j, qq[8*j +: 8], qv[65*i+32+j][7:0]);
                end
            if (qe !== qv[65*i + 64][9:0]) begin
                badq = badq + 1;
                if (badq <= 5) $display("QUANT EXP MISMATCH group %0d: %h want %h", i, qe, qv[65*i+64][9:0]);
            end
        end
        $display("act_quant32: %0d groups, %0d mismatches -> %s", NQ, badq, badq == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
