// The 4-stage FMA streams one vector per cycle; each result must appear 4 cycles later, bit-identical.
`timescale 1ns/1ps
module tb_fma_pipe;
    parameter NF = 20000, LAT = 4;
    reg [31:0] fv [0:4*NF-1];
    reg clk = 0; reg [31:0] a = 0, b = 0, c = 0; wire [31:0] y;
    if (LAT == 8) fp32_fma_pipe8 dut (.clk(clk), .a(a), .b(b), .c(c), .y(y));
    else fp32_fma_pipe dut (.clk(clk), .a(a), .b(b), .c(c), .y(y));
    integer i, bad = 0;
    function isnan; input [31:0] t; isnan = (t[30:23] == 8'hff) && (t[22:0] != 0); endfunction
    initial begin
        $readmemh("vec/fma.hex", fv);
        for (i = 0; i < NF + LAT; i = i + 1) begin
            if (i < NF) begin a = fv[4*i]; b = fv[4*i+1]; c = fv[4*i+2]; end
            #1 clk = 1; #1 clk = 0;
            if (i >= LAT - 1 && i - (LAT - 1) < NF) begin
                if (!(y === fv[4*(i-LAT+1)+3] || (isnan(y) && isnan(fv[4*(i-LAT+1)+3])))) begin
                    bad = bad + 1;
                    if (bad <= 5) $display("PIPE MISMATCH %0d: %h want %h", i - LAT + 1, y, fv[4*(i-LAT+1)+3]);
                end
            end
        end
        $display("fp32_fma_pipe (%0d stages, 1 per cycle): %0d vectors, %0d mismatches -> %s", LAT, NF, bad, bad == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
