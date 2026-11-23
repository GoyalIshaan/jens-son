`timescale 1ns/1ps
module tb_alu;
    reg [31:0] v [0:5*4000-1]; reg [31:0] s [0:2*4007-1];
    reg [2:0] op; reg neg; reg [31:0] a, b, c, x; wire [31:0] y, ys;
    vec_alu dut (.op(op), .neg(neg), .a(a), .b(b), .c(c), .y(y));
    sfu_silu ds (.x(x), .y(ys));
    integer i, bad = 0, bads = 0;
    function isnan; input [31:0] t; isnan = (t[30:23] == 8'hff) && (t[22:0] != 0); endfunction
    initial begin
        $readmemh("vec/alu.hex", v);
        for (i = 0; i < 4000; i = i + 1) begin
            op = v[5*i][2:0]; neg = v[5*i][3]; a = v[5*i+1]; b = v[5*i+2]; c = v[5*i+3]; #1;
            if (!(y === v[5*i+4] || (isnan(y) && isnan(v[5*i+4])))) begin
                bad = bad + 1; if (bad <= 6) $display("ALU MISMATCH %0d op %0d neg %0d: %h %h %h -> %h want %h", i, op, neg, a, b, c, y, v[5*i+4]);
            end
        end
        $display("vec_alu (ADD SUB MUL FMA MAX, neg): 4000 vectors, %0d mismatches -> %s", bad, bad == 0 ? "PASS" : "FAIL");
        $readmemh("vec/silu.hex", s);
        for (i = 0; i < 4007; i = i + 1) begin
            x = s[2*i]; #1;
            if (ys !== s[2*i+1]) begin bads = bads + 1; if (bads <= 6) $display("SILU MISMATCH %h -> %h want %h", x, ys, s[2*i+1]); end
        end
        $display("silu: 4007 inputs, %0d mismatches -> %s", bads, bads == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
