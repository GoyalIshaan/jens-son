// reducer_p2_seg against the Python reference: seg4 = 0 must equal reducer_p2; seg4 = 1 gives 4 segment sums.
`timescale 1ns/1ps
module tb_reducer_seg;
    parameter NB = 3000;
    reg  [64*24-1:0] rP [0:NB-1];
    reg  [64*10-1:0] rE [0:NB-1];
    reg  [0:0]       rS [0:NB-1];
    reg  [127:0]     rV [0:NB-1];
    reg  [64*24-1:0] P;
    reg  [64*10-1:0] e;
    reg              seg4;
    wire [31:0] v0, v1, v2, v3;
    reducer_p2_seg #(.F(16)) dut (.P(P), .e(e), .seg4(seg4), .v0(v0), .v1(v1), .v2(v2), .v3(v3));
    integer i, bad = 0;
    initial begin
        $readmemh("vec_s/P.hex", rP); $readmemh("vec_s/e.hex", rE); $readmemh("vec_s/s.hex", rS);
        $readmemh("vec_s/v.hex", rV);
        for (i = 0; i < NB; i = i + 1) begin
            P = rP[i]; e = rE[i]; seg4 = rS[i]; #1;
            if (v0 !== rV[i][31:0] || (seg4 && ({v3, v2, v1} !== rV[i][127:32]))) begin
                bad = bad + 1;
                if (bad <= 5) $display("MISMATCH %0d seg4=%0d: %h %h %h %h want %h", i, seg4, v3, v2, v1, v0, rV[i]);
            end
        end
        $display("reducer_p2_seg: %0d beats (half in 4-segment mode), %0d mismatches -> %s", NB, bad,
                 bad == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
