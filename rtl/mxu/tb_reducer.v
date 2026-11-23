// Tests for beat_reducer.v: (1) fp32_add alone, (2) reducer_p2 + row_acc over 4-beat outputs.
`timescale 1ns/1ps
module tb_reducer;
    parameter NADD = 6000;
    parameter NRED = 500;            // outputs; 4 beats each
    localparam NB = NRED * 4;

    // ---------------- fp32_add
    reg  [31:0] ta [0:NADD-1], tb [0:NADD-1], ty [0:NADD-1];
    reg  [31:0] a, b;
    wire [31:0] y;
    fp32_add dut_add (.a(a), .b(b), .y(y));

    // ---------------- reducer + accumulator
    reg  [64*24-1:0] rP [0:NB-1];
    reg  [64*10-1:0] rE [0:NB-1];
    reg  [31:0]      rV [0:NB-1], rA [0:NB-1];
    reg  [64*24-1:0] P;
    reg  [64*10-1:0] e;
    wire [31:0]      v, acc;
    reg              clk = 0, clear = 0, en = 0;
    reducer_p2 #(.N(64), .F(16)) dut_red (.P(P), .e(e), .v(v));
    row_acc dut_acc (.clk(clk), .clear(clear), .en(en), .v(v), .acc(acc));

    integer i, bad_add = 0, bad_v = 0, bad_acc = 0;
    initial begin
        $readmemh("vec_r/add_a.hex", ta); $readmemh("vec_r/add_b.hex", tb); $readmemh("vec_r/add_y.hex", ty);
        for (i = 0; i < NADD; i = i + 1) begin
            a = ta[i]; b = tb[i]; #1;
            if (y !== ty[i]) begin
                bad_add = bad_add + 1;
                if (bad_add <= 5) $display("ADD MISMATCH %0d: %h + %h = %h, want %h", i, a, b, y, ty[i]);
            end
        end
        $display("fp32_add: %0d vectors, %0d mismatches -> %s", NADD, bad_add, bad_add == 0 ? "PASS" : "FAIL");

        $readmemh("vec_r/red_P.hex", rP); $readmemh("vec_r/red_e.hex", rE);
        $readmemh("vec_r/red_v.hex", rV); $readmemh("vec_r/red_acc.hex", rA);
        for (i = 0; i < NB; i = i + 1) begin
            P = rP[i]; e = rE[i]; clear = (i % 4 == 0); en = 1; #1;
            if (v !== rV[i]) begin
                bad_v = bad_v + 1;
                if (bad_v <= 5) $display("REDUCE MISMATCH beat %0d: %h want %h", i, v, rV[i]);
            end
            clk = 1; #1; clk = 0;
            if (acc !== rA[i]) begin
                bad_acc = bad_acc + 1;
                if (bad_acc <= 5) $display("ACC MISMATCH beat %0d: %h want %h", i, acc, rA[i]);
            end
        end
        $display("reducer_p2: %0d beats, %0d mismatches; accumulator: %0d mismatches -> %s", NB, bad_v, bad_acc,
                 (bad_v == 0 && bad_acc == 0) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
