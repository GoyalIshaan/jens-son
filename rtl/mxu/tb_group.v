// Testbench for mxu_group: loads each stationary block, streams its 8 beats back to back,
// and compares every output bit against the Python reference (ref_group.py).
`timescale 1ns/1ps
module tb_group;
    parameter R = 4;
    parameter MODE = 1;
    parameter NB = 200;           // stationary blocks
    localparam BEATS = 8;
    localparam NV = NB * BEATS;

    reg clk = 0;
    always #1 clk = ~clk;

    reg  [16*8-1:0]   s_tbl [0:NB-1];
    reg  [R*16*8-1:0] s_act [0:NB-1];
    reg  [R*32-1:0]   s_as  [0:NB-1];
    reg  [16*4-1:0]   b_codes [0:NV-1];
    reg  [7:0]        b_ws    [0:NV-1];
    reg  [R*32-1:0]   e_y     [0:NV-1];
    reg  [R*24-1:0]   e_mant  [0:NV-1];
    reg  [R*10-1:0]   e_exp   [0:NV-1];

    reg               table_we = 0, act_we = 0, in_valid = 0;
    reg  [16*8-1:0]   table_in;
    reg  [R*16*8-1:0] act_in;
    reg  [R*32-1:0]   as_in;
    reg  [16*4-1:0]   codes;
    reg  [7:0]        ws;
    wire              out_valid;
    wire [R*32-1:0]   y;
    wire [R*24-1:0]   y_mant;
    wire [R*10-1:0]   y_exp;

    mxu_group #(.R(R), .MODE(MODE)) dut (
        .clk(clk), .table_we(table_we), .table_in(table_in), .act_we(act_we), .act_in(act_in), .as_in(as_in),
        .in_valid(in_valid), .codes(codes), .ws(ws), .out_valid(out_valid), .y(y), .y_mant(y_mant), .y_exp(y_exp));

    integer b, j, n_out = 0, n_bad = 0;
    // outputs are checked in order as they come out of the 3-stage pipeline
    always @(posedge clk) if (out_valid) begin
        if (MODE == 2 ? (y_mant !== e_mant[n_out] || y_exp !== e_exp[n_out]) : (y !== e_y[n_out])) begin
            n_bad = n_bad + 1;
            if (n_bad <= 5) begin
                if (MODE == 2) $display("MISMATCH vec %0d: got %h/%h want %h/%h", n_out, y_mant, y_exp, e_mant[n_out], e_exp[n_out]);
                else           $display("MISMATCH vec %0d: got %h want %h", n_out, y, e_y[n_out]);
            end
        end
        n_out = n_out + 1;
    end

    initial begin
        $readmemh("vec/stat0.hex", s_tbl);
        $readmemh("vec/stat1.hex", s_act);
        $readmemh("vec/stat2.hex", s_as);
        $readmemh("vec/beat0.hex", b_codes);
        $readmemh("vec/beat1.hex", b_ws);
        if (MODE == 2) begin
            $readmemh("vec/exp0.hex", e_mant);
            $readmemh("vec/exp1.hex", e_exp);
        end else
            $readmemh("vec/exp0.hex", e_y);
        for (b = 0; b < NB; b = b + 1) begin
            @(negedge clk);
            in_valid = 0; table_we = 1; act_we = 1;
            table_in = s_tbl[b]; act_in = s_act[b]; as_in = s_as[b];
            @(negedge clk);
            table_we = 0; act_we = 0;
            for (j = 0; j < BEATS; j = j + 1) begin
                in_valid = 1; codes = b_codes[b*BEATS + j]; ws = b_ws[b*BEATS + j];
                @(negedge clk);
            end
            in_valid = 0;
            repeat (4) @(negedge clk);       // drain before the stationary state changes
        end
        repeat (4) @(negedge clk);
        $display("R=%0d MODE=%0d: %0d vectors checked, %0d mismatches -> %s", R, MODE, n_out, n_bad,
                 (n_bad == 0 && n_out == NV) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
