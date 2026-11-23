// vec_reduce (512 lanes) against golden reduce_sum (rule R1): sums and sums of squares of 64..4,096 values.
`timescale 1ns/1ps
module tb_reduce;
    localparam N = 512;
    reg [31:0] w [0:40000];
    reg clk = 0, en = 0, first = 0, last = 0, square = 0;
    reg [N*32-1:0] x;
    wire [31:0] y; wire yv;
    vec_reduce #(.N(N)) dut (.clk(clk), .en(en), .first(first), .last(last), .square(square), .x(x), .y(y), .y_valid(yv));
    integer p = 0, cs, beats, bt, j, bad = 0; reg [31:0] want;
    initial begin
        $readmemh("vec/reduce.hex", w);
        for (cs = 0; cs < 10; cs = cs + 1) begin
            beats = w[p]; square = w[p+1][0]; want = w[p+2]; p = p + 3;
            for (bt = 0; bt < beats; bt = bt + 1) begin
                for (j = 0; j < N; j = j + 1) x[32*j +: 32] = w[p + j];
                p = p + N;
                en = 1; first = (bt == 0); last = (bt == beats - 1);
                #1 clk = 1; #1 clk = 0;
            end
            en = 0;
            if (y !== want) begin bad = bad + 1; $display("REDUCE MISMATCH case %0d: %h want %h", cs, y, want); end
        end
        $display("vec_reduce (512 lanes + pairwise tree): 10 cases, %0d mismatches -> %s", bad, bad == 0 ? "PASS" : "FAIL");
        $finish;
    end
endmodule
