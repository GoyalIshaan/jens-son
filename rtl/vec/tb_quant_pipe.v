// act_quant32_pipe against act_quant32: random groups with a random clock enable, three enabled clocks apart.
module tb_quant_pipe;
    reg clk = 0, ce = 0; reg [1023:0] x; integer i, k, n = 0, bad = 0;
    wire [255:0] q0, q1; wire [9:0] e0, e1;
    act_quant32 ref_q(x, q0, e0);
    act_quant32_pipe dut(clk, ce, x, q1, e1);
    reg [265:0] hist [0:3];
    integer filled = 0;
    initial begin
        for (k = 0; k < 20000; k = k + 1) begin
            for (i = 0; i < 32; i = i + 1) begin
                x[32*i+:32] = $random;
                case (k % 5)
                    1: x[32*i+23+:8] = 8'd120 + ($random & 15);                  // one group scale
                    2: x[32*i+23+:8] = ($random & 3) ? 8'd127 : 8'd0;            // zeros and subnormals
                    3: x[32*i+:23] = 23'h7e0000 + ($random & 3) - 1;              // rounding boundary
                    default: ;
                endcase
            end
            ce = ($random & 3) != 0;
            #1;
            if (ce) begin
                if (filled >= 3 && {e1, q1} !== hist[(filled - 3) % 4]) bad = bad + 1;
                hist[filled % 4] = {e0, q0}; filled = filled + 1;
            end
            clk = 1; #1 clk = 0;
            if (ce && filled >= 4) n = n + 1;
        end
        // the check above compares the output after the edge with the input three enables earlier
        if (bad) $display("FAIL %0d of %0d", bad, n); else $display("PASS %0d groups", n);
        $finish;
    end
endmodule
