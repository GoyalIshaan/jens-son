// One result on every eighth rising edge, counting the sampling edge as one.
// No stalls inside the arithmetic pipeline. The caller tracks valid slots.
`include "vec_delay.vh"

module fp32_fma_pipe8(input wire clk, input wire [31:0] a, b, c, output wire [31:0] y);
    fp32_fma_pipe8_ce core(clk, 1'b1, a, b, c, y);
endmodule
