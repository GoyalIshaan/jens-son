`ifndef VEC_DELAY_VH
`define VEC_DELAY_VH
module vec_delay #(parameter WIDTH = 32, CYCLES = 1)
    (input wire clk, input wire [WIDTH-1:0] d, output wire [WIDTH-1:0] q);
    wire [WIDTH-1:0] taps [0:CYCLES];
    assign taps[0] = d;
    genvar i;
    for (i = 0; i < CYCLES; i = i + 1) begin : stage
        reg [WIDTH-1:0] value;
        always @(posedge clk) value <= taps[i];
        assign taps[i+1] = value;
    end
    assign q = taps[CYCLES];
endmodule

`endif
