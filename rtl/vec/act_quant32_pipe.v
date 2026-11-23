// act_quant32 in three pipeline stages, for 1 GHz (the one-cycle version is 1.6 ns at ASAP7, two stages 1.1).
// Same arithmetic, so the same codes and exponent, three enabled clocks later:
//   stage 1: each lane's biased exponent proposal
//   stage 2: their maximum (the group scale)
//   stage 3: each lane's shift by (scale - exponent), round to nearest even with sticky, sign
module act_quant32_pipe (
    input wire clk, ce,
    input wire [1023:0] x,
    output reg [255:0] q,
    output reg [9:0] e
);
    wire [8:0] scale_tree [1:63];
    genvar i;
    generate
        for (i = 1; i < 32; i = i + 1) begin : reduce_scale
            assign scale_tree[i] = scale_tree[2*i] > scale_tree[2*i+1] ? scale_tree[2*i] : scale_tree[2*i+1];
        end
        for (i = 0; i < 32; i = i + 1) begin : propose
            wire [31:0] v = x[32*i +: 32];
            reg [8:0] proposal;
            always @(posedge clk) if (ce) proposal <= {1'b0, v[30:23]} + (v[22:0] > 23'h7e0000);
            assign scale_tree[32+i] = proposal;
        end
    endgenerate
    reg [1023:0] x0, x1; reg [8:0] scale;
    always @(posedge clk) if (ce) begin x0 <= x; x1 <= x0; scale <= scale_tree[1]; end
    wire [255:0] q_next;
    generate
        for (i = 0; i < 32; i = i + 1) begin : lane
            wire [31:0] v = x1[32*i +: 32];
            wire [7:0] exp_field = v[30:23];
            wire [8:0] gap = scale - exp_field;
            wire [8:0] head = {1'b1, v[22:15]};
            wire [8:0] shifted = head >> gap[2:0];
            wire [8:0] lost_mask = (9'd1 << gap[2:0]) - 1'b1;
            wire sticky = (|v[14:0]) || (|(head & lost_mask)) || shifted[0];
            wire [7:0] mag = {1'b0, shifted[8:2]} + (shifted[1] && (sticky || shifted[2]));
            wire [7:0] code = (scale < 7 || exp_field == 0 || gap >= 8) ? 8'd0 : mag;
            assign q_next[8*i +: 8] = v[31] ? -code : code;
        end
    endgenerate
    always @(posedge clk) if (ce) begin
        q <= q_next;
        e <= scale < 7 ? -10'sd126 : {1'b0, scale} - 10'd133;
    end
endmodule
