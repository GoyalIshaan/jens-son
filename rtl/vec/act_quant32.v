// D-019. Each lane proposes a biased group exponent. Taking the maximum
// of those proposals is equivalent to finding amax and then scaling it.
module act_quant32 (
    input wire [1023:0] x,
    output wire [255:0] q,
    output wire [9:0] e
);
    wire [8:0] scale_tree [1:63];
    wire [8:0] scale = scale_tree[1];
    assign e = scale < 7 ? -10'sd126 : {1'b0, scale} - 10'd133;
    genvar i;
    generate
        for (i = 1; i < 32; i = i + 1) begin : reduce_scale
            assign scale_tree[i] = scale_tree[2*i] > scale_tree[2*i+1]
                                 ? scale_tree[2*i] : scale_tree[2*i+1];
        end
        for (i = 0; i < 32; i = i + 1) begin : lane
            wire [31:0] v = x[32*i +: 32];
            wire [7:0] exp_field = v[30:23];
            assign scale_tree[32+i] = {1'b0, exp_field} + (v[22:0] > 23'h7e0000);

            // At gap zero the integer is the leading seven bits. Preserve
            // its guard bit and OR the rest into sticky before shifting.
            wire [8:0] gap = scale - exp_field;
            wire [8:0] head = {1'b1, v[22:15]};
            wire [8:0] shifted = head >> gap[2:0];
            wire [8:0] lost_mask = (9'd1 << gap[2:0]) - 1'b1;
            wire sticky = (|v[14:0]) || (|(head & lost_mask)) || shifted[0];
            wire [7:0] mag = {1'b0, shifted[8:2]} +
                              (shifted[1] && (sticky || shifted[2]));
            wire [7:0] code = (scale < 7 || exp_field == 0 || gap >= 8) ? 8'd0 : mag;
            assign q[8*i +: 8] = v[31] ? -code : code;
        end
    endgenerate
endmodule
