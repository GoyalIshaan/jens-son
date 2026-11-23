// fp32_fma_pipe8 with a clock enable: when ce is low every pipeline register holds, so the whole bank can
// freeze while a streaming command's result write is stalled (no result buffer needed). Same arithmetic,
// generated from fp32_fma_pipe8.v. fp32_fma_pipe8 is this module with ce tied high.
`include "vec_delay.vh"

module vec_delay_ce #(parameter WIDTH = 32, CYCLES = 1)
    (input wire clk, input wire ce, input wire [WIDTH-1:0] d, output wire [WIDTH-1:0] q);
    wire [WIDTH-1:0] taps [0:CYCLES];
    assign taps[0] = d;
    genvar i;
    for (i = 0; i < CYCLES; i = i + 1) begin : stage
        reg [WIDTH-1:0] value;
        always @(posedge clk) if (ce) value <= taps[i];
        assign taps[i+1] = value;
    end
    assign q = taps[CYCLES];
endmodule

module fp32_fma_pipe8_ce(input wire clk, input wire ce, input wire [31:0] a, b, c, output reg [31:0] y);
    // 1. Exact product and unbiased exponents in units of the least bit.
    wire special, ps, cs, cz;
    wire [31:0] special_value;
    // Exceptional products are overridden at retirement.
    wire [11:0] ah = {1'b1,a[22:12]}, bh = {1'b1,b[22:12]};
    reg [23:0] ll, lh, hl, hh;
    wire [23:0] mant;
    wire signed [12:0] ep, ec;
    fma_s1 unpack(a,b,c,special,special_value,,ps,cs,cz,mant,ep,ec);
    reg [47:0] p2;
    reg [23:0] m1, m2;
    reg signed [12:0] ep1, ec1;
    reg cz1;
    always @(posedge clk) if (ce) begin
        ll <= a[11:0] * b[11:0];
        lh <= a[11:0] * bh;
        hl <= ah * b[11:0];
        hh <= ah * bh;
        {m1,ep1,ec1,cz1} <= {mant,ep,ec,cz};
    end

    // 2. Classify the alignment once and narrow the shifter controls.
    wire signed [12:0] shift = ec1 - ep1 + 13'sd2;
    reg far2, right2, zero2, sticky_only2;
    reg [5:0] distance2;
    reg signed [12:0] weight2;
    // Compress the four quadrant products before the final carry propagation.
    wire [47:0] corners = {hh,ll};
    wire [47:0] cross1 = {12'd0,lh,12'd0}, cross2 = {12'd0,hl,12'd0};
    wire [47:0] carry = (corners & cross1) | ((corners ^ cross1) & cross2);
    always @(posedge clk) if (ce) begin
        p2 <= (corners ^ cross1 ^ cross2) + (carry << 1);
        m2 <= m1;
        zero2 <= cz1;
        far2 <= !cz1 && shift > 52;
        right2 <= shift < 0;
        sticky_only2 <= shift <= -24;
        distance2 <= shift < 0 ? -shift : shift;
        weight2 <= (!cz1 && shift > 52) ? ec1 - 13'sd52 : ep1 - 13'sd2;
    end

    // 3. Align magnitudes. Preserve all discarded bits in a sticky bit.
    reg [79:0] pf3, cf3;
    wire [23:0] right_value = m2 >> distance2[4:0];
    wire [23:0] discard_mask = (24'd1 << distance2[4:0]) - 1'b1;
    always @(posedge clk) if (ce) begin
        pf3 <= far2 ? 80'd1 : {30'd0,p2,2'b00};
        if (zero2) cf3 <= 0;
        else if (far2) cf3 <= {4'd0,m2,52'd0};
        else if (sticky_only2) cf3 <= 1;
        else if (right2) cf3 <= right_value | (|(m2 & discard_mask));
        else cf3 <= {56'd0,m2} << distance2;
    end

    // 4. Use the product sign as the common sign, so subtraction needs
    // only one complement and a carry-in on the same adder.
    wire opposite3, ps4;
    vec_delay_ce #(1,3) align_sign(clk, ce, ps ^ cs, opposite3);
    vec_delay_ce #(1,4) product_sign(clk, ce, ps, ps4);
    reg [80:0] sum4;
    always @(posedge clk) if (ce)
        sum4 <= {1'b0,pf3} + ({1'b0,cf3} ^ {81{opposite3}}) + opposite3;

    // 5. Convert the signed relative sum back to magnitude.
    reg [79:0] mag5;
    reg neg5;
    always @(posedge clk) if (ce) begin
        mag5 <= sum4[80] ? -sum4[79:0] : sum4[79:0];
        neg5 <= ps4 ^ sum4[80];
    end

    // 6. Locate the highest nonempty byte in parallel with each
    // byte's leading-bit encoder. The seventh stage shifts only once.
    wire [9:0] occupied;
    wire [2:0] bit_in_byte [0:9];
    genvar j;
    for (j = 0; j < 10; j = j + 1) begin : leading
        wire [7:0] v = mag5[j*8 +: 8];
        assign occupied[j] = |v;
        assign bit_in_byte[j] = v[7] ? 7 : v[6] ? 6 : v[5] ? 5 :
                                v[4] ? 4 : v[3] ? 3 : v[2] ? 2 : v[1] ? 1 : 0;
    end
    integer k;
    reg [3:0] byte_index;
    always @* begin
        byte_index = 0;
        for (k=0; k<10; k=k+1) if (occupied[k]) byte_index = k;
    end
    reg [6:0] lead6;
    reg [79:0] mag6, norm7;
    reg zero6;
    always @(posedge clk) if (ce) begin
        mag6 <= mag5;
        lead6 <= {byte_index, bit_in_byte[byte_index]};
        zero6 <= ~|occupied;
    end

    // 7. Normalize. 8. Round exactly once and apply special cases.
    wire signed [12:0] weight6;
    vec_delay_ce #(13,4) weight_delay(clk, ce, weight2, weight6);
    reg signed [12:0] exponent7;
    always @(posedge clk) if (ce) begin
        norm7 <= mag6 << (7'd79 - lead6);
        exponent7 <= weight6 + $signed({6'd0,lead6}) + 13'sd127;
    end
    wire zero7, neg7, special7;
    wire [31:0] sy7, rounded;
    vec_delay_ce #(1,1) zero_delay(clk, ce, zero6, zero7);
    vec_delay_ce #(1,2) sign_delay(clk, ce, neg5, neg7);
    vec_delay_ce #(33,7) exceptions(clk, ce, {special,special_value}, {special7,sy7});
    fma_s4 round_result(norm7,exponent7,neg7,zero7,rounded);
    always @(posedge clk) if (ce) y <= special7 ? sy7 : rounded;
endmodule
