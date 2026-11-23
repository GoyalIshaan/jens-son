// Stand-alone pieces of mxu_group, so each can be synthesized and timed on its own.

// one lane multiply: int8 x int8 -> 16-bit
module w_mult8 (input wire signed [7:0] a, input wire signed [7:0] b, output wire signed [15:0] p);
    assign p = a * b;
endmodule

// the table lookup for 16 lanes (shared by all rows of a group): 16 x (16-to-1 mux of 8 bits)
module w_lookup16 (input wire [16*8-1:0] tbl, input wire [16*4-1:0] codes, output wire [16*8-1:0] w);
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : l
        assign w[i*8 +: 8] = tbl[codes[i*4 +: 4]*8 +: 8];
    end endgenerate
endmodule

// scale step for one row, in each mode (see mxu_group.v)
module w_scale_a (input wire signed [18:0] S, input wire [7:0] ws, input wire [31:0] as, output wire [31:0] y);
    wire [3:0] M = (ws[6:3] != 0) ? {1'b1, ws[2:0]} : {1'b0, ws[2:0]};
    wire signed [9:0] E = (ws[6:3] != 0) ? ($signed({6'd0, ws[6:3]}) - 10'sd10) : -10'sd9;
    wire [31:0] ws_f, S_f, t;
    small_to_fp32 cw (.s(ws[7]), .m(M), .e(E), .f(ws_f));
    int_to_fp32 #(.W(19)) cs (.x(S), .f(S_f));
    fp32_mul m1 (.a(ws_f), .b(as), .shift(10'sd0), .y(t));
    fp32_mul m2 (.a(S_f), .b(t), .shift(10'sd0), .y(y));
endmodule

module w_scale_c (input wire signed [18:0] S, input wire [7:0] ws, input wire [31:0] as, output wire [31:0] y);
    wire [3:0] M = (ws[6:3] != 0) ? {1'b1, ws[2:0]} : {1'b0, ws[2:0]};
    wire signed [9:0] E = (ws[6:3] != 0) ? ($signed({6'd0, ws[6:3]}) - 10'sd10) : -10'sd9;
    wire signed [23:0] SM = S * $signed({1'b0, M});
    wire signed [23:0] P = ws[7] ? -SM : SM;
    wire [31:0] P_f;
    int_to_fp32 #(.W(24)) cp (.x(P), .f(P_f));
    fp32_mul m1 (.a(P_f), .b(as), .shift(E), .y(y));
endmodule

module w_scale_b (input wire signed [18:0] S, input wire [7:0] ws, input wire [7:0] k,
                  output wire signed [23:0] P, output wire signed [9:0] e);
    wire [3:0] M = (ws[6:3] != 0) ? {1'b1, ws[2:0]} : {1'b0, ws[2:0]};
    wire signed [9:0] E = (ws[6:3] != 0) ? ($signed({6'd0, ws[6:3]}) - 10'sd10) : -10'sd9;
    wire signed [23:0] SM = S * $signed({1'b0, M});
    assign P = ws[7] ? -SM : SM;
    assign e = E + $signed({{2{k[7]}}, k});
endmodule

module w_scale_d (input wire signed [18:0] S, input wire [7:0] ws, input wire [15:0] as, output wire [31:0] y);
    wire [3:0] M = (ws[6:3] != 0) ? {1'b1, ws[2:0]} : {1'b0, ws[2:0]};
    wire signed [9:0] E = (ws[6:3] != 0) ? ($signed({6'd0, ws[6:3]}) - 10'sd10) : -10'sd9;
    wire signed [23:0] SM = S * $signed({1'b0, M});
    wire signed [23:0] P = ws[7] ? -SM : SM;
    wire [31:0] P_f;
    int_to_fp32 #(.W(24)) cp (.x(P), .f(P_f));
    fp32_mul_b8 m1 (.a(P_f), .b(as), .shift(E), .y(y));
endmodule

// one row's reducer + accumulator, as synthesized
module w_reduce_p2 (input wire [64*24-1:0] P, input wire [64*10-1:0] e, output wire [31:0] v);
    reducer_p2 #(.N(64), .F(16)) r (.P(P), .e(e), .v(v));
endmodule
module w_fp32_add (input wire [31:0] a, input wire [31:0] b, output wire [31:0] y);
    fp32_add u (.a(a), .b(b), .y(y));
endmodule
module w_reduce_fp32 (input wire [64*32-1:0] y, output wire [31:0] v);
    reducer_fp32 #(.N(64)) r (.y(y), .v(v));
endmodule

// ISA v1: the 4-segment reducer, and one row's 4 accumulators' adders
module w_reduce_p2_seg (input wire [64*24-1:0] P, input wire [64*10-1:0] e, input wire seg4,
                        output wire [4*32-1:0] v);
    reducer_p2_seg #(.F(16)) r (.P(P), .e(e), .seg4(seg4), .v0(v[31:0]), .v1(v[63:32]), .v2(v[95:64]), .v3(v[127:96]));
endmodule

// note 30: the same reducer with a per-group lane mask (64 groups of 16 lanes). A masked group contributes zero
// and the smallest exponent, so it can't set emax. Measured only to price the hardware-mask option; the chosen
// design zero-pads operands in SRAM instead, which needs no mask.
module w_reduce_p2_seg_mask (input wire [64*24-1:0] P, input wire [64*10-1:0] e, input wire seg4,
                             input wire [63:0] mask, output wire [4*32-1:0] v);
    wire [64*24-1:0] Pm;
    wire [64*10-1:0] em;
    genvar g;
    generate for (g = 0; g < 64; g = g + 1) begin : m
        assign Pm[g*24 +: 24] = mask[g] ? P[g*24 +: 24] : 24'd0;
        assign em[g*10 +: 10] = mask[g] ? e[g*10 +: 10] : 10'd0;
    end endgenerate
    reducer_p2_seg #(.F(16)) r (.P(Pm), .e(em), .seg4(seg4), .v0(v[31:0]), .v1(v[63:32]), .v2(v[95:64]), .v3(v[127:96]));
endmodule
