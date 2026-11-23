module probe_w2f(input wire signed [45:0] x, input wire signed [10:0] b, output wire [31:0] f);
    wide_to_fp32 #(.W(46)) c(.x(x), .base(b), .f(f));
endmodule
module probe_grp(input wire [127:0] a, w, output wire signed [18:0] s);
    wire [255:0] prod; genvar i;
    for (i = 0; i < 16; i = i + 1) begin : l
        wire signed [7:0] x = a[8*i+:8], y = w[8*i+:8]; wire signed [15:0] p = x * y; assign prod[16*i+:16] = p;
    end
    adder_tree16 t(.p(prod), .s(s));
endmodule
module probe_align(input wire signed [23:0] P, input wire signed [9:0] e, ref_e, output wire signed [45:0] T);
    wire [10:0] sh = ref_e - e; wire signed [45:0] t0 = $signed(P) <<< 16;
    assign T = (sh > 45) ? (t0 < 0 ? -1 : 0) : (t0 >>> sh);
endmodule
module probe_add3(input wire [8*46-1:0] x, output wire signed [45:0] s);
    wire signed [45:0] a0 = $signed(x[0+:46]) + $signed(x[46+:46]), a1 = $signed(x[92+:46]) + $signed(x[138+:46]);
    wire signed [45:0] a2 = $signed(x[184+:46]) + $signed(x[230+:46]), a3 = $signed(x[276+:46]) + $signed(x[322+:46]);
    assign s = (a0 + a1) + (a2 + a3);
endmodule
module probe_sm(input wire signed [18:0] S, input wire [3:0] M, input wire neg, output wire signed [23:0] P);
    wire signed [23:0] SM = S * $signed({1'b0, M}); assign P = neg ? -SM : SM;
endmodule
module probe_row_nce(input wire clk, input wire [8191:0] w, input wire [255:0] gM, input wire [63:0] gS, input wire [639:0] gE,
                     input wire [8191:0] act, input wire [255:0] aexp, input wire seg4, output wire [31:0] v0, v1, v2, v3);
    mxu_row_pipe r(.clk(clk), .ce(1'b1), .w(w), .gM(gM), .gS(gS), .gE(gE), .act(act), .aexp(aexp), .seg4(seg4),
                   .v0(v0), .v1(v1), .v2(v2), .v3(v3));
endmodule
module probe_max(input wire [64*10-1:0] e, output wire signed [9:0] m0, output wire signed [9:0] m3);
    wire signed [9:0] mx [0:126]; genvar g;
    for (g = 0; g < 64; g = g + 1) begin : l assign mx[63 + g] = $signed(e[10*g+:10]); end
    for (g = 62; g >= 0; g = g - 1) begin : n assign mx[g] = (mx[2*g+1] > mx[2*g+2]) ? mx[2*g+1] : mx[2*g+2]; end
    assign m0 = mx[0]; assign m3 = mx[3];
endmodule
