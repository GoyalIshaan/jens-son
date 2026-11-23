// mxu_row (chip_mxu.sv) in pipeline stages for 1 GHz, same arithmetic (golden/bitexact.py::mm_exact):
//   1  products a * w and each group's 16-lane sum S                                     -> registers
//   2  P = +-S * M_g, e = E_g + k                                                         -> registers
//   3  the exponent max tree, levels 1-3 (64 -> 8; a 6-level tree is 1.2 ns)                -> registers
//   4  levels 4-6: the largest exponent, and each 16-group segment's for seg4              -> registers
//   5  each group's alignment shift                                                       -> registers
//   6  the exact add tree, levels 1-3 (64 -> 8 partial sums)                              -> registers
//   7  levels 4-6 (8 -> 1, keeping the four 16-group subtrees for seg4)                   -> registers
//   8  one rounding of each sum to fp32                                                    -> registers
// Results leave LAT = 8 enabled clocks after the inputs. ce low holds every stage.
module mxu_row_pipe (
    input wire clk, ce,
    input wire [8191:0] w, input wire [64*4-1:0] gM, input wire [63:0] gS, input wire [64*10-1:0] gE,
    input wire [8191:0] act, input wire [255:0] aexp, input wire seg4,
    output wire [31:0] v0, v1, v2, v3
);
    localparam N = 64, F = 16, W = 24 + F + 6;
    genvar g, i;
    reg [7:0] seg;                                    // seg4, one bit per stage
    always @(posedge clk) if (ce) seg <= {seg[6:0], seg4};
    // ---- 1, 2
    reg signed [18:0] S1 [0:N-1]; reg [3:0] M1 [0:N-1]; reg G1 [0:N-1]; reg signed [9:0] e1 [0:N-1];
    reg signed [23:0] P2 [0:N-1]; reg signed [9:0] e2 [0:N-1];
    for (g = 0; g < N; g = g + 1) begin : grp
        wire [16*16-1:0] prod;
        for (i = 0; i < 16; i = i + 1) begin : lane
            wire signed [7:0] a = act[8*(16*g+i)+:8], b = w[8*(16*g+i)+:8];
            wire signed [15:0] pr = a * b;
            assign prod[16*i+:16] = pr;
        end
        wire signed [18:0] S;
        adder_tree16 tree(.p(prod), .s(S));
        wire signed [9:0] k = {{2{aexp[8*(g/2)+7]}}, aexp[8*(g/2)+:8]};
        wire signed [23:0] SM = S1[g] * $signed({1'b0, M1[g]});
        always @(posedge clk) if (ce) begin
            S1[g] <= S; M1[g] <= gM[4*g+:4]; G1[g] <= gS[g]; e1[g] <= $signed(gE[10*g+:10]) + k;
            P2[g] <= G1[g] ? -SM : SM; e2[g] <= e1[g];
        end
    end
    // ---- 3, 4: max tree in two halves (heap nodes 7..14 are the 8-group maxima)
    wire signed [9:0] mx [0:2*N-2];
    for (g = 0; g < N; g = g + 1) begin : leaf assign mx[N - 1 + g] = e2[g]; end
    for (g = N - 2; g >= 7; g = g - 1) begin : node assign mx[g] = (mx[2*g+1] > mx[2*g+2]) ? mx[2*g+1] : mx[2*g+2]; end
    reg signed [9:0] mh [7:14]; reg signed [23:0] Pa [0:N-1]; reg signed [9:0] ea [0:N-1];
    for (g = 7; g <= 14; g = g + 1) begin : keep_half always @(posedge clk) if (ce) mh[g] <= mx[g]; end
    for (g = 0; g < N; g = g + 1) begin : hold_a always @(posedge clk) if (ce) begin Pa[g] <= P2[g]; ea[g] <= e2[g]; end end
    wire signed [9:0] mt [0:14];
    for (g = 7; g <= 14; g = g + 1) begin : top_l assign mt[g] = mh[g]; end
    for (g = 6; g >= 0; g = g - 1) begin : top_n assign mt[g] = (mt[2*g+1] > mt[2*g+2]) ? mt[2*g+1] : mt[2*g+2]; end
    reg signed [9:0] m3 [0:6]; reg signed [23:0] P3 [0:N-1]; reg signed [9:0] e3 [0:N-1];
    for (g = 0; g < 7; g = g + 1) begin : keep_max always @(posedge clk) if (ce) m3[g] <= mt[g]; end
    for (g = 0; g < N; g = g + 1) begin : hold3 always @(posedge clk) if (ce) begin P3[g] <= Pa[g]; e3[g] <= ea[g]; end end
    // ---- 4: align
    reg signed [W-1:0] T4 [0:N-1]; reg signed [9:0] m4 [0:6];
    for (g = 0; g < N; g = g + 1) begin : align
        wire signed [9:0] ref_e = seg[3] ? m3[3 + g / 16] : m3[0];
        wire [10:0] sh = ref_e - e3[g];
        wire signed [W-1:0] t0 = $signed(P3[g]) <<< F;
        always @(posedge clk) if (ce) T4[g] <= (sh > W - 1) ? (t0 < 0 ? -1 : 0) : (t0 >>> sh);
    end
    for (g = 0; g < 7; g = g + 1) begin : keep4 always @(posedge clk) if (ce) m4[g] <= m3[g]; end
    // ---- 5: add levels 1-3 (heap nodes 7..14 hold the 8-group sums; their order is the one-cycle tree's)
    wire signed [W-1:0] sm [0:2*N-2];
    for (g = 0; g < N; g = g + 1) begin : leaf2 assign sm[N - 1 + g] = T4[g]; end
    for (g = N - 2; g >= 7; g = g - 1) begin : add_lo assign sm[g] = sm[2*g+1] + sm[2*g+2]; end
    reg signed [W-1:0] s5 [7:14]; reg signed [9:0] m5 [0:6];
    for (g = 7; g <= 14; g = g + 1) begin : hold5 always @(posedge clk) if (ce) s5[g] <= sm[g]; end
    for (g = 0; g < 7; g = g + 1) begin : keep5 always @(posedge clk) if (ce) m5[g] <= m4[g]; end
    // ---- 6: add levels 4-6
    wire signed [W-1:0] su [0:14];
    for (g = 7; g <= 14; g = g + 1) begin : top_leaf assign su[g] = s5[g]; end
    for (g = 6; g >= 0; g = g - 1) begin : add_hi assign su[g] = su[2*g+1] + su[2*g+2]; end
    reg signed [W-1:0] s6 [0:3]; reg signed [9:0] b6 [0:3];
    always @(posedge clk) if (ce) begin
        s6[0] <= seg[5] ? su[3] : su[0]; b6[0] <= seg[5] ? m5[3] : m5[0];
        s6[1] <= su[4]; b6[1] <= m5[4];
        s6[2] <= su[5]; b6[2] <= m5[5];
        s6[3] <= su[6]; b6[3] <= m5[6];
    end
    // ---- 7: one rounding each
    wire [31:0] f [0:3];
    reg [31:0] y7 [0:3];
    for (g = 0; g < 4; g = g + 1) begin : cvt
        wide_to_fp32 #(.W(W)) c (.x(s6[g]), .base(b6[g] - F), .f(f[g]));
        always @(posedge clk) if (ce) y7[g] <= f[g];
    end
    assign v0 = y7[0]; assign v1 = y7[1]; assign v2 = y7[2]; assign v3 = y7[3];
endmodule
