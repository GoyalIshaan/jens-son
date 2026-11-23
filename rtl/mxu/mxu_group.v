// One matrix-unit group: 16 weight lanes x R token rows (D-015, D-017, D-005).
//
// Per cycle, for every token row r:
//   S_r = sum_{i<16} table[code_i] * act[r][i]           exact, 19-bit signed (worst case 16*127*127)
//   y_r = S_r * ws * as_r                                  ws: E4M3 weight scale of this group,
//                                                          as_r: activation scale of row r (fp32 or 2^k)
// The fp32 result goes on to the shared cross-group adder (not in this block).
//
// Pipeline: stage 1 table lookup + 16*R int8 x int8 multiplies, stage 2 the 16-input adder trees,
// stage 3 the scale step. Latency 3 cycles, one new weight beat per cycle.
//
// MODE selects how the scale step is built (we measure all three):
//   0 "A": D-017 as written. ws -> fp32, t = ws * as_r (fp32 mul, RNE), y = fp32(S_r) * t (fp32 mul, RNE).
//   1 "C": exact P = S_r * M (19 x 4-bit integer multiply, M = E4M3 mantissa with hidden bit),
//          then y = fp32(P) * as_r * 2^E (one fp32 mul, RNE; fp32(P) is exact since |P| < 2^24).
//          One rounding instead of two, so it is also more accurate than A.
//   3 "D": as C, but the activation scale is bf16 (8-bit mantissa): a 24 x 8 mantissa multiply instead of 24 x 24.
//   2 "B": activation scales are powers of two (as_r = 2^k_r): y is the exact pair (P, E + k_r).
//          No floating-point multiply at all. Changes the numerics, so it needs a quality check.
// Conventions (part of the bit-exact contract): fp32 results flush to +0 below 2^-126, a zero product is +0,
// rounding is round-to-nearest-even. Scales are finite and nonzero (E4M3 NaN code never used).

module mxu_group #(
    parameter R = 4,
    parameter MODE = 2          // D-019: power-of-two activation scales (mode B) is the chip's design
) (
    input  wire              clk,
    // stationary state, loaded by the program
    input  wire              table_we,
    input  wire [16*8-1:0]   table_in,      // 16 int8 entries (D-005: k/127 levels stored as k)
    input  wire              act_we,
    input  wire [R*16*8-1:0] act_in,        // R rows x 16 int8 activations
    input  wire [R*32-1:0]   as_in,         // R activation scales: fp32 (MODE 0,1) or signed exponent k in [7:0] (MODE 2)
    // streaming, one beat per cycle
    input  wire              in_valid,
    input  wire [16*4-1:0]   codes,         // 16 x 4-bit weight codes
    input  wire [7:0]        ws,            // E4M3 weight scale of this group
    output reg               out_valid,
    output reg  [R*32-1:0]   y,             // MODE 0,1: R fp32 results
    output reg  [R*24-1:0]   y_mant,        // MODE 2: R x signed 24-bit P (exact)
    output reg  [R*10-1:0]   y_exp          // MODE 2: R x signed 10-bit exponent
);
    // ------------------------------------------------------------ stationary registers
    reg  [16*8-1:0]   tbl;
    reg  [R*16*8-1:0] act;
    reg  [R*32-1:0]   as_r;
    always @(posedge clk) begin
        if (table_we) tbl  <= table_in;
        if (act_we)   act  <= act_in;
        if (act_we)   as_r <= as_in;
    end

    // ------------------------------------------------------------ stage 1: lookup + multiply
    wire [16*8-1:0] w;
    genvar i, r;
    generate
        for (i = 0; i < 16; i = i + 1) begin : lookup
            assign w[i*8 +: 8] = tbl[codes[i*4 +: 4]*8 +: 8];
        end
    endgenerate

    reg  [R*16*16-1:0] prod;     // R x 16 signed 16-bit products
    reg  [7:0]         ws1;
    reg                v1;
    generate
        for (r = 0; r < R; r = r + 1) begin : row_mul
            for (i = 0; i < 16; i = i + 1) begin : lane
                wire signed [7:0]  a = act[(r*16+i)*8 +: 8];
                wire signed [7:0]  b = w[i*8 +: 8];
                always @(posedge clk) prod[(r*16+i)*16 +: 16] <= a * b;
            end
        end
    endgenerate
    always @(posedge clk) begin ws1 <= ws; v1 <= in_valid; end

    // ------------------------------------------------------------ stage 2: 16-input adder trees
    reg  [R*19-1:0] sum;
    reg  [7:0]      ws2;
    reg             v2;
    generate
        for (r = 0; r < R; r = r + 1) begin : row_tree
            wire signed [18:0] s;
            adder_tree16 tree (.p(prod[r*16*16 +: 256]), .s(s));
            always @(posedge clk) sum[r*19 +: 19] <= s;
        end
    endgenerate
    always @(posedge clk) begin ws2 <= ws1; v2 <= v1; end

    // ------------------------------------------------------------ stage 3: scale
    // E4M3 -> integer mantissa M (4 bits incl. hidden bit) and exponent E with ws = M * 2^E
    wire        ws_s = ws2[7];
    wire [3:0]  ws_e = ws2[6:3];
    wire [2:0]  ws_m = ws2[2:0];
    wire [3:0]  M    = (ws_e != 0) ? {1'b1, ws_m} : {1'b0, ws_m};
    wire signed [9:0] E = (ws_e != 0) ? ($signed({6'd0, ws_e}) - 10'sd10) : -10'sd9;   // e-7-3, or -6-3

    generate
        for (r = 0; r < R; r = r + 1) begin : row_scale
            wire signed [18:0] S = sum[r*19 +: 19];
            if (MODE == 0) begin : mode_a
                wire [31:0] ws_f, S_f, t, yy;
                small_to_fp32 cw (.s(ws_s), .m(M), .e(E), .f(ws_f));
                int_to_fp32 #(.W(19)) cs (.x(S), .f(S_f));
                fp32_mul m1 (.a(ws_f), .b(as_r[r*32 +: 32]), .shift(10'sd0), .y(t));
                fp32_mul m2 (.a(S_f), .b(t), .shift(10'sd0), .y(yy));
                always @(posedge clk) y[r*32 +: 32] <= yy;
            end else if (MODE == 1) begin : mode_c
                wire signed [23:0] SM = S * $signed({1'b0, M});
                wire signed [23:0] P  = ws_s ? -SM : SM;
                wire [31:0] P_f, yy;
                int_to_fp32 #(.W(24)) cp (.x(P), .f(P_f));
                fp32_mul m1 (.a(P_f), .b(as_r[r*32 +: 32]), .shift(E), .y(yy));
                always @(posedge clk) y[r*32 +: 32] <= yy;
            end else if (MODE == 3) begin : mode_d
                // as C, but the activation scale is bf16 (8-bit mantissa): its upper 16 bits are used
                wire signed [23:0] SM = S * $signed({1'b0, M});
                wire signed [23:0] P  = ws_s ? -SM : SM;
                wire [31:0] P_f, yy;
                int_to_fp32 #(.W(24)) cp (.x(P), .f(P_f));
                fp32_mul_b8 m1 (.a(P_f), .b(as_r[r*32+16 +: 16]), .shift(E), .y(yy));
                always @(posedge clk) y[r*32 +: 32] <= yy;
            end else begin : mode_b
                wire signed [23:0] SM = S * $signed({1'b0, M});
                wire signed [23:0] P  = ws_s ? -SM : SM;
                wire signed [9:0]  k = {{2{as_r[r*32+7]}}, as_r[r*32 +: 8]};
                always @(posedge clk) begin
                    y_mant[r*24 +: 24] <= P;
                    y_exp[r*10 +: 10]  <= E + k;
                end
            end
        end
    endgenerate
    always @(posedge clk) out_valid <= v2;
endmodule


// Balanced tree: 16 -> 8 -> 4 -> 2 -> 1. Inputs are in -127..127 (never -128), so each product is at most
// 16,129 in size and the levels need only 16, 17, 18 and 19 bits (sums up to 32,258 / 64,516 / 129,032 / 258,064).
module adder_tree16 (
    input  wire [16*16-1:0] p,
    output wire signed [18:0] s
);
    wire signed [15:0] l1 [0:7];
    wire signed [16:0] l2 [0:3];
    wire signed [17:0] l3 [0:1];
    genvar j;
    generate
        for (j = 0; j < 8; j = j + 1) begin : g1
            assign l1[j] = $signed(p[(2*j)*16 +: 16]) + $signed(p[(2*j+1)*16 +: 16]);
        end
        for (j = 0; j < 4; j = j + 1) begin : g2
            assign l2[j] = l1[2*j] + l1[2*j+1];
        end
        for (j = 0; j < 2; j = j + 1) begin : g3
            assign l3[j] = l2[2*j] + l2[2*j+1];
        end
    endgenerate
    assign s = l3[0] + l3[1];
endmodule


// Exact conversion of a signed integer with |x| < 2^24 to fp32. Zero -> +0.
module int_to_fp32 #(parameter W = 24) (
    input  wire signed [W-1:0] x,
    output reg  [31:0] f
);
    reg [W-1:0] mag;
    reg [23:0]  norm;
    reg [7:0]   ex;
    integer k, lz;
    always @* begin
        mag = x[W-1] ? -x : x;
        lz = W;
        for (k = 0; k < W; k = k + 1)
            if (mag[k]) lz = W - 1 - k;
        norm = mag;                                   // zero-extended to 24 bits
        norm = norm << (lz + 24 - W);                 // leading one now at bit 23
        ex = 127 + W - 1 - lz;
        if (mag == 0) f = 32'd0;
        else f = {x[W-1], ex, norm[22:0]};
    end
endmodule


// A tiny positive value M * 2^E (M 4-bit, may be 0..15) to fp32, with sign s. Exact.
module small_to_fp32 (
    input  wire        s,
    input  wire [3:0]  m,
    input  wire signed [9:0] e,
    output reg  [31:0] f
);
    integer lz;
    reg [3:0] n;
    reg [7:0] ex;
    always @* begin
        if (m[3])      lz = 0;
        else if (m[2]) lz = 1;
        else if (m[1]) lz = 2;
        else           lz = 3;
        n = m << lz;
        ex = 127 + 3 - lz + e;
        if (m == 0) f = 32'd0;
        else f = {s, ex, n[2:0], 20'd0};
    end
endmodule


// fp32 multiply with an extra power-of-two factor 2^shift, round to nearest even.
// Inputs are normal numbers or zero. Results below 2^-126 flush to +0; zero inputs give +0.
// (Overflow is impossible for the scale ranges the program uses; it is not handled.)
module fp32_mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire signed [9:0] shift,
    output reg  [31:0] y
);
    reg        sg;
    reg [23:0] ma, mb;
    reg [47:0] pm;
    reg [24:0] mr;           // rounded mantissa with possible carry
    reg        g, st;
    reg signed [11:0] ex;
    always @* begin
        sg = a[31] ^ b[31];
        ma = {1'b1, a[22:0]};
        mb = {1'b1, b[22:0]};
        pm = ma * mb;                                            // in [2^46, 2^48)
        ex = $signed({4'd0, a[30:23]}) + $signed({4'd0, b[30:23]}) - 12'sd127 + shift;
        if (pm[47]) begin
            mr = {1'b0, pm[47:24]}; g = pm[23]; st = |pm[22:0]; ex = ex + 1;
        end else begin
            mr = {1'b0, pm[46:23]}; g = pm[22]; st = |pm[21:0];
        end
        if (g && (st || mr[0])) mr = mr + 1;                     // round to nearest, ties to even
        if (mr[24]) begin mr = mr >> 1; ex = ex + 1; end
        if (a[30:23] == 0 || b[30:23] == 0 || ex <= 0) y = 32'd0;
        else y = {sg, ex[7:0], mr[22:0]};
    end
endmodule


// fp32 x bf16 multiply with an extra 2^shift, round to nearest even. Same conventions as fp32_mul.
module fp32_mul_b8 (
    input  wire [31:0] a,
    input  wire [15:0] b,
    input  wire signed [9:0] shift,
    output reg  [31:0] y
);
    reg        sg;
    reg [23:0] ma;
    reg [7:0]  mb;
    reg [31:0] pm;
    reg [24:0] mr;
    reg        g, st;
    reg signed [11:0] ex;
    always @* begin
        sg = a[31] ^ b[15];
        ma = {1'b1, a[22:0]};
        mb = {1'b1, b[6:0]};
        pm = ma * mb;                                            // in [2^30, 2^32)
        ex = $signed({4'd0, a[30:23]}) + $signed({4'd0, b[14:7]}) - 12'sd127 + shift;
        if (pm[31]) begin
            mr = {1'b0, pm[31:8]}; g = pm[7]; st = |pm[6:0]; ex = ex + 1;
        end else begin
            mr = {1'b0, pm[30:7]}; g = pm[6]; st = |pm[5:0];
        end
        if (g && (st || mr[0])) mr = mr + 1;
        if (mr[24]) begin mr = mr >> 1; ex = ex + 1; end
        if (a[30:23] == 0 || b[14:7] == 0 || ex <= 0) y = 32'd0;
        else y = {sg, ex[7:0], mr[22:0]};
    end
endmodule
