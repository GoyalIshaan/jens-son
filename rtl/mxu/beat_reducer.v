// The beat reducer: one weight beat covers 1,024 lanes = 64 groups. For each token row, the 64 group results
// are combined into one number, then added into that output's running fp32 total (one beat per 1,024 inputs).
//
// Power-of-two activation scales (note 25, mode B) make every group result an exact integer times a power of
// two: v_g = P_g * 2^e_g (P_g signed 24-bit, e_g signed 10-bit). So the 64 can be added in fixed point:
//   emax = max_g e_g
//   T_g  = floor(P_g * 2^(F - (emax - e_g)))       arithmetic shift; exact unless emax - e_g > F
//   S    = sum_g T_g                                exact, 24 + F + 6 bits
//   v    = fp32(S * 2^(emax - F))                   one rounding, to nearest even
//   acc  = fp32(acc + v)                            one fp32 add per row per beat, to nearest even
// Conventions (bit-exact contract): results below 2^-126 flush to +0, zero is +0, no overflow handling.

module reducer_p2 #(
    parameter N = 64,           // groups per beat
    parameter F = 16            // fraction bits kept below the largest group's LSB
) (
    input  wire [N*24-1:0] P,
    input  wire [N*10-1:0] e,
    output wire [31:0]     v
);
    localparam W = 24 + F + 6;                       // 46 for N = 64, F = 16
    localparam L = $clog2(N);                        // tree depth, 6
    genvar g, l;
    // ---- 1. largest exponent: a balanced max tree, log2(N) levels
    wire signed [9:0] mx [0:2*N-2];                  // heap layout: leaves N-1 .. 2N-2, root 0
    generate
        for (g = 0; g < N; g = g + 1) begin : leaf
            assign mx[N - 1 + g] = $signed(e[g*10 +: 10]);
        end
        for (g = N - 2; g >= 0; g = g - 1) begin : node
            assign mx[g] = (mx[2*g+1] > mx[2*g+2]) ? mx[2*g+1] : mx[2*g+2];
        end
    endgenerate
    wire signed [9:0] emax = mx[0];
    // ---- 2. align each group (floor), 3. add exactly with a balanced tree
    wire signed [W-1:0] sm [0:2*N-2];
    generate
        for (g = 0; g < N; g = g + 1) begin : align
            wire [10:0] sh = emax - $signed(e[g*10 +: 10]);                  // 0 .. 1023
            wire signed [W-1:0] t0 = $signed(P[g*24 +: 24]) <<< F;
            assign sm[N - 1 + g] = (sh > W - 1) ? (t0 < 0 ? -1 : 0) : (t0 >>> sh);
        end
        for (g = N - 2; g >= 0; g = g - 1) begin : add
            assign sm[g] = sm[2*g+1] + sm[2*g+2];
        end
    endgenerate
    wire signed [W-1:0] S = sm[0];
    // ---- 4. one conversion to fp32
    wide_to_fp32 #(.W(W)) cv (.x(S), .base(emax - F), .f(v));
endmodule


// Signed W-bit integer times 2^base -> fp32, round to nearest even, flush below 2^-126 to +0.
module wide_to_fp32 #(parameter W = 46) (
    input  wire signed [W-1:0] x,
    input  wire signed [10:0]  base,
    output reg  [31:0] f
);
    reg [W-1:0] mag;
    reg [W+1:0] norm;            // leading one moved to bit W-1
    reg [24:0]  m;
    reg         gd, st;
    reg signed [12:0] ex;
    integer k, p;
    always @* begin
        mag = x[W-1] ? -x : x;
        p = 0;
        for (k = 0; k < W; k = k + 1)
            if (mag[k]) p = k;                                 // position of the leading one
        norm = {2'b0, mag} << (W - 1 - p);
        m  = {1'b0, norm[W-1 -: 24]};                          // 24 bits incl. the leading one
        gd = (W > 24) ? norm[W-25] : 1'b0;
        st = (W > 25) ? |(norm[W-26:0]) : 1'b0;
        if (gd && (st || m[0])) m = m + 1;
        ex = base + p + 127;
        if (m[24]) begin m = m >> 1; ex = ex + 1; end
        if (mag == 0 || ex <= 0) f = 32'd0;
        else f = {x[W-1], ex[7:0], m[22:0]};
    end
endmodule


// fp32 + fp32, round to nearest even. Inputs normal or zero; results below 2^-126 flush to +0; zero is +0.
module fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg  [31:0] y
);
    reg [31:0] x, z;             // |x| >= |z|
    reg [26:0] mx, mz;           // 24-bit mantissa + guard, round, sticky
    reg [27:0] r;
    reg [7:0]  d;
    reg        st;
    reg signed [9:0] ex;
    integer k, lz;
    reg [24:0] m;
    always @* begin
        if (a[30:0] >= b[30:0]) begin x = a; z = b; end else begin x = b; z = a; end
        mx = {1'b1, x[22:0], 3'b0};
        mz = {1'b1, z[22:0], 3'b0};
        d  = x[30:23] - z[30:23];
        if (d >= 27) begin st = |mz; mz = 27'd0; end
        else begin st = |(mz & ((27'd1 << d) - 1)); mz = mz >> d; end
        mz[0] = mz[0] | st;
        ex = {2'b0, x[30:23]};
        if (x[31] == z[31]) r = mx + mz;
        else                r = mx - mz;
        if (r[27]) begin                                   // carry out: shift right, keep sticky
            r = {1'b0, r[27:2], r[1] | r[0]};
            ex = ex + 1;
        end else begin                                     // normalize left
            lz = 27;
            for (k = 26; k >= 0; k = k - 1)
                if (r[k] && lz == 27) lz = 26 - k;
            if (lz < 27) begin r = r << lz; ex = ex - lz; end
        end
        m = {1'b0, r[26:3]};
        if (r[2] && (r[1] || r[0] || m[0])) m = m + 1;
        if (m[24]) begin m = m >> 1; ex = ex + 1; end
        if (x[30:23] == 0) y = 32'd0;                      // both zero
        else if (z[30:23] == 0) y = x;                     // adding zero
        else if (r == 0 || ex <= 0) y = 32'd0;             // exact cancellation or underflow
        else y = {x[31], ex[7:0], m[22:0]};
    end
endmodule


// One output row's accumulator: acc <= acc + v each beat; clear starts a new output.
module row_acc (
    input  wire        clk,
    input  wire        clear,
    input  wire        en,
    input  wire [31:0] v,
    output reg  [31:0] acc
);
    wire [31:0] s;
    fp32_add add (.a(clear ? 32'd0 : acc), .b(v), .y(s));
    always @(posedge clk) if (en) acc <= s;
endmodule


// For comparison: the fp32 version (modes A/C/D) needs a 63-adder fp32 tree per row. Pairwise, fixed order.
module reducer_fp32 #(parameter N = 64) (
    input  wire [N*32-1:0] y,
    output wire [31:0] v
);
    genvar i, l;
    generate
        for (l = 0; (N >> l) > 1; l = l + 1) begin : lvl
            localparam M = N >> (l + 1);
            wire [M*32-1:0] o;
            for (i = 0; i < M; i = i + 1) begin : add
                if (l == 0) begin : first
                    fp32_add a (.a(y[(2*i)*32 +: 32]), .b(y[(2*i+1)*32 +: 32]), .y(o[i*32 +: 32]));
                end else begin : next
                    fp32_add a (.a(lvl[l-1].o[(2*i)*32 +: 32]), .b(lvl[l-1].o[(2*i+1)*32 +: 32]), .y(o[i*32 +: 32]));
                end
            end
        end
    endgenerate
    assign v = lvl[5].o[31:0];
endmodule


// ISA v1 (note 28) adds a 4-segment mode for attention: q . K and P . V beats are 4 outputs x 256 inputs, i.e.
// 4 independent sums of 16 groups each. The same max tree and add tree are tapped one level below the
// root's two levels (heap nodes 3..6 cover groups 0-15, 16-31, 32-47, 48-63). In seg4 mode each group is
// aligned to its own segment's maximum, so those subtree sums are exact per segment; three more converters
// produce outputs 1..3. seg4 = 0 is exactly reducer_p2 (v0; v1..v3 are don't-care).
module reducer_p2_seg #(
    parameter F = 16
) (
    input  wire [64*24-1:0] P,
    input  wire [64*10-1:0] e,
    input  wire             seg4,
    output wire [31:0]      v0,
    output wire [31:0]      v1,
    output wire [31:0]      v2,
    output wire [31:0]      v3
);
    localparam N = 64;
    localparam W = 24 + F + 6;
    genvar g;
    wire signed [9:0] mx [0:2*N-2];
    generate
        for (g = 0; g < N; g = g + 1) begin : leaf
            assign mx[N - 1 + g] = $signed(e[g*10 +: 10]);
        end
        for (g = N - 2; g >= 0; g = g - 1) begin : node
            assign mx[g] = (mx[2*g+1] > mx[2*g+2]) ? mx[2*g+1] : mx[2*g+2];
        end
    endgenerate
    wire signed [W-1:0] sm [0:2*N-2];
    generate
        for (g = 0; g < N; g = g + 1) begin : align
            wire signed [9:0] ref_e = seg4 ? mx[3 + g / 16] : mx[0];
            wire [10:0] sh = ref_e - $signed(e[g*10 +: 10]);
            wire signed [W-1:0] t0 = $signed(P[g*24 +: 24]) <<< F;
            assign sm[N - 1 + g] = (sh > W - 1) ? (t0 < 0 ? -1 : 0) : (t0 >>> sh);
        end
        for (g = N - 2; g >= 0; g = g - 1) begin : add
            assign sm[g] = sm[2*g+1] + sm[2*g+2];
        end
    endgenerate
    wire signed [W-1:0] s0 = seg4 ? sm[3] : sm[0];
    wire signed [9:0]   b0 = seg4 ? mx[3] : mx[0];
    wide_to_fp32 #(.W(W)) c0 (.x(s0),    .base(b0 - F),    .f(v0));
    wide_to_fp32 #(.W(W)) c1 (.x(sm[4]), .base(mx[4] - F), .f(v1));
    wide_to_fp32 #(.W(W)) c2 (.x(sm[5]), .base(mx[5] - F), .f(v2));
    wide_to_fp32 #(.W(W)) c3 (.x(sm[6]), .base(mx[6] - F), .f(v3));
endmodule
