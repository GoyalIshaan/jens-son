// fp32_fma split into 4 pipeline stages (latency 4, one result per cycle). Same arithmetic as fp32_fma.v,
// bit for bit; each stage is its own combinational module so synthesis can report each stage's delay.
//   s1: unpack, special cases, the 24x24 product      s2: align c into the 80-bit frame, signed add
//   s3: magnitude, leading-one detect, normalize       s4: round to nearest even, exponent checks, pack

module fma_s1 (
    input  wire [31:0] a, b, c,
    output reg         spec,            // result decided by a special case
    output reg  [31:0] spec_y,
    output reg  [47:0] P,
    output reg         sp, sc, zc,
    output reg  [23:0] mc,
    output reg  signed [12:0] ep, ecl
);
    reg [7:0] ea, eb, ec;
    reg za, zb, ia, ib, ic, na, nb, nc;
    reg [23:0] ma, mb;
    always @* begin
        ea = a[30:23]; eb = b[30:23]; ec = c[30:23];
        za = (ea == 0); zb = (eb == 0); zc = (ec == 0);
        ia = (ea == 8'hff) && (a[22:0] == 0); ib = (eb == 8'hff) && (b[22:0] == 0); ic = (ec == 8'hff) && (c[22:0] == 0);
        na = (ea == 8'hff) && (a[22:0] != 0); nb = (eb == 8'hff) && (b[22:0] != 0); nc = (ec == 8'hff) && (c[22:0] != 0);
        ma = za ? 24'd0 : {1'b1, a[22:0]};
        mb = zb ? 24'd0 : {1'b1, b[22:0]};
        mc = zc ? 24'd0 : {1'b1, c[22:0]};
        sp = a[31] ^ b[31]; sc = c[31];
        P = ma * mb;
        ep = $signed({5'd0, ea}) + $signed({5'd0, eb}) - 13'sd300;
        ecl = $signed({5'd0, ec}) - 13'sd150;
        spec = 1'b1; spec_y = 32'd0;
        if (na || nb || nc) spec_y = 32'h7fc00000;
        else if ((ia || ib) && (za || zb)) spec_y = 32'h7fc00000;
        else if (ia || ib) spec_y = (ic && (c[31] != sp)) ? 32'h7fc00000 : {sp, 8'hff, 23'd0};
        else if (ic) spec_y = c;
        else if (za || zb) spec_y = zc ? 32'd0 : c;
        else spec = 1'b0;
    end
endmodule

module fma_s2 (
    input  wire [47:0] P,
    input  wire        sp, sc, zc,
    input  wire [23:0] mc,
    input  wire signed [12:0] ep, ecl,
    output reg  signed [80:0] S,
    output reg  signed [12:0] wf
);
    reg [79:0] Pf, Cf, mask;
    reg signed [12:0] sh;
    reg st;
    always @* begin
        sh = ecl - ep;
        wf = ep - 2;
        Pf = {30'd0, P, 2'b00};
        Cf = 80'd0;
        if (zc) Cf = 80'd0;
        else if (sh > 50) begin Cf = {4'd0, mc, 52'd0}; Pf = 80'd1; wf = ecl - 52; end
        else if (sh >= -2) Cf = {56'd0, mc} << (sh + 2);
        else if (-sh - 2 >= 26) Cf = 80'd1;
        else begin
            mask = (80'd1 << (-sh - 2)) - 1;
            st = |({56'd0, mc} & mask);
            Cf = ({56'd0, mc} >> (-sh - 2));
            Cf[0] = Cf[0] | st;
        end
        S = (sp ? -$signed({1'b0, Pf}) : $signed({1'b0, Pf})) + (sc ? -$signed({1'b0, Cf}) : $signed({1'b0, Cf}));
    end
endmodule

module fma_s3 (
    input  wire signed [80:0] S,
    input  wire signed [12:0] wf,
    output reg  [79:0] N,               // normalized: leading one at bit 79
    output reg  signed [12:0] E,        // biased exponent before rounding
    output reg         neg, zero
);
    reg [79:0] M;
    integer k, L;
    always @* begin
        zero = (S == 0);
        neg = S[80];
        M = S[80] ? -S : S;
        L = 0;
        for (k = 0; k < 80; k = k + 1) if (M[k]) L = k;
        N = M << (79 - L);
        E = L + wf + 127;
    end
endmodule

module fma_s4 (
    input  wire [79:0] N,
    input  wire signed [12:0] E,
    input  wire        neg, zero,
    output reg  [31:0] y
);
    reg [24:0] m;
    reg signed [12:0] Er;
    always @* begin
        m = {1'b0, N[79:56]};
        if (N[55] && ((|N[54:0]) || m[0])) m = m + 1;
        Er = E;
        if (m[24]) begin m = m >> 1; Er = Er + 1; end
        if (zero) y = 32'd0;
        else if (Er >= 255) y = {neg, 8'hff, 23'd0};
        else if (Er <= 0) y = 32'd0;
        else y = {neg, Er[7:0], m[22:0]};
    end
endmodule

module fp32_fma_pipe (
    input  wire        clk,
    input  wire [31:0] a, b, c,
    output wire [31:0] y
);
    // stage 1
    wire        spec1; wire [31:0] sy1; wire [47:0] P1; wire sp1, sc1, zc1; wire [23:0] mc1;
    wire signed [12:0] ep1, ecl1;
    fma_s1 s1 (.a(a), .b(b), .c(c), .spec(spec1), .spec_y(sy1), .P(P1), .sp(sp1), .sc(sc1), .zc(zc1), .mc(mc1),
               .ep(ep1), .ecl(ecl1));
    reg        spec_r1; reg [31:0] sy_r1; reg [47:0] P_r1; reg sp_r1, sc_r1, zc_r1; reg [23:0] mc_r1;
    reg signed [12:0] ep_r1, ecl_r1;
    always @(posedge clk) begin
        spec_r1 <= spec1; sy_r1 <= sy1; P_r1 <= P1; sp_r1 <= sp1; sc_r1 <= sc1; zc_r1 <= zc1; mc_r1 <= mc1;
        ep_r1 <= ep1; ecl_r1 <= ecl1;
    end
    // stage 2
    wire signed [80:0] S2; wire signed [12:0] wf2;
    fma_s2 s2 (.P(P_r1), .sp(sp_r1), .sc(sc_r1), .zc(zc_r1), .mc(mc_r1), .ep(ep_r1), .ecl(ecl_r1), .S(S2), .wf(wf2));
    reg signed [80:0] S_r2; reg signed [12:0] wf_r2; reg spec_r2; reg [31:0] sy_r2;
    always @(posedge clk) begin S_r2 <= S2; wf_r2 <= wf2; spec_r2 <= spec_r1; sy_r2 <= sy_r1; end
    // stage 3
    wire [79:0] N3; wire signed [12:0] E3; wire neg3, zero3;
    fma_s3 s3 (.S(S_r2), .wf(wf_r2), .N(N3), .E(E3), .neg(neg3), .zero(zero3));
    reg [79:0] N_r3; reg signed [12:0] E_r3; reg neg_r3, zero_r3, spec_r3; reg [31:0] sy_r3;
    always @(posedge clk) begin N_r3 <= N3; E_r3 <= E3; neg_r3 <= neg3; zero_r3 <= zero3; spec_r3 <= spec_r2; sy_r3 <= sy_r2; end
    // stage 4
    wire [31:0] y4;
    fma_s4 s4 (.N(N_r3), .E(E_r3), .neg(neg_r3), .zero(zero_r3), .y(y4));
    reg [31:0] y_r4;
    always @(posedge clk) y_r4 <= spec_r3 ? sy_r3 : y4;
    assign y = y_r4;
endmodule
