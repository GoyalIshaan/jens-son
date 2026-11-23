// fp32 fused multiply-add, y = a*b + c with ONE rounding (round to nearest, ties to even).
// The vector unit's contract (golden/bitexact_vec.py, D-024):
//   - results below 2^-126 in magnitude flush to +0, and every zero result is +0
//   - subnormal inputs are treated as zero (the chip never produces them: every result is flushed)
//   - overflow gives +-inf; inf and NaN follow IEEE (NaN out = 7fc00000)
// ADD = fma(a, 1, b), SUB = fma(a, 1, -b), MUL = fma(a, b, +0) give exactly the IEEE single results with the
// same flush, so one of these per lane does all four elementwise ops.
//
// Datapath (combinational): the 48-bit product P = ma*mb is exact. c's 24-bit mantissa is placed in a frame
// 2 guard bits below P's lsb; if c is far above P, c is placed at bit 52 and P becomes a sticky 1 in bit 0; if c
// is far below P, it is shifted down with a sticky. A signed 80-bit add, normalize, round to 24 bits.
module fp32_fma (
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    output reg  [31:0] y
);
    localparam [31:0] QNAN = 32'h7fc00000;
    reg        sa, sb, sc, sp;
    reg [7:0]  ea, eb, ec;
    reg [23:0] ma, mb, mc;
    reg        za, zb, zc, ia, ib, ic, na, nb, nc;
    reg [47:0] P;
    reg signed [12:0] ep, ecl, sh, wf, E;
    reg [79:0] Pf, Cf, M;
    reg signed [80:0] S;
    reg        st;
    integer    k, L;
    reg [24:0] m;
    reg [79:0] mask;
    reg        rb, sbit;

    always @* begin
        sa = a[31]; sb = b[31]; sc = c[31];
        ea = a[30:23]; eb = b[30:23]; ec = c[30:23];
        za = (ea == 0); zb = (eb == 0); zc = (ec == 0);
        ia = (ea == 8'hff) && (a[22:0] == 0); ib = (eb == 8'hff) && (b[22:0] == 0); ic = (ec == 8'hff) && (c[22:0] == 0);
        na = (ea == 8'hff) && (a[22:0] != 0); nb = (eb == 8'hff) && (b[22:0] != 0); nc = (ec == 8'hff) && (c[22:0] != 0);
        ma = za ? 24'd0 : {1'b1, a[22:0]};
        mb = zb ? 24'd0 : {1'b1, b[22:0]};
        mc = zc ? 24'd0 : {1'b1, c[22:0]};
        sp = sa ^ sb;
        y = 32'd0;
        P = ma * mb;
        Pf = 80'd0; Cf = 80'd0; S = 81'sd0; M = 80'd0; E = 0; L = 0; m = 25'd0;
        ep = $signed({5'd0, ea}) + $signed({5'd0, eb}) - 13'sd300;     // weight of P's bit 0
        ecl = $signed({5'd0, ec}) - 13'sd150;                          // weight of mc's bit 0
        sh = ecl - ep;
        wf = ep - 2;
        if (na || nb || nc) y = QNAN;
        else if ((ia || ib) && (za || zb)) y = QNAN;                    // inf * 0
        else if (ia || ib) begin
            if (ic && (sc != sp)) y = QNAN;                             // inf - inf
            else y = {sp, 8'hff, 23'd0};
        end else if (ic) y = c;
        else if (za || zb) y = zc ? 32'd0 : c;                          // product is zero: exactly c
        else begin
            // place the operands in the frame (lsb weight wf)
            Pf = {30'd0, P, 2'b00};
            if (zc) Cf = 80'd0;
            else if (sh > 50) begin                                    // c far above: P is only sticky
                Cf = {4'd0, mc, 52'd0};
                Pf = 80'd1;
                wf = ecl - 52;
            end else if (sh >= -2) begin
                Cf = {56'd0, mc} << (sh + 2);
            end else begin                                             // c below P's lsb: shift down, sticky
                if (-sh - 2 >= 26) begin Cf = 80'd1; end
                else begin
                    mask = (80'd1 << (-sh - 2)) - 1;
                    st = |({56'd0, mc} & mask);
                    Cf = ({56'd0, mc} >> (-sh - 2));
                    Cf[0] = Cf[0] | st;
                end
            end
            S = (sp ? -$signed({1'b0, Pf}) : $signed({1'b0, Pf})) + (sc ? -$signed({1'b0, Cf}) : $signed({1'b0, Cf}));
            if (S == 0) y = 32'd0;
            else begin
                M = S[80] ? -S : S;
                L = 0;
                for (k = 0; k < 80; k = k + 1) if (M[k]) L = k;
                // round to 24 bits: keep [L : L-23]
                if (L <= 23) begin
                    m = {1'b0, M[23:0]} << (23 - L);
                end else begin
                    m = {1'b0, M >> (L - 23)} ;
                    m = m & 25'h0ffffff;
                    rb = M[L - 24];
                    mask = (80'd1 << (L - 24)) - 1;
                    sbit = |(M & mask);
                    if (rb && (sbit || m[0])) m = m + 1;
                end
                E = L + wf + 127;
                if (m[24]) begin m = m >> 1; E = E + 1; end
                if (E >= 255) y = {S[80], 8'hff, 23'd0};
                else if (E <= 0) y = 32'd0;
                else y = {S[80], E[7:0], m[22:0]};
            end
        end
    end
endmodule
