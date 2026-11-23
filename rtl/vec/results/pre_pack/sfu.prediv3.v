`include "vec_delay.vh"
// The special-function unit (D-020), bit-exact with golden/chip_sfu.py.
// Every function is a fixed chain of fp32 operations (each one fp32_fma = one IEEE rounding) plus table ROMs
// (sfu_tables.vh, generated from chip_sfu's tables) and exact exponent/bit manipulations. Combinational here;
// the pipelined version registers between the fma stages.
//
// Two conventions follow the golden model's torch arithmetic exactly:
//   - round_bits(v, b) keeps b mantissa bits (round half to even); an input >= 2^127 becomes +-inf
//     (torch computes ldexp(m, 128) through 2^128 = inf)
//   - ldexp results below 2^-126 flush to 0
// Inputs are normal, zero or -inf (exp). RSQRT is defined for positive inputs (the program adds eps first).

`define ONE   32'h3f800000
`define TWO   32'h40000000
`define HALF  32'h3f000000
`define THREEHALF 32'h3fc00000
`define NEG(v) {~v[31], v[30:0]}

// LAT is 0 for the combinational reference or 8 for the streaming datapath. Every pipelined core also has
// a _ce form: with ce low all its registers hold, so a streaming command can freeze it on a write stall.
// LAT 9: an input register in front of the 8-stage FMA, so the glue logic feeding an FMA (table lookups,
// exponent fixes, rounding) is not on the same path as its multiplier. Used where that glue is long.
module sfu_madd #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] a,b,c, output wire [31:0] y);
    if (LAT == 0) fp32_fma comb(a,b,c,y);
    else if (LAT == 9) begin : registered_inputs
        reg [31:0] ra, rb, rc;
        always @(posedge clk) if (ce) begin ra <= a; rb <= b; rc <= c; end
        fp32_fma_pipe8_ce pipe(clk,ce,ra,rb,rc,y);
    end else fp32_fma_pipe8_ce pipe(clk,ce,a,b,c,y);
endmodule

// ---------------------------------------------------------------- shared combinational helpers
// round to b fraction bits, half to even; input >= 2^127 -> inf
module round_bits #(parameter B = 16) (input wire [31:0] v, output reg [31:0] y);
    localparam D = 23 - B;
    reg [24:0] r;
    reg [7:0]  E;
    reg        rb, st;
    always @* begin
        E = v[30:23];
        if (E == 0 || E == 8'hff) y = v;
        else if (E >= 8'd254) y = {v[31], 8'hff, 23'd0};
        else begin
            r = {1'b0, 1'b1, v[22:0]} >> D;
            rb = v[D - 1];
            st = (D > 1) ? |(v[22:0] & ((23'd1 << (D - 1)) - 1)) : 1'b0;
            if (rb && (st || r[0])) r = r + 1;
            if (r[B + 1]) begin r = r >> 1; E = E + 1; end
            y = {v[31], E, r[B - 1:0], {D{1'b0}}};
        end
    end
endmodule

// fp32 RNE of w / 3, w >= 0 normal or zero (results below 2^-126 flush)
module fp32_div3 (input wire [31:0] w, output reg [31:0] y);
    reg [49:0] N, Q;
    reg [1:0]  R;
    reg [24:0] m;
    reg        rb, st;
    reg signed [9:0] E;
    always @* begin
        if (w[30:23] == 0) y = 32'd0;
        else begin
            N = {1'b1, w[22:0], 26'd0};
            Q = N / 3;
            R = N % 3;
            if (Q[48]) begin
                m = {1'b0, Q[48:25]}; rb = Q[24]; st = (|Q[23:0]) || (R != 0); E = $signed({2'b0, w[30:23]}) - 1;
            end else begin
                m = {1'b0, Q[47:24]}; rb = Q[23]; st = (|Q[22:0]) || (R != 0); E = $signed({2'b0, w[30:23]}) - 2;
            end
            if (rb && (st || m[0])) m = m + 1;
            if (m[24]) begin m = m >> 1; E = E + 1; end
            y = (E <= 0) ? 32'd0 : {w[31], E[7:0], m[22:0]};
        end
    end
endmodule

// ---------------------------------------------------------------- e^x (16 bits)
module sfu_exp_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    `include "sfu_tables.vh"
    wire [31:0] yy, f, d, r, rr, q, a1, p, m;
    reg  [31:0] yc, nf, i64, tab, out;
    reg  signed [9:0] n;
    reg  [6:0]  ii;
    reg  [5:0]  i;
    integer     sh;
    reg  [23:0] mant;
    reg  [8:0]  ip;
    reg  signed [10:0] Eo;
    wire [31:0] tab_m;
    vec_delay_ce #(32,5*LAT) delay_tab_m(clk, ce, tab, tab_m);
    wire [31:0] a1_q;
    vec_delay_ce #(32,1*LAT) delay_a1_q(clk, ce, a1, a1_q);
    wire [9:0] n_m;
    vec_delay_ce #(10,7*LAT) delay_n_m(clk, ce, n, n_m);
    sfu_madd #(LAT) u1 (.clk(clk), .ce(ce), .a(x), .b(C_LOG2E), .c(32'd0), .y(yy));           // y = x log2 e
    // clamp to [-160, 127]; floor
    always @* begin
        yc = yy;
        if (yy[31] && (yy[30:0] > 31'h43200000)) yc = 32'hc3200000;   // < -160 (incl. -inf)
        if (!yy[31] && (yy[30:0] > 31'h42fe0000)) yc = 32'h42fe0000;  // > 127
        // floor(yc) as a signed integer
        if (yc[30:23] < 8'd127) n = (yc[31] && yc[30:0] != 0) ? -10'sd1 : 10'sd0;
        else begin
            sh = 150 - yc[30:23];                                      // 23 - unbiased exponent
            mant = {1'b1, yc[22:0]};
            ip = mant >> sh;
            if (!yc[31]) n = ip;
            else n = -$signed({1'b0, ip}) - (((mant & ((24'd1 << sh) - 1)) != 0) ? 10'sd1 : 10'sd0);
        end
        // n as fp32 (exact)
        nf = int2f(n);
    end
    sfu_madd #(LAT) u2 (.clk(clk), .ce(ce), .a(yc), .b(`ONE), .c(`NEG(nf)), .y(f));             // f = y - n, in [0, 1]
    always @* begin
        // i = clamp(floor(f * 64), 0, 63)
        if (f[30:23] == 0) ii = 0;
        else if (f[30:23] >= 8'd127) ii = 7'd64;                       // f = 1.0
        else if (f[30:23] < 8'd121) ii = 0;                            // f < 1/64
        else ii = ({1'b1, f[22:0]} >> (23 - (f[30:23] - 121)));
        i = (ii > 63) ? 6'd63 : ii[5:0];
        i64 = int2f({4'd0, i}) ;
        if (i != 0) i64 = {1'b0, i64[30:23] - 8'd6, i64[22:0]};         // i / 64
        tab = t_exp2(i);
    end
    sfu_madd #(LAT) u3 (.clk(clk), .ce(ce), .a(f), .b(`ONE), .c(`NEG(i64)), .y(d));             // f - i/64 (exact)
    sfu_madd #(LAT) u4 (.clk(clk), .ce(ce), .a(d), .b(C_LN2), .c(32'd0), .y(r));                 // r
    sfu_madd #(LAT) u5 (.clk(clk), .ce(ce), .a(r), .b(r), .c(32'd0), .y(rr));                    // r*r
    sfu_madd #(LAT) u6 (.clk(clk), .ce(ce), .a(rr), .b(`HALF), .c(32'd0), .y(q));                // r*r*0.5
    sfu_madd #(LAT) u7 (.clk(clk), .ce(ce), .a(r), .b(`ONE), .c(`ONE), .y(a1));                  // 1 + r
    sfu_madd #(LAT) u8 (.clk(clk), .ce(ce), .a(a1_q), .b(`ONE), .c(q), .y(p));                     // (1 + r) + r*r*0.5
    sfu_madd #(LAT) u9 (.clk(clk), .ce(ce), .a(tab_m), .b(p), .c(32'd0), .y(m));                   // table x polynomial
    always @* begin                                                    // ldexp(m, n), flush below 2^-126
        Eo = $signed({3'd0, m[30:23]}) + $signed(n_m);
        if (m[30:23] == 0 || $signed(n_m) <= -127 || Eo <= 0) out = 32'd0;
        else out = {m[31], Eo[7:0], m[22:0]};
    end
    round_bits #(.B(16)) rb (.v(out), .y(y));

    function [31:0] int2f;                                             // small signed integer -> fp32
        input signed [9:0] v;
        reg [9:0] a;
        integer k, L;
        begin
            a = v[9] ? -v : v;
            if (a == 0) int2f = 32'd0;
            else begin
                L = 0;
                for (k = 0; k < 10; k = k + 1) if (a[k]) L = k;
                int2f = {v[9], 8'd127 + L[7:0], 23'({a, 23'd0} >> L)};
            end
        end
    endfunction
endmodule

module sfu_exp(input wire [31:0] x, output wire [31:0] y);
    sfu_exp_core #(0) core(1'b0,x,y);
endmodule

// ---------------------------------------------------------------- 1/x (12 bits)
module sfu_recip_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    `include "sfu_tables.vh"
    wire [31:0] m2 = {1'b0, 8'd127, x[22:0]};                          // mantissa in [1, 2)
    wire [31:0] y0 = t_recip(x[22:15]);
    wire [31:0] t, u, y1;
    reg  [31:0] out;
    reg  signed [10:0] Eo;
    wire [31:0] y0_u;
    vec_delay_ce #(32,2*LAT) delay_y0_u(clk, ce, y0, y0_u);
    wire [8:0] sign_exp;
    vec_delay_ce #(9,3*LAT) delay_sign_exp(clk, ce, x[31:23], sign_exp);
    sfu_madd #(LAT) r1 (.clk(clk), .ce(ce), .a(m2), .b(y0), .c(32'd0), .y(t));
    sfu_madd #(LAT) r2 (.clk(clk), .ce(ce), .a(`NEG(t)), .b(`ONE), .c(`TWO), .y(u));            // 2 - m2*y0
    sfu_madd #(LAT) r3 (.clk(clk), .ce(ce), .a(y0_u), .b(u), .c(32'd0), .y(y1));
    always @* begin                                                    // sign(x) * ldexp(y1, 127 - E)
        Eo = $signed({3'd0, y1[30:23]}) + 11'sd127 - $signed({3'd0, sign_exp[7:0]});
        if (sign_exp[7:0] == 0 || Eo <= 0) out = 32'd0;
        else out = {sign_exp[8], Eo[7:0], y1[22:0]};
    end
    round_bits #(.B(12)) rb (.v(out), .y(y));
endmodule

module sfu_recip(input wire [31:0] x, output wire [31:0] y);
    sfu_recip_core #(0) core(1'b0,x,y);
endmodule

// ---------------------------------------------------------------- 1/sqrt(x) (16 bits), x > 0
module sfu_rsqrt_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    `include "sfu_tables.vh"
    reg  signed [9:0] k;
    reg  [31:0] u, h, out;
    wire [31:0] w, qd, y0, t1, t2, s, y1;
    reg  [8:0]  ii;
    reg  [7:0]  i;
    reg  signed [10:0] Eo;
    always @* begin
        k = ($signed({2'b0, x[30:23]}) - 10'sd127) >>> 1;              // floor((E - 127) / 2)
        u = {1'b0, x[30:23] - 8'(2 * k), x[22:0]};                     // x / 4^k, in [1, 4)
        h = {1'b0, u[30:23] - 8'd1, u[22:0]};                          // 0.5 u
    end
    // pipelined (LAT > 0): registers after the divide by 3 and after the table, so the glue between two
    // FMAs is not one path; and after the final round. Latency 5 x LAT + 3.
    localparam G = LAT > 0 ? 1 : 0;
    wire [31:0] h_y0;
    vec_delay_ce #(32,1*LAT+2*G) delay_h_y0(clk, ce, h, h_y0);
    wire [31:0] y0_t1;
    vec_delay_ce #(32,1*LAT) delay_y0_t1(clk, ce, y0, y0_t1);
    wire [31:0] y0_s;
    vec_delay_ce #(32,2*LAT) delay_y0_s(clk, ce, y0_t1, y0_s);
    wire [9:0] k_result;
    vec_delay_ce #(10,5*LAT+2*G) delay_k_result(clk, ce, k, k_result);
    sfu_madd #(LAT) q1 (.clk(clk), .ce(ce), .a(u), .b(`ONE), .c(32'hbf800000), .y(w));           // u - 1
    wire [31:0] qd_c, y0_c;
    fp32_div3 q2 (.w(w), .y(qd_c));                                    // (u - 1) / 3
    vec_delay_ce #(32,G) qd_reg(clk, ce, qd_c, qd);
    always @* begin                                                    // i = clamp(floor(qd * 256), 0, 255)
        if (qd[30:23] == 0 || qd[30:23] < 8'd119) ii = 0;
        else ii = ({1'b1, qd[22:0]} >> (23 - (qd[30:23] - 119)));
        i = (ii > 255) ? 8'd255 : ii[7:0];
    end
    assign y0_c = t_rsqrt(i);
    vec_delay_ce #(32,G) y0_reg(clk, ce, y0_c, y0);
    sfu_madd #(LAT) q3 (.clk(clk), .ce(ce), .a(h_y0), .b(y0), .c(32'd0), .y(t1));                   // 0.5 u y0
    sfu_madd #(LAT) q4 (.clk(clk), .ce(ce), .a(t1), .b(y0_t1), .c(32'd0), .y(t2));                  //   ... y0
    sfu_madd #(LAT) q5 (.clk(clk), .ce(ce), .a(`NEG(t2)), .b(`ONE), .c(`THREEHALF), .y(s));     // 1.5 - that
    sfu_madd #(LAT) q6 (.clk(clk), .ce(ce), .a(y0_s), .b(s), .c(32'd0), .y(y1));
    always @* begin
        Eo = $signed({3'd0, y1[30:23]}) - $signed(k_result);
        out = (Eo <= 0) ? 32'd0 : {y1[31], Eo[7:0], y1[22:0]};
    end
    wire [31:0] y_c;
    round_bits #(.B(16)) rb (.v(out), .y(y_c));
    vec_delay_ce #(32,G) out_reg(clk, ce, y_c, y);
endmodule

module sfu_rsqrt(input wire [31:0] x, output wire [31:0] y);
    sfu_rsqrt_core #(0) core(1'b0,x,y);
endmodule

// ---------------------------------------------------------------- sigmoid (12 bits) and SiLU
module sfu_sigmoid_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    wire [31:0] z, oz, s, zs, zsr;
    wire nonneg = !x[31] || (x[30:0] == 0);
    wire [31:0] z_s;
    vec_delay_ce #(32,4*LAT) delay_z_s(clk, ce, z, z_s);
    wire [31:0] s_zsr;
    vec_delay_ce #(32,1*LAT) delay_s_zsr(clk, ce, s, s_zsr);
    wire [0:0] positive_out;
    vec_delay_ce #(1,13*LAT) delay_positive_out(clk, ce, nonneg, positive_out);
    localparam G = LAT > 0 ? 1 : 0;                   // pipelined: the round and select get their own stage
    sfu_exp_core_ce #(LAT) e1 (.clk(clk), .ce(ce), .x({1'b1, x[30:0]}), .y(z));                         // exp(-|x|)
    sfu_madd #(LAT) g1 (.clk(clk), .ce(ce), .a(z), .b(`ONE), .c(`ONE), .y(oz));                  // 1 + z
    sfu_recip_core_ce #(LAT) r1 (.clk(clk), .ce(ce), .x(oz), .y(s));
    sfu_madd #(LAT) g2 (.clk(clk), .ce(ce), .a(z_s), .b(s), .c(32'd0), .y(zs));
    round_bits #(.B(12)) rb (.v(zs), .y(zsr));
    wire [31:0] y_c = positive_out ? s_zsr : zsr;
    vec_delay_ce #(32,G) out_reg(clk, ce, y_c, y);
endmodule

module sfu_sigmoid(input wire [31:0] x, output wire [31:0] y);
    sfu_sigmoid_core #(0) core(1'b0,x,y);
endmodule

module sfu_silu_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    wire [31:0] s;
    wire [31:0] x_s;
    vec_delay_ce #(32,13*LAT+(LAT > 0 ? 1 : 0)) delay_x_s(clk, ce, x, x_s);
    sfu_sigmoid_core_ce #(LAT) sg (.clk(clk), .ce(ce), .x(x), .y(s));
    sfu_madd #(LAT) m1 (.clk(clk), .ce(ce), .a(x_s), .b(s), .c(32'd0), .y(y));
endmodule

module sfu_silu(input wire [31:0] x, output wire [31:0] y);
    sfu_silu_core #(0) core(1'b0,x,y);
endmodule

// ---------------------------------------------------------------- softplus (12 bits)
module sfu_softplus_core_ce #(parameter LAT = 0) (input wire clk, input wire ce, input wire [31:0] x, output wire [31:0] y);
    `include "sfu_tables.vh"
    wire [31:0] z, u, w1, r, a, b, rr, cc, l2, le, big, d3, s1, s2, s3, sml, sum;
    reg  [31:0] mm, eef, i64, xp;
    reg  [5:0]  i;
    always @* begin
        // u = 1 + z in [1, 2]: mantissa mm in [1, 2), ee = 0 or 1
        mm  = {1'b0, 8'd127, u[22:0]};
        eef = (u[30:23] == 8'd128) ? `ONE : 32'd0;
        i   = u[22:17];
        i64 = (i == 0) ? 32'd0 : int6_div64(i);
        xp  = (x[31] && x[30:0] != 0) ? 32'd0 : x;                     // max(x, 0); -0 stays (adds as 0)
    end
    wire use_small = (z[30:23] < 8'd121);                              // z < 2^-6
    // pipelined: registers before and after the divide by 3 (the small-z branch runs 2G later and is
    // realigned) and after the final round. Latency 17 x LAT + 1.
    localparam G = LAT > 0 ? 1 : 0;
    wire [31:0] i64_w1;
    vec_delay_ce #(32,1*LAT) delay_i64_w1(clk, ce, i64, i64_w1);
    wire [5:0] i_r;
    vec_delay_ce #(6,2*LAT) delay_i_r(clk, ce, i, i_r);
    wire [5:0] i_a;
    vec_delay_ce #(6,1*LAT) delay_i_a(clk, ce, i_r, i_a);
    wire [31:0] eef_l2;
    vec_delay_ce #(32,5*LAT) delay_eef_l2(clk, ce, eef, eef_l2);
    wire [31:0] z_s1;
    vec_delay_ce #(32,1*LAT+2*G) delay_z_s1(clk, ce, z, z_s1);
    wire [31:0] z_s3;
    vec_delay_ce #(32,2*LAT) delay_z_s3(clk, ce, z_s1, z_s3);
    wire [0:0] select_small;
    vec_delay_ce #(1,8*LAT) delay_select_small(clk, ce, use_small, select_small);
    wire [31:0] small_ready;
    vec_delay_ce #(32,4*LAT-2*G) delay_small_ready(clk, ce, sml, small_ready);
    wire [31:0] positive_x;
    vec_delay_ce #(32,16*LAT) delay_positive_x(clk, ce, xp, positive_x);
    sfu_exp_core_ce #(LAT) e1 (.clk(clk), .ce(ce), .x({1'b1, x[30:0]}), .y(z));                          // z = exp(-|x|)
    sfu_madd #(LAT) p1 (.clk(clk), .ce(ce), .a(z),  .b(`ONE), .c(`ONE), .y(u));                  // 1 + z
    sfu_madd #(LAT) p2 (.clk(clk), .ce(ce), .a(mm), .b(`ONE), .c(32'hbf800000), .y(w1));         // mm - 1
    sfu_madd #(LAT) p3 (.clk(clk), .ce(ce), .a(w1), .b(`ONE), .c(`NEG(i64_w1)), .y(r));             //   - i/64
    sfu_madd #(LAT) p4 (.clk(clk), .ce(ce), .a(r),  .b(t_log2_1(i_r)), .c(32'd0), .y(a));          // r t1
    sfu_madd #(LAT) p5 (.clk(clk), .ce(ce), .a(a),  .b(`ONE), .c(t_log2_0(i_a)), .y(b));           // t0 + r t1
    sfu_madd #(LAT) p6 (.clk(clk), .ce(ce), .a(r),  .b(r), .c(32'd0), .y(rr));
    sfu_madd #(LAT) p7 (.clk(clk), .ce(ce), .a(rr), .b(t_log2_2(i_a)), .c(32'd0), .y(cc));         // r r t2
    sfu_madd #(LAT) p8 (.clk(clk), .ce(ce), .a(b),  .b(`ONE), .c(cc), .y(l2));
    sfu_madd #(LAT) p9 (.clk(clk), .ce(ce), .a(l2), .b(`ONE), .c(eef_l2), .y(le));                  // l2 + ee
    sfu_madd #(LAT) pa (.clk(clk), .ce(ce), .a(le), .b(C_LN2), .c(32'd0), .y(big));
    wire [31:0] d3_c, z_q;
    vec_delay_ce #(32,G) z_reg(clk, ce, z, z_q);
    fp32_div3 pb (.w(z_q), .y(d3_c));                                    // small z: z (1 - z (0.5 - z/3))
    vec_delay_ce #(32,G) d3_reg(clk, ce, d3_c, d3);
    sfu_madd #(LAT) pc (.clk(clk), .ce(ce), .a(`NEG(d3)), .b(`ONE), .c(`HALF), .y(s1));
    sfu_madd #(LAT) pd (.clk(clk), .ce(ce), .a(z_s1),  .b(s1), .c(32'd0), .y(s2));
    sfu_madd #(LAT) pe (.clk(clk), .ce(ce), .a(`NEG(s2)), .b(`ONE), .c(`ONE), .y(s3));
    sfu_madd #(LAT) pf (.clk(clk), .ce(ce), .a(z_s3),  .b(s3), .c(32'd0), .y(sml));
    sfu_madd #(LAT) pg (.clk(clk), .ce(ce), .a(positive_x), .b(`ONE), .c(select_small ? small_ready : big), .y(sum));
    wire [31:0] y_c;
    round_bits #(.B(12)) rb (.v(sum), .y(y_c));
    vec_delay_ce #(32,G) out_reg(clk, ce, y_c, y);

    function [31:0] int6_div64;                                        // i / 64 for i in 1..63, exact
        input [5:0] v;
        integer k, L;
        begin
            L = 0;
            for (k = 0; k < 6; k = k + 1) if (v[k]) L = k;
            int6_div64 = {1'b0, 8'd121 + L[7:0], 23'({v, 23'd0} >> L)};
        end
    endfunction
endmodule

module sfu_softplus(input wire [31:0] x, output wire [31:0] y);
    sfu_softplus_core #(0) core(1'b0,x,y);
endmodule

// exp with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_exp_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_exp_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule

// recip with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_recip_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_recip_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule

// rsqrt with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_rsqrt_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_rsqrt_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule

// sigmoid with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_sigmoid_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_sigmoid_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule

// silu with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_silu_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_silu_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule

// softplus with the clock enable tied high (combinational reference when LAT = 0, free-running pipeline otherwise)
module sfu_softplus_core #(parameter LAT = 0) (input wire clk, input wire [31:0] x, output wire [31:0] y);
    sfu_softplus_core_ce #(LAT) core(clk, 1'b1, x, y);
endmodule
