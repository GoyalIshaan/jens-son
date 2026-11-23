// mxu_row_pipe against mxu_row: random beats (lut4-like scales, int8, seg4 on and off) under a random clock enable.
module tb_mxu_row_pipe;
    reg clk = 0, ce = 0;
    reg [8191:0] w, act; reg [255:0] gM; reg [63:0] gS; reg [639:0] gE; reg [255:0] aexp; reg seg4;
    wire [31:0] r0, r1, r2, r3, p0, p1, p2, p3;
    mxu_row ref_row(.w(w), .gM(gM), .gS(gS), .gE(gE), .act(act), .aexp(aexp), .seg4(seg4), .v0(r0), .v1(r1), .v2(r2), .v3(r3));
    mxu_row_pipe dut(.clk(clk), .ce(ce), .w(w), .gM(gM), .gS(gS), .gE(gE), .act(act), .aexp(aexp), .seg4(seg4),
                     .v0(p0), .v1(p1), .v2(p2), .v3(p3));
    reg [128:0] hist [0:15]; integer k, i, filled = 0, bad = 0, n = 0;
    initial begin
        for (k = 0; k < 3000; k = k + 1) begin
            for (i = 0; i < 256; i = i + 1) begin w[32*i+:32] = $random; act[32*i+:32] = $random; end
            for (i = 0; i < 64; i = i + 1) begin
                gM[4*i+:4] = (k % 3 == 0) ? 4'd1 : $random; gS[i] = (k % 3 == 0) ? 0 : $random;
                gE[10*i+:10] = (k % 3 == 0) ? 0 : ($random % 12) - 9;
            end
            for (i = 0; i < 32; i = i + 1) aexp[8*i+:8] = (k % 4 == 1) ? $random : ($random % 40) - 20;
            seg4 = k % 2;
            ce = ($random & 3) != 0;
            #1;
            if (ce) begin
                if (filled >= 8) begin : check
                    reg [128:0] h; h = hist[(filled - 8) % 16];
                    n = n + 1;
                    if (p0 !== h[127:96] || (h[128] && {p1, p2, p3} !== h[95:0])) bad = bad + 1;
                end
                hist[filled % 16] = {seg4, r0, r1, r2, r3};
                filled = filled + 1;
            end
            clk = 1; #1 clk = 0;
        end
        if (bad) $display("FAIL %0d of %0d", bad, n); else $display("PASS %0d beats", n);
        $finish;
    end
endmodule
