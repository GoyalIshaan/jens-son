// Reductions (RSUM, RSUMSQ) by order rule R1 (golden/bitexact_vec.py reduce_sum):
//   lane j accumulates x[j], x[j+N], x[j+2N], ... in sequence: acc = acc + x (RSUM) or fma(x, x, acc) (RSUMSQ),
//   using the lane's own FMA; then a pairwise tree over the N lanes, level by level: (0+1), (2+3), ...
// Missing lanes (n not a multiple of N) hold +0.

// Pairwise tree, combinational, N a power of two. Every add is fp32_fma(a, 1, b): one rounding, flush to +0.
module reduce_tree #(parameter N = 512) (
    input  wire [N*32-1:0] x,
    output wire [31:0]     y
);
    generate
        if (N == 1) begin : leaf
            assign y = x[31:0];
        end else begin : lvl
            wire [(N/2)*32-1:0] s;
            genvar i;
            for (i = 0; i < N / 2; i = i + 1) begin : add
                fp32_fma f (.a(x[64*i +: 32]), .b(32'h3f800000), .c(x[64*i + 32 +: 32]), .y(s[32*i +: 32]));
            end
            reduce_tree #(.N(N / 2)) next (.x(s), .y(y));
        end
    endgenerate
endmodule

// One reduction command over beats of N lanes: `first` starts it, `last` ends it (y valid the next cycle).
module vec_reduce #(parameter N = 512) (
    input  wire            clk,
    input  wire            en,
    input  wire            first,
    input  wire            last,
    input  wire            square,
    input  wire [N*32-1:0] x,
    output reg  [31:0]     y,
    output reg             y_valid
);
    reg  [N*32-1:0] acc;
    wire [N*32-1:0] nxt;
    wire [31:0]     t;
    genvar j;
    generate
        for (j = 0; j < N; j = j + 1) begin : lane
            wire [31:0] a0 = first ? 32'd0 : acc[32*j +: 32];
            fp32_fma f (.a(x[32*j +: 32]), .b(square ? x[32*j +: 32] : 32'h3f800000), .c(a0), .y(nxt[32*j +: 32]));
        end
    endgenerate
    reduce_tree #(.N(N)) tree (.x(nxt), .y(t));
    always @(posedge clk) begin
        y_valid <= 1'b0;
        if (en) begin
            acc <= nxt;
            if (last) begin y <= t; y_valid <= 1'b1; end
        end
    end
endmodule
