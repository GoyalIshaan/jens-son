// One vector-unit lane's elementwise ALU (ISA v1 V slot: ADD SUB MUL MAX FMA, with the `neg` flag).
// All arithmetic goes through one fp32_fma, so every op rounds once (rule R2, golden/bitexact_vec.py):
//   ADD  y = fma(a', 1, b)      SUB  y = fma(a', 1, -b)      MUL  y = fma(a', b, +0)      FMA  y = fma(a', b, c)
//   MAX  y = max(a', b)         where a' = -a when neg = 1
module vec_alu (
    input  wire [2:0]  op,          // 0 ADD, 1 SUB, 2 MUL, 3 FMA, 4 MAX
    input  wire        neg,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [31:0] c,
    output reg  [31:0] y
);
    localparam [31:0] ONE = 32'h3f800000;
    wire [31:0] an = {a[31] ^ neg, a[30:0]};
    // operand selection (continuous: never in the same block as the result, which would form a loop)
    wire [31:0] fa = an;
    wire [31:0] fb = (op == 3'd2 || op == 3'd3) ? b : ONE;
    wire [31:0] fc = (op == 3'd0) ? b : (op == 3'd1) ? {~b[31], b[30:0]} : (op == 3'd3) ? c : 32'd0;
    wire [31:0] fy;
    fp32_fma fma (.a(fa), .b(fb), .c(fc), .y(fy));

    // fp32 ordering as signed-magnitude integers (no NaN handling needed beyond propagation)
    function lt;
        input [31:0] x, z;
        begin
            if (x[31] != z[31]) lt = x[31] && !((x[30:0] == 0) && (z[30:0] == 0));
            else if (x[31]) lt = x[30:0] > z[30:0];
            else lt = x[30:0] < z[30:0];
        end
    endfunction
    wire anan = (an[30:23] == 8'hff) && (an[22:0] != 0);
    wire bnan = (b[30:23] == 8'hff) && (b[22:0] != 0);

    always @* begin
        if (op == 3'd4) y = (anan || bnan) ? 32'h7fc00000 : (lt(an, b) ? b : an);
        else y = fy;
    end
endmodule
