// Byte addressing for an ISA v1 operand. base includes the scalar register offset.
// The inner index of a grouped e8 source advances once per 32 elements.
module vec_box_addr(
    input wire [22:0] base,
    input wire signed [19:0] sn, sm, sp,
    input wire [15:0] n, m, p,
    input wire grouped,
    output wire [22:0] addr
);
    wire signed [16:0] col = {1'b0, grouped ? n >> 5 : n};
    assign addr = $signed({1'b0,base}) + col * sn + $signed({1'b0,m}) * sm + $signed({1'b0,p}) * sp;
endmodule

// Walk descriptors in (p,m,n) order. Hold the current beat until accepted.
// A command completes only after its last beat has retired at the consumer.
module vec_box_walk #(parameter LANES = 512)(
    input wire clk, reset, start, advance,
    input wire [15:0] n_count, m_count, p_count,
    output reg active, done,
    output reg [15:0] n, m, p,
    output wire [LANES-1:0] mask
);
    reg [15:0] cols, rows, planes;
    wire [16:0] end_col = {1'b0,n} + LANES;
    genvar i;
    for (i=0; i<LANES; i=i+1) assign mask[i] = active && ({1'b0,n} + i < cols);
    always @(posedge clk) begin
        done <= 0;
        if (reset) active <= 0;
        else if (!active && start) begin
            {cols,rows,planes} <= {n_count,m_count,p_count};
            {n,m,p} <= 0;
            active <= n_count != 0 && m_count != 0 && p_count != 0;
            done <= n_count == 0 || m_count == 0 || p_count == 0;
        end else if (active && advance) begin
            if (end_col < cols) n <= end_col[15:0];
            else begin
                n <= 0;
                if (m != rows-1'b1) m <= m+1'b1;
                else begin
                    m <= 0;
                    if (p != planes-1'b1) p <= p+1'b1;
                    else begin active <= 0; done <= 1; end
                end
            end
        end
    end
endmodule
