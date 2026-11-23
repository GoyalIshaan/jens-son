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
// Packed mode (pack_lg != 0, cols = 2^pack_lg < LANES): one beat carries G = LANES / cols whole rows.
// Lane j holds column j % cols of slot k = j / cols. Slots run along rows (m, m+1, ...) or, when
// pack_p, along planes (rows == 1). The beat's (m, p) is slot 0's; a partial last group masks its slots.
module vec_box_walk #(parameter LANES = 512)(
    input wire clk, reset, start, advance,
    input wire [15:0] n_count, m_count, p_count,
    output reg active, done,
    output reg [15:0] n, m, p,
    output wire [LANES-1:0] mask,
    input wire [3:0] pack_lg, input wire pack_p
);
    reg [15:0] cols, rows, planes;
    wire [16:0] end_col = {1'b0,n} + LANES;
    wire pkd = pack_lg != 0;
    wire [16:0] group = 17'd1 << ($clog2(LANES) - pack_lg);          // rows per packed beat
    wire [16:0] left = pack_p ? {1'b0,planes} - p : {1'b0,rows} - m; // rows (or planes) not yet issued
    wire [16:0] next_m = {1'b0,m} + group, next_p = {1'b0,p} + group;
    genvar i;
    for (i=0; i<LANES; i=i+1) begin : lane_mask
        wire [16:0] slot = i >> pack_lg;
        assign mask[i] = active && (pkd ? slot < left : {1'b0,n} + i < cols);
    end
    always @(posedge clk) begin
        done <= 0;
        if (reset) active <= 0;
        else if (!active && start) begin
            {cols,rows,planes} <= {n_count,m_count,p_count};
            {n,m,p} <= 0;
            active <= n_count != 0 && m_count != 0 && p_count != 0;
            done <= n_count == 0 || m_count == 0 || p_count == 0;
        end else if (active && advance) begin
            if (pkd && pack_p) begin
                if (next_p < planes) p <= next_p[15:0];
                else begin active <= 0; done <= 1; end
            end else if (pkd) begin
                if (next_m < rows) m <= next_m[15:0];
                else begin
                    m <= 0;
                    if (p != planes-1'b1) p <= p+1'b1;
                    else begin active <= 0; done <= 1; end
                end
            end else if (end_col < cols) n <= end_col[15:0];
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
