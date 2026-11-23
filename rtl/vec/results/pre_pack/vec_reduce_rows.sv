// R1 over every row and plane of one command (RSUM, RSUMSQ), with CTX rows in flight on the shared bank.
//
// The bank's FMA takes 8 clocks, so one row's chain (its accumulation beats, then nine tree levels) can
// issue only every 9th clock. CTX contexts take slots 0..CTX-1 of a 9-clock calendar: context i issues
// operation j of its row at enabled clock 9j + i. Each context holds one 512-lane accumulator, captured
// from the bank the clock before its next slot, so every lane's sequence of partial sums and every tree
// pair is exactly the one-row-at-a-time order (R1): only independent rows interleave.
//
// A context's slot that needs an SRAM beat which is not ready freezes the bank, the calendar and the
// tags together (bank_ce), so no result buffer is needed. Finished sums wait in a per-context result
// register; a context takes a new row only after its result is written. Results can retire out of
// row order; every row writes its own scalar.
module vec_reduce_rows #(parameter LANES=512, parameter CTX=4)(
    input wire clk, reset, start, square,
    input wire [15:0] cols, rows, planes,
    // SRAM beat for (read_row, read_plane, read_col); x is already masked beyond cols
    output wire read_valid, input wire read_ready,
    output wire [15:0] read_row, read_plane, read_col,
    input wire [LANES*32-1:0] x,
    // one scalar result per row
    output wire write_valid, input wire write_ready,
    output wire [15:0] write_row, write_plane, output wire [31:0] result,
    output reg done, output wire busy,
    output wire [LANES*32-1:0] fa_bus, fb_bus, fc_bus,
    input wire [LANES*32-1:0] bank_result,
    output wire bank_ce
);
    localparam P = 9, LEVELS = $clog2(LANES), CW = $clog2(CTX) > 0 ? $clog2(CTX) : 1;
    reg running;
    reg [3:0] slot;                                   // calendar position, 0..8
    reg [15:0] next_row, next_plane; reg rows_left;
    reg [CTX-1:0] active, pending;
    reg [15:0] c_row[0:CTX-1], c_plane[0:CTX-1], c_col[0:CTX-1];
    reg [4:0] c_level[0:CTX-1];                       // 0 = accumulating, 1..LEVELS = tree level to issue
    reg [31:0] c_result[0:CTX-1];
    reg [LANES*32-1:0] acc[0:CTX-1];
    assign busy = running;

    // ---- the context whose slot this is, and what it issues
    wire [CW-1:0] c = slot[CW-1:0];
    wire in_ctx = running && slot < CTX;
    // a context whose result is being written this clock can already start its next row
    reg [CW-1:0] wsel; reg wany; integer i;
    wire freed = !pending[c] || (wany && wsel == c && write_ready);
    wire take = in_ctx && !active[c] && freed && rows_left;      // starts a new row with beat 0
    wire cont = in_ctx && active[c];
    wire tree = cont && c_level[c] != 0;
    wire beat = take || (cont && c_level[c] == 0);
    wire first = take;
    wire [15:0] row_now = take ? next_row : c_row[c], plane_now = take ? next_plane : c_plane[c];
    wire [15:0] col_now = take ? 16'd0 : c_col[c];
    wire last_beat = {1'b0, col_now} + LANES >= cols;
    wire issue = beat || tree;
    assign read_valid = beat;
    assign read_row = row_now; assign read_plane = plane_now; assign read_col = col_now;
    wire ce = !(beat && !read_ready);
    assign bank_ce = running ? ce : 1'b1;

    wire [LANES*32-1:0] cur = acc[c];
    genvar j;
    for (j = 0; j < LANES; j = j + 1) begin : lane
        wire [31:0] left = 2*j+1 < LANES ? cur[64*j+:32] : 32'd0;
        wire [31:0] right = 2*j+1 < LANES ? cur[64*j+32+:32] : 32'd0;
        wire [31:0] sample = x[32*j+:32];
        assign fa_bus[32*j+:32] = tree ? left : sample;
        assign fb_bus[32*j+:32] = tree || !square ? 32'h3f800000 : sample;
        assign fc_bus[32*j+:32] = tree ? right : first ? 32'd0 : cur[32*j+:32];
    end

    // ---- issue tags, aligned with the bank's 8-clock result
    wire tv, t_final; wire [CW-1:0] tc;
    wire final_level = tree && c_level[c] == LEVELS;
    vec_row_tag_delay #(CW+1, 8) tags(clk, reset, ce, issue, {final_level, c}, tv, {t_final, tc});

    // ---- result write: lowest pending context
    always @* begin
        wany = 0; wsel = 0;
        for (i = CTX-1; i >= 0; i = i - 1) if (pending[i]) begin wany = 1; wsel = i; end
    end
    assign write_valid = wany;
    assign write_row = c_row[wsel]; assign write_plane = c_plane[wsel]; assign result = c_result[wsel];

    always @(posedge clk) begin
        done <= 0;
        if (reset) begin running <= 0; active <= 0; pending <= 0; end
        else if (!running) begin
            if (start) begin
                slot <= 0; next_row <= 0; next_plane <= 0; active <= 0; pending <= 0;
                rows_left <= cols != 0 && rows != 0 && planes != 0;
                if (cols == 0 || rows == 0 || planes == 0) done <= 1; else running <= 1;
            end
        end else begin
            if (write_valid && write_ready) pending[wsel] <= 0;
            if (ce) begin
                slot <= slot == P-1 ? 4'd0 : slot + 1'b1;
                if (take) begin
                    active[c] <= 1; c_row[c] <= next_row; c_plane[c] <= next_plane;
                    if (next_row != rows-1'b1) next_row <= next_row + 1'b1;
                    else begin
                        next_row <= 0;
                        if (next_plane != planes-1'b1) next_plane <= next_plane + 1'b1; else rows_left <= 0;
                    end
                end
                if (beat) begin
                    c_col[c] <= col_now + LANES;
                    c_level[c] <= last_beat ? 5'd1 : 5'd0;
                end else if (tree) c_level[c] <= c_level[c] + 1'b1;
                if (tv) begin
                    acc[tc] <= bank_result;
                    if (t_final) begin c_result[tc] <= bank_result[31:0]; pending[tc] <= 1; active[tc] <= 0; end
                end
            end
            if (!rows_left && active == 0 && pending == 0) begin running <= 0; done <= 1; end
        end
    end
endmodule
