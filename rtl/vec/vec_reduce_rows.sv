// R1 over every row and plane of one command (RSUM, RSUMSQ), on the shared FMA bank.
//
// The bank alternates two kinds of clock (phase A, phase B), so it never reads and writes in one clock:
//   A: read one beat and accumulate it on all LANES lanes: fma(x, x or 1, acc). acc is 0 on a row's
//      first beat; on later beats it is the same row's previous partial sum, which leaves the 8-stage
//      FMA exactly now (rows go in waves of 4 over the A clocks, so a row's beats are 8 clocks apart).
//   B: the whole pairwise tree at once, one level per lane slice: level 1 on lanes [0, L/2), level 2 on
//      [L/2, 3L/4), ..., level log2(L) on lane L-2 (L-1 lanes in all). Level 1 takes the row sums that
//      left the bank in the A clock just before; level l takes level l-1's results, which leave the bank
//      now, 8 clocks (four B clocks) after they issued. Up to log2(L) beats are in the tree, one per level.
// Short rows pack: a power-of-two row of n < L values shares the beat with L/n - 1 neighbours (along
// rows, or along planes when there is one row). The tree's pairs never cross a row, and level log2(n)
// holds one sum per row. The levels above it would only add +0 lanes, which changes no value (an
// accumulated lane is never -0: it starts at +0), so they are skipped and every bit is the one-row R1.
// A row's results are written in the B clock they leave the bank, one lane per packed row.
// A read that is not ready (A) or a write that is not ready (B) freezes the bank, the calendar and the
// tags together (bank_ce), so no result buffer is needed.
module vec_reduce_rows #(parameter LANES=512, parameter CTX=4)(
    input wire clk, reset, start, square,
    input wire [15:0] cols, rows, planes,
    // SRAM beat: slot 0 at (read_row, read_plane, read_col); with packing, lane j reads column j % cols
    // of slot j / cols (pack_lg = log2 cols, 0 when not packed). x is zeroed here outside read_mask.
    output wire read_valid, input wire read_ready,
    output wire [15:0] read_row, read_plane, read_col,
    output wire [LANES-1:0] read_mask,
    input wire [LANES*32-1:0] x,
    // results: lane k is slot k's sum (row write_row + k, or plane write_plane + k when pack_p)
    output wire write_valid, input wire write_ready,
    output wire [15:0] write_row, write_plane,
    output wire [LANES-1:0] write_mask,
    output wire [LANES*32-1:0] result,
    output wire [3:0] pack_lg, output wire pack_p,
    output reg done, output wire busy,
    output wire [LANES*32-1:0] fa_bus, fb_bus, fc_bus,
    input wire [LANES*32-1:0] bank_result,
    output wire bank_ce
);
    localparam LEVELS = $clog2(LANES), WAVE = 4, TL = 4*LEVELS;
    localparam [31:0] ONE = 32'h3f800000;
    function automatic [3:0] log2_16(input [15:0] v);
        integer b; log2_16 = 0;
        for (b = 0; b < 16; b = b + 1) if (v[b]) log2_16 = b;
    endfunction
    // ---- command constants (the descriptor is stable until done)
    // tree level of lane j in a B clock: level l on lanes [L - L/2^(l-1), L - L/2^l)
    function automatic integer lane_level(input integer jj);
        integer l; lane_level = 0;
        for (l = 1; l <= LEVELS; l = l + 1)
            if (jj >= LANES - (LANES >> (l-1)) && jj < LANES - (LANES >> l)) lane_level = l;
    endfunction
    wire pow2 = cols != 0 && (cols & (cols - 16'd1)) == 0;
    wire pk = pow2 && cols >= 2 && cols < LANES && (rows > 1 || planes > 1);
    assign pack_lg = pk ? log2_16(cols) : 4'd0;
    assign pack_p = rows == 1;
    wire [3:0] lout = pk ? pack_lg : LEVELS;                      // tree level that holds the row sums
    wire [16:0] group = pk ? 17'd1 << (LEVELS - pack_lg) : 17'd1; // rows per beat
    wire [15:0] nbeats = pk ? 16'd1 : (cols + LANES - 1) / LANES; // beats per row

    reg running, ph;                                              // ph 0 = A clock, 1 = B clock
    reg [15:0] next_row, next_plane; reg rows_left;
    reg [WAVE-1:0] w_valid;
    reg [15:0] w_row[0:WAVE-1], w_plane[0:WAVE-1], w_count[0:WAVE-1];
    reg [1:0] w_i; reg [15:0] w_j;                                // wave slot, beat index
    reg [31:0] taken, written;
    assign busy = running;

    // ---- phase A: which row group, which beat
    wire load = w_j == 0;                                         // first beat: take a new group
    wire [16:0] left = pack_p ? {1'b0,planes} - next_plane : {1'b0,rows} - next_row;
    wire [15:0] take_count = !pk ? 16'd1 : left > group ? group[15:0] : left[15:0];
    wire a_clk = running && !ph;
    wire a_take = a_clk && load && rows_left;
    wire a_valid = a_clk && (load ? rows_left : w_valid[w_i]);
    wire [15:0] a_row = load ? next_row : w_row[w_i], a_plane = load ? next_plane : w_plane[w_i];
    wire [15:0] a_count = load ? take_count : w_count[w_i];
    wire a_last = w_j == nbeats - 1'b1;
    assign read_valid = a_valid;
    assign read_row = a_row; assign read_plane = a_plane; assign read_col = w_j * LANES;

    // ---- tags: A issues, aligned with the bank's 8-clock result
    localparam TW = 16 + 16 + 16;
    wire tv, t_last; wire [TW-1:0] t_tag;
    // ---- tree tags: one entry per B clock; level l of an entry issues when it sits at index 4(l-1)
    reg tl_v[0:TL-1]; reg [TW-1:0] tl_tag[0:TL-1];
    reg l1_v; reg [TW-1:0] l1_tag; reg [LANES*32-1:0] l1;          // row sums for the next B clock
    wire b_clk = running && ph;
    wire out_v = b_clk && tl_v[4*lout-1];
    wire [TW-1:0] out_tag = tl_tag[4*lout-1];
    assign write_valid = out_v;
    assign {write_row, write_plane} = out_tag[TW-1:16];
    wire [15:0] out_count = out_tag[15:0];

    wire ce = !(a_valid && !read_ready) && !(out_v && !write_ready);
    assign bank_ce = running ? ce : 1'b1;
    vec_row_tag_delay #(TW+1, 8) tags(clk, reset, ce, a_valid, {a_last, a_row, a_plane, a_count}, tv, {t_last, t_tag});

    genvar j;
    for (j = 0; j < LANES; j = j + 1) begin : lane
        // A: this lane's sample (zero outside the row, or outside the valid slots of a packed beat)
        wire [15:0] slot = j >> pack_lg;
        wire on = pk ? slot < a_count : {1'b0,read_col} + j < cols;
        assign read_mask[j] = a_valid && on;
        wire [31:0] sample = on ? x[32*j+:32] : 32'd0;
        // B: the level and pair this lane computes
        localparam integer LV = lane_level(j);                    // 0: the one lane no level uses
        localparam integer SRC = LV >= 2 ? (LANES - (LANES >> (LV-2))) + 2*(j - (LANES - (LANES >> (LV-1)))) : 2*j;
        wire [31:0] left_v, right_v;
        if (LV == 1) begin : level_one
            assign left_v = l1[32*SRC+:32]; assign right_v = l1[32*SRC+32+:32];
        end else if (LV >= 2) begin : level_up
            assign left_v = bank_result[32*SRC+:32]; assign right_v = bank_result[32*SRC+32+:32];
        end else begin : unused
            assign left_v = 32'd0; assign right_v = 32'd0;
        end
        assign fa_bus[32*j+:32] = ph ? left_v : sample;
        assign fb_bus[32*j+:32] = ph || !square ? ONE : sample;
        assign fc_bus[32*j+:32] = ph ? right_v : load ? 32'd0 : bank_result[32*j+:32];
        // results: slot k's sum sits at lane (base of level lout) + k
        wire [15:0] src = (LANES - (LANES >> (lout-1))) + j;
        assign result[32*j+:32] = src < LANES ? bank_result[32*src+:32] : 32'd0;
        assign write_mask[j] = out_v && j < out_count;
    end

    integer t;
    always @(posedge clk) begin
        done <= 0;
        if (reset) begin running <= 0; l1_v <= 0; for (t = 0; t < TL; t = t + 1) tl_v[t] <= 0; end
        else if (!running) begin
            if (start) begin
                ph <= 0; w_i <= 0; w_j <= 0; w_valid <= 0; next_row <= 0; next_plane <= 0;
                taken <= 0; written <= 0; l1_v <= 0;
                for (t = 0; t < TL; t = t + 1) tl_v[t] <= 0;
                rows_left <= cols != 0 && rows != 0 && planes != 0;
                if (cols == 0 || rows == 0 || planes == 0) done <= 1; else running <= 1;
            end
        end else begin
            if (ce) begin
                ph <= !ph;
                if (!ph) begin                                    // ---- A clock
                    if (load) begin
                        w_valid[w_i] <= rows_left; w_row[w_i] <= next_row; w_plane[w_i] <= next_plane;
                        w_count[w_i] <= take_count;
                    end
                    if (a_take) begin
                        taken <= taken + 1'b1;
                        if (pack_p && pk) begin
                            if ({1'b0,next_plane} + group >= planes) rows_left <= 0;
                            else next_plane <= next_plane + group[15:0];
                        end else if ({1'b0,next_row} + group < rows) next_row <= next_row + group[15:0];
                        else begin
                            next_row <= 0;
                            if (next_plane != planes-1'b1) next_plane <= next_plane + 1'b1; else rows_left <= 0;
                        end
                    end
                    if (w_i == WAVE-1 || nbeats == 1) begin       // a one-beat row needs no wave
                        w_i <= 0;
                        w_j <= a_last ? 16'd0 : w_j + 1'b1;
                    end else w_i <= w_i + 1'b1;
                    // the previous A clock's last beats leave the bank now: they feed the next B clock
                    l1_v <= tv && t_last; l1_tag <= t_tag; l1 <= bank_result;
                end else begin                                    // ---- B clock
                    tl_v[0] <= l1_v; tl_tag[0] <= l1_tag;
                    for (t = 1; t < TL; t = t + 1) begin tl_v[t] <= tl_v[t-1]; tl_tag[t] <= tl_tag[t-1]; end
                    if (out_v) written <= written + 1'b1;
                end
            end
            if (!rows_left && taken == written) begin
                running <= 0; done <= 1;
            end
        end
    end
endmodule
