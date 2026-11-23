// One DMA queue's engine: a 3-D copy of n0 contiguous bytes, repeated n1 times (strides s1), repeated n2 times
// (strides s2). It moves up to PB bytes per clock (512: DRAM's bandwidth at 1 GHz) and never splits a piece
// across a row of n0 bytes.
//
//   KIND 0  W: DRAM -> weight FIFO (requests only while the FIFO has room for everything in flight)
//   KIND 1  R: DRAM -> SRAM
//   KIND 2  X: SRAM -> DRAM
//   KIND 3  L: SRAM -> SRAM, or SRAM -> host (S2H)
//
// DRAM reads are split into request and in-order response, so any latency works: up to OUTST pieces are in
// flight, each remembering its destination and whether it ends its command. A DRAM-reading engine takes its
// next command as soon as the current one has issued its last request (busy drops), as the timing model
// does; each command's done still pulses when its own last byte arrives, so dones come back in order. done pulses when the last byte is written (R, L, W) or when DRAM
// has accepted the last write (X): the ISA's signal points.
//
// Transpose path (L only): a copy of single bytes whose source runs along i1 (n0 = 1, s1_src = 1) and whose
// destination runs along i2 (s2_dst = 1) is a byte transpose. One piece per clock would move 1 byte per clock,
// so instead it goes through a 64 x 64-byte tile: 64 clocks read up to 64 source rows (64 bytes each), then
// 64 clocks write up to 64 destination lines, each a column of the tile. 32 bytes per clock. The source and
// destination must not overlap (every program's transposes are disjoint; alias_lint checks it).
//
// Host input (H2S, L only): the source is the host input FIFO instead of SRAM. A piece moves when the FIFO holds
// its bytes, and is written at the destination; the source address is ignored (the host stream is in order).
module chip_dma #(parameter KIND = 1, OUTST = 256, PB = 512)(
    input wire clk, reset,
    input wire start, input wire to_host, input wire from_host,
    input wire [35:0] src, dst, input wire [23:0] n0, n1, n2,
    input wire signed [31:0] s1_src, s1_dst, s2_src, s2_dst,
    output wire busy, output reg done,
    input wire sram_ready,                     // the SRAM served this clock's read (X, L) and write (L)
    // DRAM read (W, R)
    output wire dr_valid, input wire dr_ready, output wire [35:0] dr_addr, output wire [9:0] dr_len,
    input wire dr_resp_valid, input wire [8*PB-1:0] dr_resp_data,
    // DRAM write (X)
    output wire dw_valid, input wire dw_ready, output wire [35:0] dw_addr, output wire [8*PB-1:0] dw_data,
    output wire [PB-1:0] dw_mask,
    // SRAM read (X, L): data arrives in the same clock
    output wire sr_valid, output wire [22:0] sr_addr, input wire [8*PB-1:0] sr_data,
    // SRAM write (R, L)
    output wire sw_valid, output wire [22:0] sw_addr, output wire [8*PB-1:0] sw_data, output wire [PB-1:0] sw_mask,
    // weight FIFO (W)
    output wire fw_valid, output wire [8*PB-1:0] fw_data, output wire [9:0] fw_len, input wire [31:0] fifo_free,
    // host (S2H)
    output wire h_valid, output wire [8*PB-1:0] h_data, output wire [9:0] h_len,
    // host input (H2S): the FIFO's front PB bytes and its level; a pop takes the piece
    input wire [31:0] hi_level, input wire [8*PB-1:0] hi_data, output wire hi_pop, output wire [9:0] hi_pop_len
);
    localparam FROM_DRAM = KIND == 0 || KIND == 1;
    reg running, host, hin;
    reg [35:0] r_src, r_dst; reg [23:0] r_n0, r_n1, r_n2;
    reg signed [31:0] r_s1s, r_s1d, r_s2s, r_s2d;
    reg [23:0] off, i1, i2; reg issuing;
    wire [23:0] left = r_n0 - off;
    wire [9:0] len = left >= PB ? PB[9:0] : left[9:0];
    // signed stride products, then modulo-2^36 address arithmetic
    wire signed [24:0] si1 = $signed({1'b0, i1}), si2 = $signed({1'b0, i2});
    wire signed [57:0] p1s = si1 * r_s1s, p2s = si2 * r_s2s, p1d = si1 * r_s1d, p2d = si2 * r_s2d;
    wire [35:0] a_src = r_src + p2s[35:0] + p1s[35:0] + {12'd0, off};
    wire [35:0] a_dst = r_dst + p2d[35:0] + p1d[35:0] + {12'd0, off};
    wire last_piece = off + len >= r_n0 && i1 + 1 >= r_n1 && i2 + 1 >= r_n2;
    function [PB-1:0] lenmask(input [9:0] l);
        integer b;
        for (b = 0; b < PB; b = b + 1) lenmask[b] = b < l;
    endfunction

    // ---- outstanding DRAM reads: destination and length, in order
    reg [35:0] o_dst [0:OUTST-1]; reg [9:0] o_len [0:OUTST-1]; reg o_last [0:OUTST-1];
    reg [$clog2(OUTST):0] o_count; reg [$clog2(OUTST)-1:0] o_head, o_tail;
    reg [31:0] o_bytes;                          // bytes requested but not yet delivered (FIFO reservation)
    // a zero-size command completes at once, so it must not overtake commands whose bytes are still arriving
    wire zero_in = n0 == 0 || n1 == 0 || n2 == 0;
    assign busy = running || (FROM_DRAM && zero_in && o_count != 0);
    wire room = KIND != 0 || fifo_free >= o_bytes + len;
    wire can_issue = running && issuing && (FROM_DRAM ? (o_count < OUTST && room) : 1'b1);
    // SRAM-source engines move a piece in the clock they read it
    wire sink_ready = KIND == 2 ? dw_ready : 1'b1;
    // H2S: the piece's bytes must be in the host FIFO (not an SRAM ready, so the write request may wait on it)
    wire hi_ok = !hin || hi_level >= len;
    wire move = !FROM_DRAM && can_issue && sink_ready && sram_ready && hi_ok;
    wire req = FROM_DRAM && can_issue;
    wire req_go = req && dr_ready;
    assign dr_valid = req; assign dr_addr = a_src; assign dr_len = len;
    assign sr_valid = (!FROM_DRAM && can_issue && !hin) || (tp_go && !tp_wr);
    assign sr_addr = tp_go ? tp_src[22:0] : a_src[22:0];
    wire [35:0] head_dst = o_dst[o_head]; wire [9:0] head_len = o_len[o_head];
    wire deliver = FROM_DRAM && dr_resp_valid;
    // sinks
    assign sw_valid = (KIND == 1 && deliver) || (KIND == 3 && !FROM_DRAM && can_issue && !host && hi_ok) || (tp_go && tp_wr);
    assign sw_addr = KIND == 1 ? head_dst[22:0] : tp_go ? tp_dst[22:0] : a_dst[22:0];
    assign sw_data = KIND == 1 ? dr_resp_data : tp_go ? {{(8*PB-8*TT){1'b0}}, tp_col} : hin ? hi_data : sr_data;
    assign sw_mask = lenmask(KIND == 1 ? head_len : tp_go ? {3'd0, th} : len);
    assign dw_valid = KIND == 2 && running && issuing && sram_ready;
    assign dw_addr = a_dst; assign dw_data = sr_data; assign dw_mask = lenmask(len);
    assign fw_valid = KIND == 0 && deliver; assign fw_data = dr_resp_data; assign fw_len = head_len;
    assign h_valid = KIND == 3 && move && host; assign h_data = sr_data; assign h_len = len;
    assign hi_pop = KIND == 3 && move && hin; assign hi_pop_len = len;

    // ---- transpose path (KIND 3)
    localparam TT = 64;
    wire tp_start = KIND == 3 && !to_host && !from_host && n0 == 24'd1 && s1_src == 32'sd1 && s2_dst == 32'sd1;
    reg tp, tp_wr; reg [23:0] t1, t2; reg [6:0] tk;
    reg [8*TT-1:0] trow [0:(KIND == 3 ? TT : 1)-1];
    wire [23:0] tw_left = r_n1 - t1, th_left = r_n2 - t2;
    wire [6:0] tw = tw_left >= TT ? TT[6:0] : tw_left[6:0];      // tile columns (i1)
    wire [6:0] th = th_left >= TT ? TT[6:0] : th_left[6:0];      // tile rows (i2)
    wire signed [24:0] st_r = $signed({1'b0, t2 + tk}), st_c = $signed({1'b0, t1 + tk});
    wire signed [57:0] tp_ps = st_r * r_s2s, tp_pd = st_c * r_s1d;
    wire [35:0] tp_src = r_src + tp_ps[35:0] + {12'd0, t1};
    wire [35:0] tp_dst = r_dst + tp_pd[35:0] + {12'd0, t2};
    reg [8*TT-1:0] tp_col; integer tq;
    always @* for (tq = 0; tq < TT; tq = tq + 1) tp_col[8*tq+:8] = trow[tq][8*tk[5:0]+:8];
    wire tp_last_k = tk + 1 >= (tp_wr ? tw : th);
    wire tp_last_tile = t1 + TT >= r_n1 && t2 + TT >= r_n2;
    wire tp_go = KIND == 3 && running && tp;

    wire advance = req_go || move;
    always @(posedge clk) begin
        done <= 0;
        if (reset) begin running <= 0; hin <= 0; o_count <= 0; o_head <= 0; o_tail <= 0; o_bytes <= 0; end
        else begin
            if (!running && start) begin
                r_src <= src; r_dst <= dst; r_n0 <= n0; r_n1 <= n1; r_n2 <= n2; host <= to_host; hin <= KIND == 3 && from_host;
                r_s1s <= s1_src; r_s1d <= s1_dst; r_s2s <= s2_src; r_s2d <= s2_dst;
                off <= 0; i1 <= 0; i2 <= 0;
                tp <= tp_start; tp_wr <= 0; t1 <= 0; t2 <= 0; tk <= 0;
                if (n0 == 0 || n1 == 0 || n2 == 0) done <= 1;
                else begin running <= 1; issuing <= !tp_start; end
            end
            if (advance) begin
                if (off + len < r_n0) off <= off + len;
                else begin
                    off <= 0;
                    if (i1 + 1 < r_n1) i1 <= i1 + 1;
                    else begin i1 <= 0; i2 <= i2 + 1; end
                end
                if (last_piece) issuing <= 0;
            end
            if (req_go) begin
                o_dst[o_tail] <= a_dst; o_len[o_tail] <= len; o_last[o_tail] <= last_piece; o_tail <= o_tail + 1'b1;
            end
            if (deliver && o_last[o_head]) done <= 1;
            if (deliver) o_head <= o_head + 1'b1;
            o_count <= o_count + req_go - deliver;
            o_bytes <= o_bytes + (req_go ? len : 0) - (deliver ? head_len : 0);
            if (tp_go && sram_ready) begin
                if (!tp_wr) trow[tk[5:0]] <= sr_data[8*TT-1:0];
                if (!tp_last_k) tk <= tk + 1'b1;
                else begin
                    tk <= 0; tp_wr <= !tp_wr;
                    if (tp_wr) begin
                        if (tp_last_tile) begin running <= 0; done <= 1; tp <= 0; end
                        else if (t1 + TT < r_n1) t1 <= t1 + TT;
                        else begin t1 <= 0; t2 <= t2 + TT; end
                    end
                end
            end
            if (running && !tp) begin
                if (FROM_DRAM) begin
                    if (req_go && last_piece) running <= 0;             // done comes with the last byte
                end else if (advance && last_piece) begin
                    running <= 0; done <= 1;
                end
            end
        end
    end
endmodule
