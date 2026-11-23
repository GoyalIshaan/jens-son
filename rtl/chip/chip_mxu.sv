// The matrix unit: TAB, MM (weights from the FIFO) and MMS (int8 weights from SRAM), golden v2's matrix
// contract (golden/bitexact.py::mm_exact) bit for bit.
//
// One beat = 1,024 weights against one activation chunk of every row:
//   lut4 (576 bytes): 512 bytes of 4-bit codes (input 2i in the low nibble) looked up in the TAB table, then
//                     64 E4M3 scales, one per group of 16 inputs. One output per beat.
//   int8 (1,024 bytes): raw weights, scale 1. seg 1: one output over 1,024 inputs. seg 4: four outputs over
//                     the same 256 inputs (the activation segment is broadcast to the four quarters).
// Per group g of 16 lanes and row r: S = sum(a * w) exact; P = S * M_g (signed); e = E_g + k_r (the row's
// exponent for that 32-block). The beat reducer (rtl/mxu/beat_reducer.v, reducer_p2_seg) aligns the 64 groups
// to the largest exponent (or each segment's own) with 16 fraction bits rounded down and rounds once to fp32.
// Each output accumulates in input order: acc = fp32(acc + v). At the end, y = fp32(acc * tscale), then the
// epilogue: f32 writes y, acc writes fp32(dst + y), argmax keeps a running (best, index) per row, ties to the
// lowest index, and writes index + id_base.
//
// Order of work, which is also the order of the weight stream: tiles of outputs, each tile's chunks of inputs,
// each chunk's beats.
//   lut4: tiles of 64 outputs, 1,024-input chunks, one output per beat
//   int8, one segment (n_in = 1,024, or 256 with seg 4): tiles of 16 outputs, one chunk
//   int8, several segments: one tile of all n_out outputs, one chunk per segment
// Three engines run at once (D-029), so the beats are the only clocks a command costs in steady state:
//   loader   fills the stationary activation buffer for the next (tile, chunk) while the beats use the other one
//            (two buffers of MAXR rows). One row per clock (1,024 bytes on port A, its exponents on port E);
//            four rows per clock for seg 4 when rows are 256 bytes apart and their exponents fit one 64-byte read.
//            A one-chunk command loads once for all its tiles.
//   beats    RP rows per clock (the chip: RP = 64); more rows hold each beat ceil(rows / RP) clocks.
//   drain    the epilogue of a finished tile while later tiles' beats run: ER rows x 16 outputs per clock
//            (the chip: ER = 16, 256 outputs), one 64-byte line per row, on ER write ports (and, for the acc
//            epilogue, ER read ports first).
// Accumulator slots: the 2 x ACCW words hold as many tiles as fit, so beats run several tiles ahead of the
// drain and the 17-clock accumulate pipeline and the drain hide behind them. A small command (a tile of at
// most ACCW / 2 words: 16 outputs x 128 rows is 2,048) keeps its slots in one half, the half of its command
// slot; a big one (lut4: 64 x 128 = 8,192) uses both halves, two slots.
// Commands overlap (the front and the back): the loader and the beats (the front) take the next command as
// soon as the current one's last beat has issued, while the drain, the end wait and argmax's index writes
// (the back) finish the older one. Each command's back-side settings live in its command slot (two), so a
// new command never changes an older one's drain. Every beat carries its own accumulator address down the
// pipeline. Two small commands overlap fully (they use different halves); a big one's beats wait for the
// older command's back. Commands finish (done, or fault) in the order they started, TAB and NOP included.
module chip_mxu #(parameter RP = 4, MAXR = 128, ACCW = 8192, ER = 16)(
    input wire clk, reset,
    input wire ce,                                    // SRAM ready for every M port this clock; else the unit holds
    input wire start, input wire [1:0] op,            // 0 NOP, 1 TAB, 2 MM, 3 MMS
    input wire [19:0] n_in, n_out, input wire [9:0] rows,
    input wire [22:0] act, exps, dst, wsrc, input wire [19:0] act_stride, exps_stride, dst_stride,
    input wire wfmt_int8, seg4, input wire [1:0] epi, input wire [31:0] id_base, tscale,
    output wire busy, output reg done, output reg fault,
    // weight FIFO: the first 1,024 bytes are visible; pop removes bytes from the front
    input wire [31:0] fifo_level, input wire [8191:0] fifo_head, output wire fifo_pop, output wire [10:0] fifo_pop_len,
    // SRAM ports: W (MMS weights, 1,024 bytes), A (activations, 1,024), E (exponents, 64),
    // D (ER lines of 64 bytes, the acc epilogue's dst), O (ER masked 64-byte line writes)
    output wire w_rd_valid, output wire [22:0] w_rd_addr, input wire [8191:0] w_rd_data,
    output wire a_rd_valid, output wire [22:0] a_rd_addr, input wire [8191:0] a_rd_data,
    output wire e_rd_valid, output wire [22:0] e_rd_addr, input wire [511:0] e_rd_data,
    output wire [ER-1:0] d_rd_valid, output wire [23*ER-1:0] d_rd_addr, input wire [512*ER-1:0] d_rd_data,
    output wire [ER-1:0] o_wr_valid, output wire [23*ER-1:0] o_wr_addr, output wire [512*ER-1:0] o_wr_data,
    output wire [64*ER-1:0] o_wr_mask
);
    localparam F_IDLE = 0, F_TAB = 1, F_RUN = 2;
    localparam B_DRAIN = 0, B_DWAIT = 1, B_ARGW = 2;
    localparam QW = ACCW / 4;                         // a quarter of the accumulator (the smallest slot)
    genvar i, g, r;
    integer t, j, jj;

    // ---- command slots (two, in order: tp fills, bp finishes)
    reg e_valid [0:1], e_fdone [0:1], e_mm [0:1], e_fault [0:1], e_small [0:1];
    reg [1:0] e_nslg [0:1];                           // log2 of the slots per tile ring (1 or 2)
    reg [31:0] e_ss [0:1];                            // words per slot
    reg [9:0] e_rows [0:1]; reg [19:0] e_tout [0:1], e_ntiles [0:1], e_nout [0:1];
    reg [22:0] e_dst [0:1]; reg [19:0] e_ds [0:1]; reg [1:0] e_epi [0:1]; reg [31:0] e_id [0:1], e_tsu [0:1];
    reg tp, bp;
    reg [1:0] f_st; reg fp;                           // the front's state and its command slot
    assign busy = f_st != F_IDLE || e_valid[tp];

    // ---- the front's command, latched
    reg r_mms, r_int8, r_seg4;
    reg [19:0] r_nin, r_nout; reg [9:0] r_rows;
    reg [22:0] r_act, r_exps, r_wsrc; reg [19:0] r_as, r_es; reg [31:0] r_ts;
    // ---- stationary state
    reg [127:0] table_r; reg [31:0] hdr_ts;
    reg [8191:0] abuf [0:2*MAXR-1]; reg [255:0] ebuf [0:2*MAXR-1];   // [buffer * MAXR + row]
    reg [31:0] acc [0:2*ACCW-1];
    reg [31:0] best [0:MAXR-1]; reg [19:0] besti [0:MAXR-1];

    // ---- a command's shape, from its fields (used on the unlatched command at start, and on the front's)
    function [19:0] f_tout(input int8, input sg4, input [19:0] nin, nout);
        reg [19:0] nch;
        begin nch = (int8 && sg4) ? nin >> 8 : nin >> 10; f_tout = !int8 ? 20'd64 : nch == 1 ? 20'd16 : nout; end
    endfunction
    function [19:0] f_ntiles(input int8, input sg4, input [19:0] nin, nout);
        reg [19:0] nch;
        begin nch = (int8 && sg4) ? nin >> 8 : nin >> 10; f_ntiles = !int8 ? nout >> 6 : nch == 1 ? (nout + 15) >> 4 : 20'd1; end
    endfunction
    wire [10:0] sw = r_seg4 ? 11'd256 : 11'd1024;
    wire [2:0] segw = r_seg4 ? 3'd4 : 3'd1;
    wire [19:0] nchunks = (r_int8 && r_seg4) ? r_nin >> 8 : r_nin >> 10;
    wire shared = nchunks == 1;                       // one chunk: the activations load once for every tile
    wire [19:0] tile_outs = f_tout(r_int8, r_seg4, r_nin, r_nout);
    wire [19:0] ntiles = f_ntiles(r_int8, r_seg4, r_nin, r_nout);
    wire [19:0] nseq = shared ? 20'd1 : ntiles * nchunks;
    wire [19:0] bpt_full = r_seg4 ? tile_outs >> 2 : tile_outs;   // beats per chunk of a full tile
    wire [10:0] beat_bytes = r_int8 ? 11'd1024 : 11'd576;
    wire [9:0] rgroups = (r_rows + RP - 1) / RP;
    wire pk4 = r_seg4 && r_as == 20'd256 && r_es * 3 + 8 <= 64;
    wire [2:0] lrpc = pk4 ? 3'd4 : 3'd1;           // rows loaded per clock
    function [19:0] tout(input [19:0] t);              // outputs in the front's tile t
        tout = r_nout - t * tile_outs < tile_outs ? r_nout - t * tile_outs : tile_outs;
    endfunction
    // a tile's slot: its word address and its in-flight counter (the quarter it starts in)
    function [31:0] slot_base(input sml, input par, input [1:0] nslg, input [31:0] ss, input [19:0] t);
        reg [19:0] s;
        begin s = t & ((20'd1 << nslg) - 1); slot_base = (sml ? par * ACCW : 0) + s * ss; end
    endfunction

    // ---- engine state
    reg [19:0] ld_seq, ld_done; reg [9:0] ld_row;   // loader: (tile, chunk) sequence number being loaded
    reg [19:0] c_tile, c_chunk, c_k; reg [9:0] c_rg;  // beats
    reg [19:0] dr_tile, dr_og; reg [9:0] dr_rb;       // drain (the back's command): tile, 16 outputs, ER rows
    reg [1:0] b_ph; reg [4:0] dwait; reg [9:0] ar_rb;
    reg [3:0] ts_cnt;                                 // clocks since the front's command started (tscale)

    // ---- the back's command
    wire b_mm = e_valid[bp] && e_mm[bp];
    wire [9:0] b_rows = e_rows[bp]; wire [19:0] b_tout = e_tout[bp], b_ntiles = e_ntiles[bp], b_nout = e_nout[bp];
    wire [22:0] b_dst = e_dst[bp]; wire [19:0] b_ds = e_ds[bp]; wire [1:0] b_epi = e_epi[bp];
    wire same = f_st == F_RUN && fp == bp;            // the front and the back are on one command
    // tiles of the back's command whose beats have all issued
    wire [19:0] beaten = same ? c_tile : b_ntiles;

    // ---- loader
    wire [19:0] c_seq = shared ? 20'd0 : c_tile * nchunks + c_chunk;
    wire ld_go = f_st == F_RUN && ld_seq < nseq && ld_seq <= c_seq + 1;
    wire ld_set = ld_seq[0];
    wire [19:0] ld_chunk = shared ? 20'd0 : ld_seq % nchunks;
    assign a_rd_valid = ld_go;
    assign a_rd_addr = r_act + ld_row * r_as + ld_chunk * sw;
    assign e_rd_valid = ld_go;
    assign e_rd_addr = r_exps + ld_row * r_es + ld_chunk * (sw >> 5);

    // ---- the beat's weights: from the FIFO head, or from SRAM (MMS)
    wire c_active = f_st == F_RUN && c_tile < ntiles;
    wire [19:0] c_bpt = r_seg4 ? tout(c_tile) >> 2 : tout(c_tile);
    wire [19:0] wbeat = (c_tile * nchunks + c_chunk) * bpt_full + c_k;
    assign w_rd_valid = c_active && r_mms;
    assign w_rd_addr = r_wsrc + wbeat * 1024;
    wire [8191:0] wb = r_mms ? w_rd_data : fifo_head;
    wire w_ready = r_mms ? 1'b1 : fifo_level >= beat_bytes;
    // the front's slots
    wire f_small = e_small[fp]; wire [1:0] f_nslg = e_nslg[fp]; wire [31:0] f_ss = e_ss[fp];
    wire [3:0] f_nslot = 4'd1 << f_nslg;
    wire [31:0] c_base = slot_base(f_small, fp, f_nslg, f_ss, c_tile);
    wire [2:0] c_sid = c_base / QW;
    // beats in the pipeline per quarter; the drain waits for its tile's to land, and a chunk shorter than the
    // pipeline waits for the previous chunk's writes (they revisit the same entries)
    reg [5:0] inflight [0:7];
    wire bpc_short = c_bpt * rgroups < 6'd17;          // RL + AL clocks from issue to the acc write
    wire chunk_start = c_k == 0 && c_rg == 0 && c_chunk != 0;
    wire hazard = bpc_short && chunk_start && inflight[c_sid] != 0;
    // the drain's progress on the front's command (none yet while the back finishes an older one), and
    // whether the front may beat beside an older command's back (both small: different halves)
    wire [19:0] f_drained = same ? dr_tile : 20'd0;
    wire f_clear = same || !b_mm || (f_small && e_small[bp]);
    wire beat_go = c_active && w_ready && ld_done > c_seq && c_tile < f_drained + f_nslot && !hazard && f_clear;
    wire beat_last_rg = c_rg + 1 >= rgroups;
    wire c_set = c_seq[0];
    wire [8191:0] w8; wire [64*4-1:0] gM; wire [63:0] gS; wire [64*10-1:0] gE;
    for (i = 0; i < 1024; i = i + 1) begin : wdec
        wire [3:0] code = wb[(i/2)*8 + (i%2)*4 +: 4];
        assign w8[8*i+:8] = r_int8 ? wb[8*i+:8] : table_r[8*code+:8];
    end
    for (g = 0; g < 64; g = g + 1) begin : sdec
        wire [7:0] ws = wb[(512+g)*8 +: 8];
        wire [3:0] we = ws[6:3];
        assign gM[4*g+:4] = r_int8 ? 4'd1 : (we != 0 ? {1'b1, ws[2:0]} : {1'b0, ws[2:0]});
        assign gS[g] = r_int8 ? 1'b0 : ws[7];
        assign gE[10*g+:10] = r_int8 ? 10'sd0 : (we != 0 ? $signed({6'd0, we}) - 10'sd10 : -10'sd9);
    end
    // ---- the beat pipeline (1 GHz): stage 0 registers the decoded weights; the RP rows run in mxu_row_pipe
    // (8 stages); the accumulate reads acc at stage RL = 9, adds on an 8-stage FMA and writes at RL + 8.
    // A chunk's first beat adds to +0 (exact: a row sum v is never -0 and never below 2^-126).
    // Control rides alongside, all of it the beat's own (a later command may have started by the time it
    // lands): valid, quarter, word address of (output 0, row group), rows, row group, seg 4, first chunk.
    // ce low holds every stage.
    localparam RL = 9, AL = 8;
    reg [8191:0] w8_q; reg [64*4-1:0] gM_q; reg [63:0] gS_q; reg [64*10-1:0] gE_q; reg set_q; reg [9:0] rg_q;
    reg seg4_q;
    always @(posedge clk) if (ce) begin
        w8_q <= w8; gM_q <= gM; gS_q <= gS; gE_q <= gE; set_q <= c_set; rg_q <= c_rg; seg4_q <= r_seg4;
    end
    wire [31:0] c_aoff = c_base + c_k * segw * r_rows + c_rg * RP;
    localparam CW = 1 + 3 + 16 + 10 + 10 + 1 + 1;
    wire [CW-1:0] ctl_in = {beat_go, c_sid, c_aoff[15:0], r_rows, c_rg, r_seg4, c_chunk == 0};
    wire [CW-1:0] ctl_rd, ctl_wr;                       // at the acc read (RL) and the acc write (RL + AL)
    vec_delay_ce #(CW, RL) ctl_a(clk, ce, ctl_in, ctl_rd);
    vec_delay_ce #(CW, AL) ctl_b(clk, ce, ctl_rd, ctl_wr);
    wire rd_v = ctl_rd[CW-1], rd_first = ctl_rd[0];
    wire [15:0] rd_aoff = ctl_rd[CW-5 -: 16]; wire [9:0] rd_rows = ctl_rd[CW-21 -: 10];
    wire wr_v = ctl_wr[CW-1]; wire [2:0] wr_sid = ctl_wr[CW-2 -: 3];
    wire [15:0] wr_aoff = ctl_wr[CW-5 -: 16]; wire [9:0] wr_rows = ctl_wr[CW-21 -: 10], wr_rg = ctl_wr[CW-31 -: 10];
    wire wr_seg4 = ctl_wr[1];
    wire [31:0] v [0:RP-1][0:3];
    for (r = 0; r < RP; r = r + 1) begin : rows_dp
        wire [9:0] row = rg_q * RP + r;
        wire [8191:0] a8 = abuf[set_q * MAXR + (row < MAXR ? row : 0)];
        wire [255:0] e8 = ebuf[set_q * MAXR + (row < MAXR ? row : 0)];
        mxu_row_pipe dp(.clk(clk), .ce(ce), .w(w8_q), .gM(gM_q), .gS(gS_q), .gE(gE_q), .act(a8), .aexp(e8), .seg4(seg4_q),
                        .v0(v[r][0]), .v1(v[r][1]), .v2(v[r][2]), .v3(v[r][3]));
    end
    // accumulate: word = beat address + output-in-beat * rows + row
    wire [31:0] acc_new [0:RP-1][0:3];
    for (r = 0; r < RP; r = r + 1) begin : accum
        for (g = 0; g < 4; g = g + 1) begin : q
            wire [31:0] aw = rd_aoff + g * rd_rows + r;
            wire [31:0] old = rd_first ? 32'd0 : acc[aw < 2 * ACCW ? aw : 0];
            fp32_fma_pipe8_ce add(clk, ce, old, 32'h3f800000, v[r][g], acc_new[r][g]);
        end
    end

    // ---- drain (the back's command): ER rows x 16 outputs of a finished tile per clock, pipelined (1 GHz)
    //   stage 0   read acc (once the tile's last beats have landed); y = fp32(acc * tscale) on an 8-stage FMA
    //   stage 8   f32: write y.  acc: read dst, y2 = fp32(dst + y) on an 8-stage FMA; write at stage 16.
    //             argmax: a 16-way max tree (ties to the lower index) in stages 8-9, compared with the row's
    //             running best at stage 10 (one compare per clock, the only loop)
    // Reads (stage 8) and writes (stage 16) of the acc epilogue are different lines in any clock.
    wire [31:0] dr_base = slot_base(e_small[bp], bp, e_nslg[bp], e_ss[bp], dr_tile);
    wire [2:0] dr_sid = dr_base / QW;
    wire dr_active = b_mm && b_ph == B_DRAIN && dr_tile < beaten && inflight[dr_sid] == 0;
    wire [19:0] dr_tout = b_nout - dr_tile * b_tout < b_tout ? b_nout - dr_tile * b_tout : b_tout;
    wire [19:0] dr_out0 = dr_tile * b_tout + dr_og * 16;       // first output of this clock's lines
    wire [31:0] ts;
    fp32_fma_pipe8_ce tsmul(clk, 1'b1, hdr_ts, r_ts, 32'd0, ts);  // the front's command: settled 8 clocks after start
    wire [31:0] tscale_use = e_tsu[bp];
    function fgt(input [31:0] a, b);                   // a > b; no NaN, zeros are +0
        if (a[31] != b[31]) fgt = !a[31];
        else fgt = a[31] ? a[30:0] < b[30:0] : a[30:0] > b[30:0];
    endfunction
    wire b_out = b_mm && (b_ph == B_DRAIN || b_ph == B_DWAIT);   // drain stages 8-16 may still be in flight
    // the drain clock's control, at stages 8, 10 and 16: (valid, first block, rows on, outputs on, row block, out0)
    wire [15:0] y_on0; wire [ER-1:0] er_on0;
    for (g = 0; g < 16; g = g + 1) begin : yon assign y_on0[g] = dr_og * 16 + g < dr_tout; end
    localparam DW = 1 + 1 + ER + 16 + 10 + 20;
    wire [DW-1:0] d0 = {dr_active, dr_tile == 0 && dr_og == 0, er_on0, y_on0, dr_rb, dr_out0};
    wire [DW-1:0] d8, d10, d16;
    vec_delay_ce #(DW, 8) dd8(clk, ce, d0, d8);
    vec_delay_ce #(DW, 2) dd10(clk, ce, d8, d10);
    vec_delay_ce #(DW, 6) dd16(clk, ce, d10, d16);
    `define DV(d) d[DW-1]
    `define DFIRST(d) d[DW-2]
    `define DER(d) d[DW-3 -: ER]
    `define DYON(d) d[DW-3-ER -: 16]
    `define DRB(d) d[29:20]
    `define DOUT0(d) d[19:0]
    wire [15:0] yon8 = `DYON(d8), yon16 = `DYON(d16);
    wire [ER-1:0] ron8 = `DER(d8), ron10 = `DER(d10), ron16 = `DER(d16);
    for (r = 0; r < ER; r = r + 1) begin : ep
        wire [9:0] er0 = dr_rb * ER + r;
        assign er_on0[r] = er0 < b_rows;
        wire [31:0] y [0:15], y2 [0:15];
        wire [9:0] er8 = `DRB(d8) * ER + r, er16 = `DRB(d16) * ER + r, er10 = `DRB(d10) * ER + r;
        wire [22:0] addr8 = b_dst + er8 * b_ds + `DOUT0(d8) * 4, addr16 = b_dst + er16 * b_ds + `DOUT0(d16) * 4;
        for (g = 0; g < 16; g = g + 1) begin : o
            wire [31:0] aw = dr_base + (dr_og * 16 + g) * b_rows + er0;
            wire [31:0] a_ = acc[aw < 2 * ACCW ? aw : 0];
            fp32_fma_pipe8_ce m(clk, ce, a_, tscale_use, 32'd0, y[g]);
            fp32_fma_pipe8_ce s(clk, ce, d_rd_data[512*r + 32*g +: 32], 32'h3f800000, y[g], y2[g]);
        end
        // argmax: (value, index, valid) max tree over the 16 outputs, lower index on ties
        reg [36:0] t1 [0:3]; reg [36:0] t2;            // {valid, index 4 bits, value}
        function [36:0] pick(input [36:0] lo, hi);
            pick = hi[36] && (!lo[36] || fgt(hi[31:0], lo[31:0])) ? hi : lo;
        endfunction
        integer q;
        always @(posedge clk) if (ce) begin
            for (q = 0; q < 4; q = q + 1)
                t1[q] <= pick(pick({yon8[4*q], 4'(4*q), y[4*q]}, {yon8[4*q+1], 4'(4*q+1), y[4*q+1]}),
                              pick({yon8[4*q+2], 4'(4*q+2), y[4*q+2]}, {yon8[4*q+3], 4'(4*q+3), y[4*q+3]}));
            t2 <= pick(pick(t1[0], t1[1]), pick(t1[2], t1[3]));
            if (b_out && b_epi == 2 && `DV(d10) && ron10[r] && t2[36] &&
                (`DFIRST(d10) || fgt(t2[31:0], best[er10 < MAXR ? er10 : 0]))) begin
                best[er10 < MAXR ? er10 : 0] <= t2[31:0]; besti[er10 < MAXR ? er10 : 0] <= `DOUT0(d10) + t2[35:32];
            end
        end
        reg [511:0] wdata; reg [63:0] wmask; wire [9:0] era = ar_rb * ER + r;
        wire argw = b_mm && b_ph == B_ARGW;
        always @* begin
            wdata = 0; wmask = 0;
            if (argw) begin wdata[31:0] = besti[era < MAXR ? era : 0] + e_id[bp]; wmask = 64'hf; end
            else for (q = 0; q < 16; q = q + 1) if (b_epi == 1 ? yon16[q] : yon8[q]) begin
                wdata[32*q+:32] = b_epi == 1 ? y2[q] : y[q]; wmask[4*q+:4] = 4'hf;
            end
        end
        assign d_rd_valid[r] = b_out && b_epi == 1 && `DV(d8) && ron8[r];
        assign d_rd_addr[23*r+:23] = addr8;
        wire w_on = b_epi == 0 ? `DV(d8) && ron8[r] : b_epi == 1 ? `DV(d16) && ron16[r] : 1'b0;
        assign o_wr_valid[r] = argw ? era < b_rows : b_out && w_on;
        assign o_wr_addr[23*r+:23] = argw ? b_dst + era * b_ds : b_epi == 1 ? addr16 : addr8;
        assign o_wr_data[512*r+:512] = wdata;
        assign o_wr_mask[64*r+:64] = wmask;
    end

    assign fifo_pop = ce && ((f_st == F_TAB && fifo_level >= 32) || (beat_go && beat_last_rg && !r_mms));
    assign fifo_pop_len = f_st == F_TAB ? 11'd32 : beat_bytes;

    // ---- the starting command: its checks and its shape
    wire i_int8 = wfmt_int8 || op == 3;
    wire [19:0] i_nch = (i_int8 && seg4) ? n_in >> 8 : n_in >> 10;
    wire [19:0] i_tout = f_tout(i_int8, seg4, n_in, n_out);
    wire bad = rows == 0 || rows > MAXR || n_in == 0 || n_out == 0 || i_tout * rows > ACCW ||
               (i_int8 ? (seg4 && n_out[1:0] != 0) || (seg4 ? n_in[7:0] : n_in[9:0]) != 0
                       : n_out[5:0] != 0 || n_in[9:0] != 0 || seg4);
    wire [31:0] i_words = i_tout * rows;
    wire i_small = i_words <= ACCW / 2;
    wire [1:0] i_nslg = i_small ? (i_words <= QW ? 2'd2 : 2'd1) : 2'd1;
    wire [31:0] i_ss = i_small ? (i_words <= QW ? QW : ACCW / 2) : ACCW;
    wire accept = start && !busy;

    always @(posedge clk) begin
        done <= 0; fault <= 0;
        if (reset) begin
            f_st <= F_IDLE; tp <= 0; bp <= 0; fp <= 0; b_ph <= B_DRAIN;
            e_valid[0] <= 0; e_valid[1] <= 0;
            for (t = 0; t < 8; t = t + 1) inflight[t] <= 0;
            dr_tile <= 0; dr_og <= 0; dr_rb <= 0; ar_rb <= 0;
        end else begin
            // ---- accept a command into slot tp (whatever ce: the core's start is one clock)
            if (accept) begin
                e_valid[tp] <= 1; e_mm[tp] <= op >= 2 && !bad; e_fault[tp] <= op >= 2 && bad;
                e_fdone[tp] <= op == 0 || (op >= 2 && bad);
                e_rows[tp] <= rows; e_tout[tp] <= i_tout; e_ntiles[tp] <= f_ntiles(i_int8, seg4, n_in, n_out);
                e_nout[tp] <= n_out; e_dst[tp] <= dst; e_ds[tp] <= dst_stride; e_epi[tp] <= epi; e_id[tp] <= id_base;
                e_small[tp] <= i_small; e_nslg[tp] <= i_nslg; e_ss[tp] <= i_ss;
                tp <= !tp; fp <= tp;
                if (op == 1) f_st <= F_TAB;
                else if (op >= 2 && !bad) begin
                    f_st <= F_RUN;
                    r_mms <= op == 3; r_int8 <= i_int8; r_seg4 <= seg4;
                    r_nin <= n_in; r_nout <= n_out; r_rows <= rows; r_act <= act; r_exps <= exps;
                    r_wsrc <= wsrc; r_as <= act_stride; r_es <= exps_stride; r_ts <= tscale;
                    ld_seq <= 0; ld_done <= 0; ld_row <= 0; c_tile <= 0; c_chunk <= 0; c_k <= 0; c_rg <= 0;
                    ts_cnt <= 0;
                end
            end
            // ---- the scale settles 8 clocks after the front's command starts (tsmul runs every clock)
            if (f_st == F_RUN && ts_cnt != 4'd9) begin
                ts_cnt <= ts_cnt + 1'b1;
                if (ts_cnt == 4'd8) e_tsu[fp] <= r_int8 ? r_ts : ts;
            end
            if (ce) begin
                // ---- front
                if (f_st == F_TAB && fifo_level >= 32) begin
                    table_r <= fifo_head[127:0]; hdr_ts <= fifo_head[159:128]; e_fdone[fp] <= 1; f_st <= F_IDLE;
                end
                if (f_st == F_RUN) begin
                    if (ld_go) begin
                        for (jj = 0; jj < 4; jj = jj + 1)
                            if (jj < lrpc && ld_row + jj < r_rows) begin
                                for (t = 0; t < 4; t = t + 1) begin
                                    abuf[ld_set * MAXR + ld_row + jj][2048*t+:2048] <= r_seg4 ? a_rd_data[2048*jj+:2048]
                                                                                      : a_rd_data[2048*t+:2048];
                                    ebuf[ld_set * MAXR + ld_row + jj][64*t+:64] <= r_seg4 ? e_rd_data[8*(jj*r_es)+:64]
                                                                                  : e_rd_data[64*t+:64];
                                end
                            end
                        if (ld_row + lrpc >= r_rows) begin ld_row <= 0; ld_seq <= ld_seq + 1; ld_done <= ld_seq + 1; end
                        else ld_row <= ld_row + lrpc;
                    end
                    if (beat_go) begin
                        if (!beat_last_rg) c_rg <= c_rg + 1;
                        else begin
                            c_rg <= 0;
                            if (c_k + 1 < c_bpt) c_k <= c_k + 1;
                            else begin
                                c_k <= 0;
                                if (c_chunk + 1 < nchunks) c_chunk <= c_chunk + 1;
                                else begin c_chunk <= 0; c_tile <= c_tile + 1; end
                            end
                        end
                    end
                    // every beat issued and the scale captured: the front takes the next command
                    if (c_tile == ntiles && ts_cnt == 4'd9) begin e_fdone[fp] <= 1; f_st <= F_IDLE; end
                end
                // ---- accumulate writes, and the beats in flight per quarter
                if (wr_v)
                    for (t = 0; t < RP; t = t + 1)
                        if (wr_rg * RP + t < wr_rows)
                            for (j = 0; j < 4; j = j + 1)
                                if (j == 0 || wr_seg4)
                                    acc[wr_aoff + j * wr_rows + t] <= acc_new[t][j];
                for (t = 0; t < 8; t = t + 1)
                    inflight[t] <= inflight[t] + (beat_go && c_sid == t) - (wr_v && wr_sid == t);
                // ---- back
                if (e_valid[bp]) begin
                    if (!e_mm[bp]) begin
                        // TAB, NOP or a fault: finished once its front is, in order
                        if (e_fdone[bp]) begin
                            if (e_fault[bp]) fault <= 1; else done <= 1;
                            e_valid[bp] <= 0; bp <= !bp;
                        end
                    end else case (b_ph)
                        B_DRAIN: begin
                            if (dr_active) begin
                                if ((dr_rb + 1) * ER < b_rows) dr_rb <= dr_rb + 1;
                                else begin
                                    dr_rb <= 0;
                                    if ((dr_og + 1) * 16 < dr_tout) dr_og <= dr_og + 1;
                                    else begin dr_og <= 0; dr_tile <= dr_tile + 1; end
                                end
                            end
                            if (dr_tile == b_ntiles) begin b_ph <= B_DWAIT; dwait <= 5'd17; end
                        end
                        B_DWAIT: if (dwait != 0) dwait <= dwait - 1'b1;        // the drain pipeline empties
                                 else if (b_epi == 2) begin b_ph <= B_ARGW; ar_rb <= 0; end
                                 else begin done <= 1; e_valid[bp] <= 0; bp <= !bp; b_ph <= B_DRAIN; dr_tile <= 0; end
                        B_ARGW: if ((ar_rb + 1) * ER < b_rows) ar_rb <= ar_rb + 1;
                                else begin done <= 1; e_valid[bp] <= 0; bp <= !bp; b_ph <= B_DRAIN; dr_tile <= 0; end
                        default: b_ph <= B_DRAIN;
                    endcase
                end
            end
        end
    end
endmodule

// One row of the datapath: 64 groups of 16 lanes, then the beat reducer.
module mxu_row (
    input wire [8191:0] w, input wire [64*4-1:0] gM, input wire [63:0] gS, input wire [64*10-1:0] gE,
    input wire [8191:0] act, input wire [255:0] aexp, input wire seg4,
    output wire [31:0] v0, v1, v2, v3
);
    wire [64*24-1:0] P; wire [64*10-1:0] e;
    genvar g, i;
    for (g = 0; g < 64; g = g + 1) begin : grp
        wire [16*16-1:0] prod;
        for (i = 0; i < 16; i = i + 1) begin : lane
            wire signed [7:0] a = act[8*(16*g+i)+:8], b = w[8*(16*g+i)+:8];
            wire signed [15:0] pr = a * b;
            assign prod[16*i+:16] = pr;
        end
        wire signed [18:0] S;
        adder_tree16 tree(.p(prod), .s(S));
        wire signed [23:0] SM = S * $signed({1'b0, gM[4*g+:4]});
        assign P[24*g+:24] = gS[g] ? -SM : SM;
        wire signed [9:0] k = {{2{aexp[8*(g/2)+7]}}, aexp[8*(g/2)+:8]};
        assign e[10*g+:10] = $signed(gE[10*g+:10]) + k;
    end
    reducer_p2_seg #(.F(16)) red(.P(P), .e(e), .seg4(seg4), .v0(v0), .v1(v1), .v2(v2), .v3(v3));
endmodule
