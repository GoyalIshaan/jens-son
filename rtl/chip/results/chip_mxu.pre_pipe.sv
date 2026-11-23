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
//   beats    RP rows per clock (the chip: RP = 64); more rows hold each beat ceil(rows / RP) clocks. Each tile
//            accumulates into one of two accumulator halves (ACCW words each).
//   drain    the epilogue of a finished tile, from its half, while the next tile's beats fill the other:
//            ER rows x 16 outputs per clock (the chip: ER = 16, 256 outputs), one 64-byte line per row, written
//            on ER write ports (and, for the acc epilogue, read first on ER read ports).
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
    localparam IDLE = 0, TAB = 1, RUN = 2, ARGW = 3, FIN = 4;
    reg [2:0] st;
    assign busy = st != IDLE;
    // ---- the command, latched
    reg r_mms, r_int8, r_seg4; reg [1:0] r_epi;
    reg [19:0] r_nin, r_nout; reg [9:0] r_rows;
    reg [22:0] r_act, r_exps, r_dst, r_wsrc; reg [19:0] r_as, r_es, r_ds; reg [31:0] r_id, r_ts;
    // ---- stationary state
    reg [127:0] table_r; reg [31:0] hdr_ts;
    reg [8191:0] abuf [0:2*MAXR-1]; reg [255:0] ebuf [0:2*MAXR-1];   // [buffer * MAXR + row]
    reg [31:0] acc [0:2*ACCW-1];                                       // [half * ACCW + index]
    reg [31:0] best [0:MAXR-1]; reg [19:0] besti [0:MAXR-1];

    // ---- the command's shape
    wire [10:0] sw = r_seg4 ? 11'd256 : 11'd1024;
    wire [2:0] segw = r_seg4 ? 3'd4 : 3'd1;
    wire [19:0] nchunks = (r_int8 && r_seg4) ? r_nin >> 8 : r_nin >> 10;
    wire shared = nchunks == 1;                       // one chunk: the activations load once for every tile
    wire [19:0] tile_outs = !r_int8 ? 20'd64 : shared ? 20'd16 : r_nout;
    wire [19:0] ntiles = !r_int8 ? r_nout >> 6 : shared ? (r_nout + 15) >> 4 : 20'd1;
    wire [19:0] nseq = shared ? 20'd1 : ntiles * nchunks;
    wire [19:0] bpt_full = r_seg4 ? tile_outs >> 2 : tile_outs;   // beats per chunk of a full tile
    wire [10:0] beat_bytes = r_int8 ? 11'd1024 : 11'd576;
    wire [9:0] rgroups = (r_rows + RP - 1) / RP;
    wire pk4 = r_seg4 && r_as == 20'd256 && r_es * 3 + 8 <= 64;
    wire [2:0] lrpc = pk4 ? 3'd4 : 3'd1;           // rows loaded per clock
    function [19:0] tout(input [19:0] t);              // outputs in tile t
        tout = r_nout - t * tile_outs < tile_outs ? r_nout - t * tile_outs : tile_outs;
    endfunction

    // ---- engine state
    reg [19:0] ld_seq, ld_done; reg [9:0] ld_row;   // loader: (tile, chunk) sequence number being loaded
    reg [19:0] c_tile, c_chunk, c_k; reg [9:0] c_rg;  // beats
    reg [19:0] dr_tile, dr_og; reg [9:0] dr_rb;       // drain: tile, group of 16 outputs, block of ER rows

    // ---- loader
    wire [19:0] c_seq = shared ? 20'd0 : c_tile * nchunks + c_chunk;
    wire ld_go = st == RUN && ld_seq < nseq && ld_seq <= c_seq + 1;
    wire ld_set = ld_seq[0];
    wire [19:0] ld_chunk = shared ? 20'd0 : ld_seq % nchunks;
    assign a_rd_valid = ld_go;
    assign a_rd_addr = r_act + ld_row * r_as + ld_chunk * sw;
    assign e_rd_valid = ld_go;
    assign e_rd_addr = r_exps + ld_row * r_es + ld_chunk * (sw >> 5);

    // ---- the beat's weights: from the FIFO head, or from SRAM (MMS)
    wire c_active = st == RUN && c_tile < ntiles;
    wire [19:0] c_bpt = r_seg4 ? tout(c_tile) >> 2 : tout(c_tile);
    wire [19:0] wbeat = (c_tile * nchunks + c_chunk) * bpt_full + c_k;
    assign w_rd_valid = c_active && r_mms;
    assign w_rd_addr = r_wsrc + wbeat * 1024;
    wire [8191:0] wb = r_mms ? w_rd_data : fifo_head;
    wire w_ready = r_mms ? 1'b1 : fifo_level >= beat_bytes;
    wire beat_go = c_active && w_ready && ld_done > c_seq && c_tile <= dr_tile + 1;
    wire beat_last_rg = c_rg + 1 >= rgroups;
    wire c_set = c_seq[0], c_half = c_tile[0];
    wire [8191:0] w8; wire [64*4-1:0] gM; wire [63:0] gS; wire [64*10-1:0] gE;
    genvar i, g, r;
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
    // ---- RP rows of the datapath
    wire [31:0] v [0:RP-1][0:3];
    for (r = 0; r < RP; r = r + 1) begin : rows_dp
        wire [9:0] row = c_rg * RP + r;
        wire [8191:0] a8 = abuf[c_set * MAXR + (row < MAXR ? row : 0)];
        wire [255:0] e8 = ebuf[c_set * MAXR + (row < MAXR ? row : 0)];
        mxu_row dp(.w(w8), .gM(gM), .gS(gS), .gE(gE), .act(a8), .aexp(e8), .seg4(r_seg4),
                   .v0(v[r][0]), .v1(v[r][1]), .v2(v[r][2]), .v3(v[r][3]));
    end
    // accumulate: index in the half = (output in tile) * rows + row
    wire [31:0] acc_new [0:RP-1][0:3];
    for (r = 0; r < RP; r = r + 1) begin : accum
        for (g = 0; g < 4; g = g + 1) begin : q
            wire [31:0] old = acc[c_half * ACCW + ((c_k * segw + g) * r_rows + c_rg * RP + r) % ACCW];
            wire [31:0] sum;
            fp32_fma add(.a(old), .b(32'h3f800000), .c(v[r][g]), .y(sum));
            assign acc_new[r][g] = c_chunk == 0 ? v[r][g] : sum;
        end
    end

    // ---- drain: ER rows x 16 outputs of a finished tile per clock
    wire dr_active = st == RUN && dr_tile < c_tile;
    wire [19:0] dr_tout = tout(dr_tile);
    wire [19:0] dr_out0 = dr_tile * tile_outs + dr_og * 16;       // first output of this clock's lines
    wire [31:0] ts;
    fp32_fma tsmul(.a(hdr_ts), .b(r_ts), .c(32'd0), .y(ts));
    wire [31:0] tscale_use = r_int8 ? r_ts : ts;
    wire [31:0] y [0:ER-1][0:15], y2 [0:ER-1][0:15];
    wire [15:0] y_on [0:ER-1];
    wire [ER-1:0] er_on;
    function fgt(input [31:0] a, b);                   // a > b; no NaN, zeros are +0
        if (a[31] != b[31]) fgt = !a[31];
        else fgt = a[31] ? a[30:0] < b[30:0] : a[30:0] > b[30:0];
    endfunction
    // ARGW writes ER rows' indices per clock; ar_rb is its block of rows
    reg [9:0] ar_rb;
    for (r = 0; r < ER; r = r + 1) begin : ep
        wire [9:0] er = (st == ARGW ? ar_rb : dr_rb) * ER + r;
        assign er_on[r] = er < r_rows;
        for (g = 0; g < 16; g = g + 1) begin : o
            wire [31:0] a_ = acc[dr_tile[0] * ACCW + ((dr_og * 16 + g) * r_rows + er) % ACCW];
            fp32_fma m(.a(a_), .b(tscale_use), .c(32'd0), .y(y[r][g]));
            fp32_fma s(.a(d_rd_data[512*r + 32*g +: 32]), .b(32'h3f800000), .c(y[r][g]), .y(y2[r][g]));
            assign y_on[r][g] = dr_og * 16 + g < dr_tout;
        end
        reg [511:0] wdata; reg [63:0] wmask; integer q;
        always @* begin
            wdata = 0; wmask = 0;
            if (st == ARGW) begin wdata[31:0] = besti[er < MAXR ? er : 0] + r_id; wmask = 64'hf; end
            else for (q = 0; q < 16; q = q + 1) if (y_on[r][q]) begin
                wdata[32*q+:32] = r_epi == 1 ? y2[r][q] : y[r][q]; wmask[4*q+:4] = 4'hf;
            end
        end
        wire [22:0] line_addr = st == ARGW ? r_dst + er * r_ds : r_dst + er * r_ds + dr_out0 * 4;
        assign d_rd_valid[r] = dr_active && r_epi == 1 && er_on[r];
        assign d_rd_addr[23*r+:23] = line_addr;
        // A drain block's lines are written one clock after they are read (and computed), alongside the next
        // block's reads, so no line is read and written in the same clock (a bank does one or the other).
        reg pw_v; reg [22:0] pw_a; reg [511:0] pw_d; reg [63:0] pw_m;
        always @(posedge clk)
            if (reset) pw_v <= 0;
            else if (ce) begin
                pw_v <= st == RUN && dr_active && r_epi != 2 && er_on[r];
                pw_a <= line_addr; pw_d <= wdata; pw_m <= wmask;
            end
        assign o_wr_valid[r] = pw_v || (st == ARGW && er_on[r]);
        assign o_wr_addr[23*r+:23] = st == ARGW ? line_addr : pw_a;
        assign o_wr_data[512*r+:512] = st == ARGW ? wdata : pw_d;
        assign o_wr_mask[64*r+:64] = st == ARGW ? wmask : pw_m;
    end

    assign fifo_pop = ce && ((st == TAB && fifo_level >= 32) || (beat_go && beat_last_rg && !r_mms));
    assign fifo_pop_len = st == TAB ? 11'd32 : beat_bytes;

    // the shape checks, on the unlatched command
    wire i_int8 = wfmt_int8 || op == 3;
    wire [19:0] i_nch = (i_int8 && seg4) ? n_in >> 8 : n_in >> 10;
    wire [19:0] i_tout = !i_int8 ? 20'd64 : i_nch == 1 ? 20'd16 : n_out;
    wire bad = rows == 0 || rows > MAXR || n_in == 0 || n_out == 0 || i_tout * rows > ACCW ||
               (i_int8 ? (seg4 && n_out[1:0] != 0) || (seg4 ? n_in[7:0] : n_in[9:0]) != 0
                       : n_out[5:0] != 0 || n_in[9:0] != 0 || seg4);

    integer t, j, jj;
    reg [31:0] b_; reg [19:0] bi_;
    always @(posedge clk) begin
        done <= 0; fault <= 0;
        if (reset) st <= IDLE;
        else if (ce || st == IDLE) case (st)
            IDLE: if (start) begin
                r_mms <= op == 3; r_int8 <= i_int8; r_seg4 <= seg4; r_epi <= epi;
                r_nin <= n_in; r_nout <= n_out; r_rows <= rows; r_act <= act; r_exps <= exps; r_dst <= dst;
                r_wsrc <= wsrc; r_as <= act_stride; r_es <= exps_stride; r_ds <= dst_stride; r_id <= id_base;
                r_ts <= tscale;
                ld_seq <= 0; ld_done <= 0; ld_row <= 0; c_tile <= 0; c_chunk <= 0; c_k <= 0; c_rg <= 0;
                dr_tile <= 0; dr_og <= 0; dr_rb <= 0; ar_rb <= 0;
                if (op == 0) st <= FIN;
                else if (op == 1) st <= TAB;
                else if (bad) begin fault <= 1; st <= IDLE; end
                else st <= RUN;
            end
            TAB: if (fifo_level >= 32) begin
                table_r <= fifo_head[127:0]; hdr_ts <= fifo_head[159:128]; st <= FIN;
            end
            RUN: begin
                // loader
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
                // beats
                if (beat_go) begin
                    for (t = 0; t < RP; t = t + 1)
                        if (c_rg * RP + t < r_rows)
                            for (j = 0; j < 4; j = j + 1)
                                if (j < segw)
                                    acc[c_half * ACCW + ((c_k * segw + j) * r_rows + c_rg * RP + t) % ACCW] <= acc_new[t][j];
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
                // drain
                if (dr_active) begin
                    if (r_epi == 2)
                        for (t = 0; t < ER; t = t + 1)
                            if (dr_rb * ER + t < r_rows) begin
                                b_ = best[dr_rb * ER + t]; bi_ = besti[dr_rb * ER + t];
                                for (j = 0; j < 16; j = j + 1)
                                    if (y_on[t][j] && ((dr_tile == 0 && dr_og == 0 && j == 0) || fgt(y[t][j], b_))) begin
                                        b_ = y[t][j]; bi_ = dr_out0 + j;
                                    end
                                best[dr_rb * ER + t] <= b_; besti[dr_rb * ER + t] <= bi_;
                            end
                    if ((dr_rb + 1) * ER < r_rows) dr_rb <= dr_rb + 1;
                    else begin
                        dr_rb <= 0;
                        if ((dr_og + 1) * 16 < dr_tout) dr_og <= dr_og + 1;
                        else begin dr_og <= 0; dr_tile <= dr_tile + 1; end
                    end
                end
                if (dr_tile == ntiles) st <= r_epi == 2 ? ARGW : FIN;
            end
            ARGW: if ((ar_rb + 1) * ER < r_rows) ar_rb <= ar_rb + 1; else st <= FIN;
            FIN: begin done <= 1; st <= IDLE; end
        endcase
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
