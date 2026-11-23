// Row operations, streaming, AW lanes per beat (default 128 = 4 groups of 32).
//   unary special functions (EXP SIGMOID SILU SOFTPLUS RSQRT RECIP): y = f(a*b + c)
//   QUANT (int8 + one exponent per 32), QKV (int8 + one fp16 scale per row, two passes), DEQ,
//   RMAX and MATCH (scans, one scalar result).
//
// Stream: one beat read every S clocks -> the pre-op a*b + c on AW lanes of the shared FMA bank (a*1 + 0
// when there is none: exact, since chip values are never subnormal or -0) -> for unary ops, the function's
// unit pool, fed in S sub-beats of U elements -> one write per beat. Unit pools are sized to D-020/D-023:
// 128 exp, 64 sigmoid (SiLU = sigmoid and one multiply), 2 each of softplus, 1/sqrt, 1/x; so S = 1, 2 or 64.
// QUANT, QKV and DEQ use no unit pool (S = 1). A stalled read or write freezes the bank, the unit pools and
// all bookkeeping together (ce), so no result buffer is needed.
// Sidebands travel in the same write: QUANT's group exponents in group_exp (one per 32-group of the beat),
// QKV's row scale in kv_scale; the caller places them on spare write lanes.
// Every op walks all rows and planes of the command: one cursor walks columns, then rows, then planes,
// and every beat carries its (plane, row) tag to its write, so the pipeline latency is paid once per command.
// RMAX and MATCH scan a row, write its scalar, then scan the next. QKV scans row r + 1 while row r's
// stream beats are still in the bank; each beat carries its own row's fp16 scale to the write.
module vec_row_ops #(parameter AW=128)(
    input wire clk,reset,start,input wire [4:0] opcode,input wire [15:0] count,rows,planes,
    output wire [15:0] read_row,read_plane,write_row,write_plane,
    input wire has_b,has_c,negate,input wire signed [9:0] exp_bias,
    output wire [15:0] col,output wire read_valid,input wire read_ready,
    input wire [AW*32-1:0] a,b,c,
    output wire write_valid,input wire write_ready,output wire [AW-1:0] lane_mask,
    output wire [15:0] write_col,
    output wire [AW*32-1:0] data,output wire [(AW/32)*10-1:0] group_exp,output wire [15:0] kv_scale,
    output reg done,output wire busy,
    output wire [AW*32-1:0] fa,fb,fc,input wire [AW*32-1:0] fy,
    output wire bank_ce
);
    localparam RMAX=15,QUANT=16,QKV=17,DEQ=18,MATCH=23;
    localparam EXP=7,SIG=8,SILU=9,SPLUS=10,RSQ=11,RCP=12;
    localparam G=AW/32, LAT=8;
    localparam IDLE=0,SCAN=1,SCALE=2,STREAM=3,SCALAR=4;
    reg [2:0] phase;
    wire unary = opcode>=EXP && opcode<=RCP;
    wire scalar_out = opcode==MATCH || opcode==RMAX;
    // units per function and sub-beats per beat
    localparam UE=AW, US=AW>=128 ? 64 : AW/2, UO=2;      // exp, sigmoid/SiLU, the other three
    wire [7:0] S = opcode==EXP ? 1 : (opcode==SIG || opcode==SILU) ? AW/US : unary ? AW/UO : 1;
    wire [7:0] U = opcode==EXP ? UE : (opcode==SIG || opcode==SILU) ? US : UO;
    // sub-beats a beat starting at column c2 really needs: only its active elements (a short row's
    // 2 values take 1 sub-beat, not S)
    function [7:0] subs(input [15:0] column, row_count,
                        input is_unary, input [7:0] full_beat, units);
        reg [16:0] left;
        begin
            left = row_count - column;
            if (!is_unary || left >= AW) subs = full_beat;
            else subs = (left + units - 1) / units;
        end
    endfunction
    assign busy = phase!=IDLE;

    // ------------------------------------------------------------------ scan pass (QKV, RMAX, MATCH)
    reg [15:0] scol, rcol;                    // scan column; stream read column
    reg [15:0] rrow, rplane;                  // stream read row and plane
    wire [15:0] nrows = opcode==MATCH ? 16'd1 : rows, nplanes = opcode==MATCH ? 16'd1 : planes;   // MATCH is one row (ISA)
    reg [15:0] kv_cur;                        // the current row's QKV scale
    wire row_end = {1'b0,rcol}+AW >= count;
    wire last_row = rrow==nrows-1'b1 && rplane==nplanes-1'b1;
    wire last_read = row_end && last_row;
    reg [31:0] max_so_far;
    reg [15:0] matched;
    reg [31:0] inv_scale;
    reg [AW-1:0] mism;
    wire [15:0] cur_col = phase==SCAN ? scol : rcol;
    assign col = cur_col;
    wire [AW-1:0] active;
    genvar j;
    function [31:0] fmax(input [31:0] a_,b_);
        if(a_[30:0]>31'h7f800000 || b_[30:0]>31'h7f800000) fmax=32'h7fc00000;
        else if(a_[31]!=b_[31]) fmax=a_[31] ? b_ : a_;
        else fmax=(a_[30:0]>b_[30:0]) ^ a_[31] ? a_ : b_;
    endfunction
    wire [31:0] maximum[1:2*AW-1];
    for(j=1;j<AW;j=j+1) begin : mtree assign maximum[j]=fmax(maximum[2*j],maximum[2*j+1]); end
    for(j=0;j<AW;j=j+1) begin : lanes
        wire [31:0] ax=a[32*j+:32];
        assign active[j]=({1'b0,cur_col}+j)<count;
        assign maximum[AW+j]=!active[j] ? 32'hff800000 : opcode==QKV ? {1'b0,ax[30:0]} : ax ^ {negate,31'd0};
        always @* mism[j]=active[j] && ax!=b[32*j+:32];
    end
    reg [15:0] first_mism; integer s_;
    always @* begin first_mism=AW; for(s_=AW-1;s_>=0;s_=s_-1) if(mism[s_]) first_mism=s_; end
    wire [31:0] beat_max=fmax(max_so_far,maximum[1]);
    wire scan_last={1'b0,scol}+AW>=count;
    wire [15:0] kvs16; wire [31:0] kvinv;
    vec_kv_scale row_scale(max_so_far[30:0],kvs16,kvinv);

    // ------------------------------------------------------------------ stream pass
    reg [7:0] sub, sub_n;                     // phase within the current beat's sub-beats; reads at sub == 0
    reg reads_left;
    wire do_read = phase==STREAM && reads_left && sub==0;
    wire [7:0] read_subs = subs(rcol, count, unary, S, U);
    // pre-op operands (lane j)
    for(j=0;j<AW;j=j+1) begin : pre
        wire on = ({1'b0,rcol}+j)<count;
        wire [31:0] ax=a[32*j+:32];
        assign fa[32*j+:32]=on ? ax ^ {negate,31'd0} : 0;
        assign fb[32*j+:32]=!on ? 32'h3f800000 : opcode==QKV ? inv_scale : (has_b || opcode==DEQ) ? b[32*j+:32] : 32'h3f800000;
        assign fc[32*j+:32]=unary && has_c && on ? c[32*j+:32] : 0;
    end
    // bank metadata: (valid, plane, row, column)
    wire ce;
    wire m1v; wire [15:0] m1c, m1r, m1p;
    vec_row_tag_delay #(64,LAT) meta1(clk, reset, ce, do_read, {kv_cur, rplane, rrow, rcol}, m1v, {kv_scale, m1p, m1r, m1c});
    // non-unary results at bank emergence; QUANT's codes three clocks later (the quantizer is pipelined)
    localparam QL=3;
    wire m4v; wire [15:0] m4c, m4r, m4p;
    vec_row_tag_delay #(48,QL) meta4(clk, reset, ce, m1v && opcode==QUANT, {m1p, m1r, m1c}, m4v, {m4p, m4r, m4c});
    wire [AW*8-1:0] qcodes, kvcodes; wire [G*10-1:0] qexp;
    for(j=0;j<G;j=j+1) begin : quant
        wire [9:0] e;
        act_quant32_pipe qz(clk, ce, fy[1024*j+:1024], qcodes[256*j+:256], e);
        assign qexp[10*j+:10] = e + exp_bias;
    end
    for(j=0;j<AW;j=j+1) begin : kvround vec_round_i8 rq(fy[32*j+:32], kvcodes[8*j+:8]); end

    // unary: x buffer and dispatch of S sub-beats
    reg [AW*32-1:0] xbuf;
    reg [7:0] dk; reg dvalid; reg [15:0] dcol, drow, dplane;   // dispatch counter and tags for the beat in xbuf
    wire disp0 = phase==STREAM && m1v && unary;   // a beat arrives: dispatch sub-beat 0 from fy now
    wire [7:0] k_now = disp0 ? 8'd0 : dk;
    wire disp = phase==STREAM && (disp0 || dvalid);
    wire [AW*32-1:0] xsrc = disp0 ? fy : xbuf;
    // unit pools
    wire [UE*32-1:0] ye; wire [US*32-1:0] ys, ysilu, xs_d; wire [UO*32-1:0] ysp, yq, yr;
    for(j=0;j<UE;j=j+1) begin : pe sfu_exp_core_ce #(8) u(clk, ce, xsrc[32*j+:32], ye[32*j+:32]); end
    wire [US*32-1:0] xs_in = xsrc >> (k_now * US * 32);
    for(j=0;j<US;j=j+1) begin : ps
        sfu_sigmoid_core_ce #(9) u(clk, ce, xs_in[32*j+:32], ys[32*j+:32]);   // 9: input-registered FMAs (1 GHz)
        vec_delay_ce #(32,118) xd(clk, ce, xs_in[32*j+:32], xs_d[32*j+:32]);
        fp32_fma_pipe8_ce m(clk, ce, xs_d[32*j+:32], ys[32*j+:32], 32'd0, ysilu[32*j+:32]);
    end
    wire [UO*32-1:0] xo_in = xsrc >> (k_now * UO * 32);
    for(j=0;j<UO;j=j+1) begin : po
        sfu_softplus_core_ce #(8) u1(clk, ce, xo_in[32*j+:32], ysp[32*j+:32]);
        sfu_rsqrt_core_ce #(9) u2(clk, ce, xo_in[32*j+:32], yq[32*j+:32]);
        sfu_recip_core_ce #(8) u3(clk, ce, xo_in[32*j+:32], yr[32*j+:32]);
    end
    // pool metadata: (valid, last sub-beat, sub-beat index, column), one delay line per latency
    // only the active function's line gets entries, so nothing is left in flight for a later command
    wire [7:0] beat_subs = subs(disp0 ? m1c : dcol, count, unary, S, U);
    wire [56:0] md_in = {k_now==beat_subs-1, k_now, disp0 ? m1p : dplane, disp0 ? m1r : drow,
                         disp0 ? m1c : dcol};  // last, sub-beat, plane, row, column
    wire [57:0] md64, md104, md112, md136, md40, md24;
    vec_row_tag_delay #(57,64) d64(clk, reset, ce, disp && opcode==EXP, md_in, md64[57], md64[56:0]);
    vec_row_tag_delay #(57,118) d104(clk, reset, ce, disp && opcode==SIG, md_in, md104[57], md104[56:0]);
    vec_row_tag_delay #(57,126) d112(clk, reset, ce, disp && opcode==SILU, md_in, md112[57], md112[56:0]);
    vec_row_tag_delay #(57,137) d136(clk, reset, ce, disp && opcode==SPLUS, md_in, md136[57], md136[56:0]);
    vec_row_tag_delay #(57,48) d40(clk, reset, ce, disp && opcode==RSQ, md_in, md40[57], md40[56:0]);
    vec_row_tag_delay #(57,24) d24(clk, reset, ce, disp && opcode==RCP, md_in, md24[57], md24[56:0]);
    reg [57:0] md_out; reg [AW*32-1:0] ypool; reg [7:0] unit_w;
    always @* begin
        case(opcode)
            EXP:   begin md_out=md64;  unit_w=UE; ypool=ye; end
            SIG:   begin md_out=md104; unit_w=US; ypool=ys; end
            SILU:  begin md_out=md112; unit_w=US; ypool=ysilu; end
            SPLUS: begin md_out=md136; unit_w=UO; ypool=ysp; end
            RSQ:   begin md_out=md40;  unit_w=UO; ypool=yq; end
            default: begin md_out=md24; unit_w=UO; ypool=yr; end
        endcase
    end
    wire mo_v = md_out[57], mo_last = md_out[56]; wire [7:0] mo_k = md_out[55:48];
    wire [15:0] mo_p = md_out[47:32], mo_r = md_out[31:16], mo_c = md_out[15:0];
    reg [AW*32-1:0] obuf;
    // merged output beat: obuf with the emerging sub-beat placed at mo_k
    reg [AW*32-1:0] merged; integer w_;
    always @* begin
        merged = obuf;
        for (w_ = 0; w_ < AW; w_ = w_ + 1)
            if (w_ / unit_w == mo_k) merged[32*w_+:32] = ypool[32*(w_ % unit_w)+:32];
    end
    // write side
    wire wr_unary = unary && mo_v && mo_last;
    wire wr_plain = !unary && (opcode==QUANT ? m4v : m1v);
    // QKV's stream beats can still be draining while the next row scans
    wire s_write = (phase==STREAM || phase==SCAN || phase==SCALE) && (wr_unary || wr_plain);
    assign ce = !((do_read || phase==SCAN) && !read_ready) && !(s_write && !write_ready);
    assign bank_ce = (phase==STREAM || phase==SCAN || phase==SCALE) ? ce : 1'b1;
    wire [15:0] wcol = unary ? mo_c : opcode==QUANT ? m4c : m1c;
    assign write_col = phase==SCALAR ? 16'd0 : wcol;
    assign write_row = phase==SCALAR ? rrow : unary ? mo_r : opcode==QUANT ? m4r : m1r;
    assign write_plane = phase==SCALAR ? rplane : unary ? mo_p : opcode==QUANT ? m4p : m1p;
    assign read_row = rrow;
    assign read_plane = rplane;
    for(j=0;j<AW;j=j+1) begin : outl
        assign lane_mask[j] = phase==SCALAR ? j==0 : ({1'b0,wcol}+j)<count;
        assign data[32*j+:32] = phase==SCALAR ? (j==0 ? (opcode==RMAX ? max_so_far : {16'd0,matched}) : 32'd0) :
                                unary ? merged[32*j+:32] :
                                opcode==QUANT ? {24'd0,qcodes[8*j+:8]} : opcode==QKV ? {24'd0,kvcodes[8*j+:8]} : fy[32*j+:32];
    end
    assign group_exp = qexp;
    assign read_valid = phase==SCAN || do_read;
    assign write_valid = s_write || phase==SCALAR;
    // beats read but not yet written
    reg [9:0] inflight;

    always @(posedge clk) begin
        done <= 0;
        if (reset) begin phase <= IDLE; inflight <= 0; dvalid <= 0; end
        else case (phase)
            IDLE: if (start) begin
                scol <= 0; rcol <= 0; rrow <= 0; rplane <= 0; sub <= 0; sub_n <= 1; matched <= 0; dvalid <= 0; inflight <= 0;
                max_so_far <= opcode==QKV ? 0 : 32'hff800000;
                reads_left <= count != 0 && nrows != 0 && nplanes != 0;
                if (count==0 || nrows==0 || nplanes==0) done <= 1;
                else phase <= (scalar_out || opcode==QKV) ? SCAN : STREAM;
            end
            SCAN: if (ce) begin
                max_so_far <= beat_max;
                if (opcode==MATCH && matched==scol)
                    matched <= |mism ? scol+first_mism : scan_last ? count : scol+AW;
                if (scan_last) phase <= scalar_out ? SCALAR : SCALE;
                else scol <= scol+AW;
                inflight <= inflight - s_write;
            end
            SCALE: if (ce) begin
                kv_cur <= kvs16; inv_scale <= kvinv; phase <= STREAM; rcol <= 0; sub <= 0; sub_n <= 1;
                inflight <= inflight - s_write;
            end
            SCALAR: if (write_ready) begin
                if (last_row) begin done <= 1; phase <= IDLE; end
                else begin
                    if (rrow != nrows-1'b1) rrow <= rrow + 1'b1; else begin rrow <= 0; rplane <= rplane + 1'b1; end
                    scol <= 0; matched <= 0; max_so_far <= 32'hff800000; phase <= SCAN;
                end
            end
            STREAM: if (ce) begin
                sub <= (do_read ? (read_subs <= 1) : (sub + 1'b1 >= sub_n)) ? 8'd0 : sub + 1'b1;
                if (do_read) sub_n <= read_subs;
                if (do_read) begin
                    if (last_read) reads_left <= 0;
                    if (!row_end) rcol <= rcol + AW;
                    else begin
                        rcol <= 0;
                        if (rrow != nrows-1'b1) rrow <= rrow + 1'b1;
                        else begin rrow <= 0; rplane <= rplane + 1'b1; end
                        // QKV: scan the next row while this one drains
                        if (opcode==QKV && !last_read) begin scol <= 0; max_so_far <= 0; phase <= SCAN; end
                    end
                end
                // dispatch bookkeeping
                if (disp0) begin
                    xbuf <= fy; dcol <= m1c; drow <= m1r; dplane <= m1p;
                    if (beat_subs > 1) begin dvalid <= 1; dk <= 1; end
                end else if (dvalid) begin
                    if (dk == beat_subs-1) dvalid <= 0; else dk <= dk + 1'b1;
                end
                // assemble unary results
                if (unary && mo_v) obuf <= merged;
                inflight <= inflight + do_read - s_write;
                if (!reads_left && inflight + do_read - s_write == 0 && !do_read) begin done <= 1; phase <= IDLE; end
            end
        endcase
    end
endmodule

// Clear validity at reset. The metadata payload can retain its contents
// until a valid beat arrives.
module vec_row_tag_delay #(parameter WIDTH=16, CYCLES=8)(
    input wire clk, reset, ce, in_valid, input wire [WIDTH-1:0] payload,
    output wire out_valid, output wire [WIDTH-1:0] delayed_payload
);
    reg [CYCLES-1:0] valid;
    always @(posedge clk) begin
        if (reset) valid <= 0;
        else if (ce) valid <= (valid << 1) | in_valid;
    end
    assign out_valid = valid[CYCLES-1];
    vec_delay_ce #(WIDTH,CYCLES) data_delay(clk, ce, payload, delayed_payload);
endmodule

// QUANT and DEQ at full width (WW = 512 lanes per beat, one beat per clock): the same arithmetic as
// vec_row_ops (pre-op a*b + 0 on the shared FMA bank, then act_quant32 per 32-group for QUANT), four
// times the lanes. QUANT's 512 codes leave as 128 packed words (lane k holds codes 4k..4k+3, low byte
// first) so its 16 group exponents fit on write lanes 128..143; the caller writes them raw (U32 / E8).
// lane_mask covers the exponent lanes too (lane 128 + g: group g of the beat is real).
// Short rows pack (pack_lg = log2 n, for n = 32..256 and more than one row): lane j holds column j % n of
// slot j / n; slots run along rows (m, m+1, ...) or, with one row, along planes. A group of 32 never
// straddles two slots, so each slot's codes and exponents are exactly its own row's.
// One cursor walks columns (or slot groups), rows and planes; every beat carries its tags to its write.
module vec_rows_wide #(parameter WW=512)(
    input wire clk, reset, start, input wire [4:0] opcode, input wire [15:0] count, rows, planes,
    input wire has_b, negate, input wire signed [9:0] exp_bias,
    output wire [15:0] col, read_row, read_plane, output wire read_valid, input wire read_ready,
    output wire [WW-1:0] read_mask,
    input wire [WW*32-1:0] a, b,
    output wire write_valid, input wire write_ready, output wire [WW-1:0] lane_mask,
    output wire [15:0] write_col, write_row, write_plane,
    output wire [WW*32-1:0] data, output wire [(WW/32)*10-1:0] group_exp,
    output wire [3:0] pack_lg, output wire pack_p,
    output reg done, output wire busy,
    output wire [WW*32-1:0] fa, fb, fc, input wire [WW*32-1:0] fy, output wire bank_ce
);
    localparam QUANT=16, DEQ=18, LAT=8, QL=3, G=WW/32;
    reg run, reads_left; reg [15:0] rcol, rrow, rplane; reg [9:0] inflight;
    reg [3:0] lg; reg pp;                                // the command's packing, latched at start
    assign busy = run; assign pack_lg = lg; assign pack_p = pp;
    wire is_q = opcode == QUANT;
    // packing of the starting command
    wire [3:0] lg_in = count == 32 ? 5 : count == 64 ? 6 : count == 128 ? 7 : count == 256 ? 8 : 0;
    wire pk_in = lg_in != 0 && (rows > 1 || planes > 1);
    wire pk = lg != 0;
    wire [15:0] slots = 16'(WW) >> lg;                   // slots per packed beat
    wire [15:0] left = !pk ? 16'hffff : pp ? planes - rplane : rows - rrow;   // slots still to go
    wire grp_end = left <= slots;
    wire row_end = pk ? (pp || grp_end) : {1'b0,rcol} + WW >= count;
    wire last_read = pk ? (pp ? grp_end : grp_end && rplane == planes - 1'b1)
                        : row_end && rrow == rows - 1'b1 && rplane == planes - 1'b1;
    wire do_read = run && reads_left;
    genvar j;
    for (j = 0; j < WW; j = j + 1) begin : pre
        wire [15:0] slot = pk ? 16'(j) >> lg : 16'd0;
        wire on = pk ? slot < left : ({1'b0,rcol} + j) < count;
        assign read_mask[j] = on;
        assign fa[32*j+:32] = on ? a[32*j+:32] ^ {negate,31'd0} : 32'd0;
        assign fb[32*j+:32] = on && (has_b || !is_q) ? b[32*j+:32] : 32'h3f800000;
        assign fc[32*j+:32] = 32'd0;
    end
    wire ce;
    wire m1v, m4v; wire [15:0] m1c, m1r, m1p, m1l, m4c, m4r, m4p, m4l;
    vec_row_tag_delay #(64,LAT) meta1(clk, reset, ce, do_read, {left, rplane, rrow, rcol}, m1v, {m1l, m1p, m1r, m1c});
    vec_row_tag_delay #(64,QL) meta4(clk, reset, ce, m1v && is_q, {m1l, m1p, m1r, m1c}, m4v, {m4l, m4p, m4r, m4c});
    wire [WW*8-1:0] q;
    for (j = 0; j < G; j = j + 1) begin : quant
        wire [9:0] e;
        act_quant32_pipe qz(clk, ce, fy[1024*j+:1024], q[256*j+:256], e);
        assign group_exp[10*j+:10] = e + exp_bias;
    end
    wire wv = run && (is_q ? m4v : m1v);
    assign ce = !(do_read && !read_ready) && !(wv && !write_ready);
    assign bank_ce = run ? ce : 1'b1;
    assign write_col = is_q ? m4c : m1c;
    assign write_row = is_q ? m4r : m1r;
    assign write_plane = is_q ? m4p : m1p;
    wire [15:0] wleft = is_q ? m4l : m1l;
    // a write element e (its position in the beat) is real when its slot is, or, unpacked, its column is
    function on_w(input [15:0] e);
        on_w = pk ? (e >> lg) < wleft : ({1'b0,write_col} + e) < count;
    endfunction
    for (j = 0; j < WW; j = j + 1) begin : outl
        if (j < WW/4) begin : packed_lane
            assign lane_mask[j] = is_q ? on_w(16'(4*j)) : on_w(16'(j));
            assign data[32*j+:32] = is_q ? q[32*j+:32] : fy[32*j+:32];
        end else if (j < WW/4 + G) begin : exp_lane
            assign lane_mask[j] = is_q ? on_w(16'(32*(j-WW/4))) : on_w(16'(j));
            assign data[32*j+:32] = fy[32*j+:32];
        end else begin : plain_lane
            assign lane_mask[j] = !is_q && on_w(16'(j));
            assign data[32*j+:32] = fy[32*j+:32];
        end
    end
    assign col = rcol; assign read_row = rrow; assign read_plane = rplane;
    assign read_valid = do_read;
    assign write_valid = wv;
    always @(posedge clk) begin
        done <= 0;
        if (reset) begin run <= 0; inflight <= 0; lg <= 0; pp <= 0; end
        else if (!run) begin
            if (start) begin
                rcol <= 0; rrow <= 0; rplane <= 0; inflight <= 0;
                lg <= pk_in ? lg_in : 4'd0; pp <= pk_in && rows == 1;
                if (count == 0 || rows == 0 || planes == 0) done <= 1;
                else begin run <= 1; reads_left <= 1; end
            end
        end else if (ce) begin
            if (do_read) begin
                if (last_read) reads_left <= 0;
                if (pk && pp) rplane <= rplane + slots;
                else if (pk && !grp_end) rrow <= rrow + slots;
                else if (!pk && !row_end) rcol <= rcol + WW;
                else begin
                    rcol <= 0;
                    if (!pk && rrow != rows - 1'b1) rrow <= rrow + 1'b1; else begin rrow <= 0; rplane <= rplane + 1'b1; end
                end
            end
            inflight <= inflight + do_read - wv;
            if (!reads_left && !do_read && inflight == wv) begin done <= 1; run <= 0; end
        end
    end
endmodule
