// Decoded ISA v1 box commands. The bank interface gathers fp32 operands at byte addresses.
// Requests and writes commit only on valid && ready. Keep the descriptor stable until done.
//
// Two paths:
//  STREAM (ADD SUB MUL FMA, ROPE, CONV4 without a saved window): one bank issue per clock.
//    Beats are issued in waves of 8: tap 0 of beats 0..7, then tap 1 of the same beats, ...
//    Each beat's previous-tap result leaves the 8-stage FMA exactly when its next tap issues, so it
//    is fed straight back (no accumulator registers); short waves are padded with bubbles to keep
//    that alignment. A last-tap result is written the clock it leaves the pipeline. A read stall at
//    the issue side or a write stall at the result side freezes the whole bank (bank_ce), which keeps
//    the schedule aligned with no result buffer. Rates: 1 beat/clock for single-tap ops, 2 clocks per
//    beat for ROPE, 4 for CONV4 (the timing model's rates).
//  LEGACY (CAUSAL, COPY, MAX, CONV4 with save_window): one beat at a time, as before.
module vec_box_command #(parameter LANES=512, parameter LOCAL_BANK=1)(
    input wire clk, reset, start,
    input wire [4:0] opcode,
    input wire [15:0] cols, rows, planes,
    input wire [22:0] base[0:3],
    input wire signed [19:0] sn[0:3], sm[0:3], sp[0:3],
    input wire [3:0] grouped,
    input wire negate,
    input wire [15:0] rot,
    input wire [31:0] position, limit, causal_step,
    input wire save_window, input wire [22:0] window_base,
    output wire done, output wire busy, output wire fault,
    output wire read_valid, input wire read_ready,
    output wire [22:0] addr_a[0:LANES-1], addr_b[0:LANES-1], addr_c[0:LANES-1],
    input wire [31:0] data_a[0:LANES-1], data_b[0:LANES-1], data_c[0:LANES-1],
    output wire write_valid, input wire write_ready,
    output wire [22:0] addr_y[0:LANES-1],
    output wire [31:0] data_y[0:LANES-1],
    output wire [LANES-1:0] mask,
    output wire [LANES*32-1:0] fa_bus,fb_bus,fc_bus,
    input wire [LANES*32-1:0] bank_result,
    output wire bank_ce,
    output wire [LANES-1:0] read_mask
);
    localparam ADD=1,SUB=2,MUL=3,MAXIMUM=4,FMA=5,COPY=6,ROPE=19,CAUSAL=20,CONV=21;
    localparam READ=0, WAIT_FMA=1, WINDOW=2, WRITE=3;
    localparam LAT=8, WAVE=8;
    wire rope_ok = rot<=64 && rot<=LANES && (LANES%64)==0 && !rot[0] && rot<=cols;
    wire supported = (opcode >= ADD && opcode <= COPY) || opcode==CAUSAL ||
                     opcode==CONV || (opcode==ROPE && rope_ok);
    wire stream = (opcode >= ADD && opcode <= FMA && opcode != MAXIMUM) || opcode==ROPE ||
                  (opcode==CONV && !save_window);
    wire [2:0] taps = opcode==CONV ? 3'd4 : opcode==ROPE ? 3'd2 : 3'd1;
    // Short rows pack: a power-of-two row narrower than the bank shares one beat with its neighbours
    // (along rows, or along planes when there is one row). Elementwise, so the bits do not change.
    function automatic [3:0] log2_16(input [15:0] v);
        integer b; log2_16 = 0;
        for (b = 0; b < 16; b = b + 1) if (v[b]) log2_16 = b;
    endfunction
    wire pow2 = cols != 0 && (cols & (cols - 16'd1)) == 0;
    wire pack_ok = stream && pow2 && cols >= 2 && cols < LANES && (rows > 1 || planes > 1);
    wire [3:0] pack_lg = pack_ok ? log2_16(cols) : 4'd0;
    wire pack_p = rows == 1;
    wire pk = pack_lg != 0;
    reg running;                                   // either path busy
    wire idle;                                     // not running, or finishing this clock (accepts the next start)
    assign busy = running;
    assign fault = !reset && idle && start && !supported;

    // ------------------------------------------------------------------ walker (both paths)
    wire walk_active, walk_done;
    wire [15:0] wn, wm, wp;
    wire [LANES-1:0] wmask;
    wire walk_advance;
    vec_box_walk #(LANES) walk(clk,reset,start && supported && idle,walk_advance,cols,rows,planes,
                               walk_active,walk_done,wn,wm,wp,wmask,pack_lg,pack_p);

    // ------------------------------------------------------------------ stream sequencing
    reg [2:0] s_tap; reg [2:0] s_i;
    reg [15:0] wb_n[0:WAVE-1], wb_m[0:WAVE-1], wb_p[0:WAVE-1];
    reg [LANES-1:0] wb_mask[0:WAVE-1];
    reg [WAVE-1:0] wb_valid;
    reg s_active;                                  // stream command in progress
    wire take = s_active && s_tap == 0 && walk_active;                   // load a new beat at this slot
    wire slot_valid = s_active && (s_tap == 0 ? walk_active : wb_valid[s_i]);
    wire [15:0] in_n = s_tap == 0 ? wn : wb_n[s_i];
    wire [15:0] in_m = s_tap == 0 ? wm : wb_m[s_i];
    wire [15:0] in_p = s_tap == 0 ? wp : wb_p[s_i];
    wire [LANES-1:0] in_mask = s_tap == 0 ? wmask : wb_mask[s_i];
    // result delay line (advances with the bank)
    reg [LAT-1:0] d_valid, d_last;
    reg [15:0] d_n[0:LAT-1], d_m[0:LAT-1], d_p[0:LAT-1];
    reg [LANES-1:0] d_mask[0:LAT-1];
    wire emerge = d_valid[LAT-1] && d_last[LAT-1];
    wire s_read = slot_valid;
    wire s_ce = (!s_read || read_ready) && (!emerge || write_ready);
    wire pending = |d_valid;
    wire s_finished = s_active && !walk_active && wb_valid == 0 && s_tap == 0 && !pending;

    // ------------------------------------------------------------------ legacy sequencing
    reg [1:0] phase, tap;
    reg [2:0] delay_count;
    reg l_active;
    wire l_advance = l_active && write_valid && write_ready && phase==WRITE;
    assign walk_advance = s_active ? (s_ce && take) : l_advance;
    wire window_tap = opcode==CONV && save_window && (wp==0 || tap==3);
    wire [16:0] l_hist = {1'b0,wp}+tap;
    wire last_tap = opcode==CONV ? tap==3 : opcode==ROPE ? tap==1 : 1'b1;

    // issue-side coordinates for the lane datapath
    wire [15:0] i_n = s_active ? in_n : wn;
    wire [15:0] i_m = s_active ? in_m : wm;
    wire [15:0] i_p = s_active ? in_p : wp;
    wire [1:0]  i_tap = s_active ? s_tap[1:0] : tap;

    assign read_valid = s_active ? s_read : (l_active && phase==READ);
    assign write_valid = s_active ? emerge : (l_active && (phase==WINDOW || phase==WRITE));
    assign read_mask = s_active ? in_mask : wmask;
    assign mask = s_active ? d_mask[LAT-1] : wmask;
    assign bank_ce = s_active ? s_ce : 1'b1;
    assign done = s_active ? s_finished : (l_active && walk_done);
    assign idle = !running || done;

    genvar j;
    for(j=0;j<LANES;j=j+1) begin : lane
        wire [15:0] slot = j >> pack_lg;
        wire [15:0] col = pk ? (j & (cols - 16'd1)) : i_n+j;
        wire [15:0] lm = pk && !pack_p ? i_m + slot : i_m;            // this lane's row and plane
        wire [15:0] lp = pk && pack_p ? i_p + slot : i_p;
        wire [16:0] history_index = {1'b0,lp} + i_tap;
        wire [15:0] half_rot = rot>>1;
        wire upper = col>=half_rot;
        wire rotated = opcode==ROPE && col<rot;
        wire [15:0] partner = upper ? col-half_rot : col+half_rot;
        wire from_hist = opcode==CONV && history_index<3;
        wire [15:0] source_p = opcode==CONV ? history_index-3 : lp;
        wire [22:0] aa, ba, ca, da, partner_addr, sa;
        vec_box_addr a_box(base[1],sn[1],sm[1],sp[1],col,lm,source_p,grouped[1],aa);
        vec_box_addr c_box(base[3],sn[3],sm[3],sp[3],col,lm,
                           opcode==CONV ? history_index[15:0] : lp,grouped[3],ca);
        vec_box_addr dst_box(base[0],sn[0],sm[0],sp[0],col,lm,lp,1'b0,da);
        vec_box_addr paired(base[1],sn[1],sm[1],sp[1],partner,lm,lp,1'b0,partner_addr);
        wire [15:0] trig_index = (upper ? partner : col) + (i_tap==1 ? 16'd32 : 16'd0);
        vec_box_addr b_box(base[2],sn[2],sm[2],sp[2],rotated ? trig_index : col,
                           rotated ? 16'd0 : lm,opcode==CONV ? 16'd0 : lp,grouped[2],ba);
        // stream write address from the emerging beat's coordinates
        wire [15:0] out_col=pk ? col : d_n[LAT-1]+j;
        wire [15:0] out_m=pk && !pack_p ? d_m[LAT-1] + slot : d_m[LAT-1];
        wire [15:0] out_p=pk && pack_p ? d_p[LAT-1] + slot : d_p[LAT-1];
        vec_box_addr out_box(base[0],sn[0],sm[0],sp[0],out_col,out_m,out_p,1'b0,sa);
        assign addr_a[j] = from_hist ? ca : rotated && i_tap==1 ? partner_addr : aa;
        assign addr_b[j] = opcode==CONV ? ba + {i_tap,2'b00} : ba;
        assign addr_c[j] = ca;
        wire [31:0] ax = data_a[j], bx = data_b[j], cx = data_c[j];
        wire [31:0] an = {ax[31] ^ negate, ax[30:0]};
        wire [31:0] fy;
        reg [31:0] acc, window_word, bypass;
        // ---- operands
        reg [31:0] fa, fb, fc;
        always @* begin
            if (s_active) begin
                fa = an; fb = 32'h3f800000; fc = 32'd0;
                case (opcode)
                    ADD: fc = bx;
                    SUB: fc = {~bx[31], bx[30:0]};
                    MUL: fb = bx;
                    FMA: begin fb = bx; fc = cx; end
                    CONV: begin fa = ax; fb = bx; fc = i_tap != 0 ? fy : 32'd0; end
                    ROPE: if (rotated) begin
                              fa = i_tap == 1 ? (upper ? ax : {~ax[31], ax[30:0]}) : ax;
                              fb = bx; fc = i_tap == 1 ? fy : 32'd0;
                          end else begin                 // unrotated dims pass through: fma(x, 1, 0) then + 0
                              fa = i_tap == 1 ? 32'd0 : ax; fb = 32'h3f800000; fc = i_tap == 1 ? fy : 32'd0;
                          end
                    default: ;
                endcase
            end else begin
                fa = rotated && i_tap==1 && !upper ? {~ax[31],ax[30:0]} : (opcode==CONV || rotated) ? ax : an;
                fb = opcode==ADD || opcode==SUB ? 32'h3f800000 : bx;
                fc = opcode==FMA ? cx : opcode==ADD ? bx : opcode==SUB ? {~bx[31],bx[30:0]} :
                     (rotated || opcode==CONV) && i_tap!=0 ? acc : 32'd0;
            end
        end
        assign fa_bus[32*j+:32] = fa;
        assign fb_bus[32*j+:32] = fb;
        assign fc_bus[32*j+:32] = fc;
        if (LOCAL_BANK) fp32_fma_pipe8_ce madd(clk,bank_ce,fa,fb,fc,fy);
        else assign fy=bank_result[32*j+:32];
        wire [31:0] max_value;
        wire a_less = an[31]!=bx[31] ? an[31] && (|{an[30:0],bx[30:0]}) :
                      an[31] ? an[30:0]>bx[30:0] : an[30:0]<bx[30:0];
        assign max_value = (an[30:0]>31'h7f800000 || bx[30:0]>31'h7f800000) ? 32'h7fc00000 : a_less ? bx : an;
        always @(posedge clk) begin
            if (!s_active && read_valid && read_ready) begin
                window_word <= ax;
                if (opcode==MAXIMUM) bypass<=max_value;
                else if (opcode==CAUSAL) bypass<=({1'b0,position}+col > {1'b0,limit}+wp*causal_step) ? 32'hff800000 : ax;
                else bypass<=opcode==COPY ? an : ax;
            end
            if(!s_active && phase==WAIT_FMA && delay_count==0) acc<=fy;
        end
        assign addr_y[j] = s_active ? sa : phase==WINDOW ? window_base+((l_hist*rows+wm)*cols+col)*4 : da;
        assign data_y[j] = s_active ? fy : phase==WINDOW ? window_word :
                           opcode==CAUSAL || opcode==COPY || opcode==MAXIMUM ||
                           (opcode==ROPE && !rotated) ? bypass : acc;
    end

    integer t;
    always @(posedge clk) begin
        if (reset) begin
            running <= 0; s_active <= 0; l_active <= 0; d_valid <= 0; d_last <= 0; wb_valid <= 0;
            s_tap <= 0; s_i <= 0; phase <= READ; tap <= 0;
        end else begin
            // ---- stream path
            if (s_active && s_ce) begin
                d_valid <= {d_valid[LAT-2:0], slot_valid};
                d_last <= {d_last[LAT-2:0], s_tap == taps - 1};
                for (t = LAT - 1; t > 0; t = t - 1) begin
                    d_n[t] <= d_n[t-1]; d_m[t] <= d_m[t-1]; d_p[t] <= d_p[t-1]; d_mask[t] <= d_mask[t-1];
                end
                d_n[0] <= in_n; d_m[0] <= in_m; d_p[0] <= in_p; d_mask[0] <= in_mask;
                if (s_tap == 0) begin
                    wb_valid[s_i] <= walk_active;
                    wb_n[s_i] <= wn; wb_m[s_i] <= wm; wb_p[s_i] <= wp; wb_mask[s_i] <= wmask;
                end
                if (s_i == WAVE - 1) begin
                    s_i <= 0;
                    s_tap <= (s_tap == taps - 1) ? 3'd0 : s_tap + 1'b1;
                    if (s_tap == taps - 1 && taps != 1) wb_valid <= 0;     // wave finished
                    if (taps == 1) wb_valid <= 0;
                end else s_i <= s_i + 1'b1;
            end
            if (s_finished) begin s_active <= 0; running <= 0; end
            // ---- legacy path
            if (l_active) case(phase)
                READ: if(read_ready) begin
                    if(opcode==CAUSAL || opcode==COPY || opcode==MAXIMUM) phase<=WRITE;
                    else begin delay_count<=7; phase<=WAIT_FMA; end
                end
                WAIT_FMA: if(delay_count!=0) delay_count<=delay_count-1'b1;
                          else if(window_tap) phase<=WINDOW;
                          else if(last_tap) phase<=WRITE;
                          else begin tap<=tap+1'b1; phase<=READ; end
                WINDOW: if(write_ready) begin
                    if(last_tap) phase<=WRITE;
                    else begin tap<=tap+1'b1; phase<=READ; end
                end
                WRITE: if(write_ready) begin tap<=0; phase<=READ; end
            endcase
            if (l_active && walk_done) begin l_active <= 0; running <= 0; end
            // a start in the finishing clock wins over the clears above
            if (idle && start && supported) begin
                running <= 1; s_active <= stream; l_active <= !stream;
                s_tap <= 0; s_i <= 0; wb_valid <= 0; d_valid <= 0; phase <= READ; tap <= 0;
            end
        end
    end
endmodule
