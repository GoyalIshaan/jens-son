// R7, the fused softmax tile, for one row of up to 1,024 positions (groups of 32), streaming.
//   pass 1, one 128-position beat per clock:  s' = s * ks (FMA; a -inf input stays -inf), running max
//   then mn = max(m_old, row max); alpha = exp(m_old - mn) (its own exp unit, hidden under pass 2)
//   pass 2, one beat per clock, three ops side by side in quarters of the shared 512-lane bank:
//     lanes   0..127  p = exp(s' - mn): the subtraction, then the 128-unit exp pool
//     lanes 128..255  p * vs, then int8 with a power-of-two exponent per 32 (R7), written as one beat:
//                     128 codes + 4 group exponents
//     lanes 256..383  the R1 lane sums: position j goes to lane j mod 512 (the first 512 are stored, the
//                     rest added with an FMA)
//   then the R1 pairwise tree over the 512 lane sums (9 levels) and l = fma(l_old, alpha, sum).
// Stalls: a read stall in pass 1 or a write stall in pass 2 freezes the bank, the exp pool and the
// bookkeeping together (bank_ce). The tree and the final FMA touch no SRAM.
// The broadcast fast path and serial fallback share the exp and quantizer pools.
module vec_smtile #(parameter LOCAL_BANK=1, parameter W=128, parameter OVERLAP=0)(
    input wire clk,reset,start,input wire [5:0] groups,
    input wire [31:0] old_max,old_sum,
    output wire [2:0] read_beat,output wire read_valid,input wire read_ready,
    input wire [W*32-1:0] scores,k_scale,v_scale,
    output wire write_valid,input wire write_ready,output wire [2:0] write_beat,
    output wire [W*8-1:0] codes,output wire [(W/32)*10-1:0] exponents,output wire [W-1:0] write_mask,
    output wire [31:0] new_max,new_sum,alpha, output wire done,busy,
    output wire [512*32-1:0] fa_bus,fb_bus,fc_bus,
    input wire [512*32-1:0] bank_result,
    output wire bank_ce,
    input wire overlap,
    input wire [15:0] overlap_rows,
    input wire [(W*16)-1:0] v_scale_half,
    output wire read_state,write_state,
    output wire [15:0] read_row,write_row
);
    wire fast_mode=OVERLAP && overlap;
    reg [31:0] serial_max,serial_sum,serial_alpha;
    reg serial_done;
    wire fast_read,fast_write,fast_ce,fast_done,fast_busy,fast_exp_valid,fast_alpha_valid;
    wire [2:0] fast_rbeat,fast_wbeat;
    wire [31:0] fast_max,fast_sum,fast_alpha;
    wire [16383:0] fast_fa,fast_fb,fast_fc;
    wire [W*32-1:0] fast_scaled;
    localparam LAT=8, ELAT=64, G=W/32, QL=3;          // QL: the pipelined quantizer's latency
    localparam IDLE=0, P1=1, P2=2, TREE=3, FIN=4;
    localparam [31:0] ONE=32'h3f800000, NINF=32'hff800000;
    reg [2:0] phase;
    wire serial_busy = phase != IDLE;
    wire [3:0] beats = (groups + G - 1) / G;          // 1..8
    reg [3:0] rb;                                      // next beat to read / issue
    reg [3:0] emerged;
    // buffers
    reg [W*32-1:0] sbuf[0:7], vsbuf[0:7];
    reg [512*32-1:0] acc;
    reg [31:0] rowmax;
    wire [512*32-1:0] fy;
    function [31:0] fmax(input [31:0] a,b);
        begin
            if(a[30:0]>31'h7f800000 || b[30:0]>31'h7f800000) fmax=32'h7fc00000;
            else if(a[31]!=b[31]) fmax=a[31] ? b : a;
            else fmax=(a[30:0]>b[30:0]) ^ a[31] ? a : b;
        end
    endfunction
    reg [31:0] old_max_r, old_sum_r;
    wire [31:0] mn = fmax(old_max_r, rowmax);

    // ------------------------------------------------------------------ pass 1: scale and max
    wire p1_issue = phase==P1 && rb < beats;
    wire [W-1:0] masked_in;
    genvar j;
    for (j = 0; j < W; j = j + 1) begin : msk
        assign masked_in[j] = (rb * W + j >= groups * 32) || scores[32*j+:32] == NINF;
    end
    wire p1v; wire [2:0] p1b; wire [W-1:0] p1m;
    wire ce;
    vec_delay_ce #(4+W,LAT) d1(clk, ce, {p1_issue, rb[2:0], masked_in}, {p1v, p1b, p1m});
    wire [W*32-1:0] scaled, serial_scaled;
    assign scaled=fast_mode ? fast_scaled : serial_scaled;
    for (j = 0; j < W; j = j + 1) begin : scl assign serial_scaled[32*j+:32] = p1m[j] ? NINF : fy[32*j+:32]; end
    wire [31:0] mt[1:2*W-1];
    for (j = 1; j < W; j = j + 1) begin : mtree assign mt[j] = fmax(mt[2*j], mt[2*j+1]); end
    for (j = 0; j < W; j = j + 1) begin : mleaf assign mt[W+j] = scaled[32*j+:32]; end

    // ------------------------------------------------------------------ pass 2
    reg [3:0] sb;                                      // next beat for the subtraction
    reg alpha_issued;
    wire p2_issue = phase==P2 && sb < beats;
    wire alpha_slot = phase==P2 && !alpha_issued;      // m_old - mn rides on lane 511 of the first issue
    wire s1v; wire [2:0] s1b; wire s1a;
    vec_delay_ce #(5,LAT) d2(clk, ce, {p2_issue, sb[2:0], alpha_slot}, {s1v, s1b, s1a});
    // exp pool (128) and alpha's own unit
    wire [W*32-1:0] pin, pout, prob;
    assign pin = (fast_mode ? fast_exp_valid : phase==P2 && s1v) ? fy[W*32-1:0] : 0;
    for (j = 0; j < W; j = j + 1) begin : pool sfu_exp_core_ce #(8) u(clk, bank_ce, pin[32*j+:32], pout[32*j+:32]); end
    wire [31:0] alpha_y;
    wire alpha_input_valid=fast_mode ? fast_alpha_valid : phase==P2 && s1a;
    sfu_exp_core_ce #(8) ua(clk, bank_ce, alpha_input_valid ? fy[511*32+:32] : 32'd0, alpha_y);
    wire e1v; wire [2:0] e1b; wire e1a;
    vec_delay_ce #(5,ELAT) d3(clk, ce, {s1v, s1b, s1a}, {e1v, e1b, e1a});
    // Unused lanes in the last beat are R1 zero padding, including when mn is nonfinite.
    for (j = 0; j < W; j = j + 1) begin : pad
        assign prob[32*j+:32] = e1b * W + j < groups * 32 ? pout[32*j+:32] : 32'd0;
    end
    // p * vs and the lane sums
    wire second_half = e1b >= 4;
    wire q1v; wire [2:0] q1b;
    vec_delay_ce #(4,LAT+QL) d4(clk, ce, {e1v, e1b}, {q1v, q1b});      // written once quantized
    wire a1v; wire [2:0] a1b;
    vec_delay_ce #(4,LAT) d5(clk, ce, {e1v && second_half, e1b}, {a1v, a1b});
    // quantize the emerging p * vs beat
    for (j = 0; j < G; j = j + 1) begin : qz
        act_quant32_pipe q(clk, bank_ce, fy[(W + 32*j)*32 +: 1024], codes[256*j+:256], exponents[10*j+:10]);
    end
    wire serial_write=phase==P2 && q1v;
    assign write_valid=fast_mode ? fast_write : serial_write;
    assign write_beat=fast_mode ? fast_wbeat : q1b;
    for (j = 0; j < W; j = j + 1) begin : wm
        assign write_mask[j] = fast_mode || (q1b * W + j) < groups * 32;
    end
    assign read_valid=fast_mode ? fast_read : p1_issue;
    assign read_beat=fast_mode ? fast_rbeat : rb[2:0];
    assign ce=!(p1_issue && !read_ready) && !(serial_write && !write_ready);
    assign bank_ce=fast_mode ? fast_ce : (phase==P1 || phase==P2) ? ce : 1'b1;

    assign {new_max,new_sum,alpha}=fast_mode ? {fast_max,fast_sum,fast_alpha} :
                                           {serial_max,serial_sum,serial_alpha};
    assign done=fast_mode ? fast_done : serial_done;
    assign busy=fast_mode ? fast_busy : serial_busy;

    if(OVERLAP) begin : streaming
        // 1,024 positions: the 17-clock calendar (vec_smtile_overlap). 256: a row every 3 clocks (vec_smtile_p3).
        wire wide=groups==32;
        wire o_read,o_write,o_ce,o_done,o_busy,o_exp,o_alpha,o_rstate,o_wstate;
        wire p_read,p_write,p_ce,p_done,p_busy,p_exp,p_alpha,p_rstate,p_wstate;
        wire [2:0] o_rbeat,o_wbeat,p_rbeat,p_wbeat;
        wire [15:0] o_rrow,o_wrow,p_rrow,p_wrow;
        wire [31:0] o_max,o_sum,o_alpha_v,p_max,p_sum,p_alpha_v;
        wire [16383:0] o_fa,o_fb,o_fc,p_fa,p_fb,p_fc;
        wire [W*32-1:0] o_scaled,p_scaled;
        vec_smtile_overlap engine(
            .clk(clk),.reset(reset),.start(start && fast_mode && wide),.full_width(1'b1),
            .rows(overlap_rows),.rd_valid(o_read),.rd_state(o_rstate),
            .rd_row(o_rrow),.rd_beat(o_rbeat),.rd_ready(read_ready),
            .old_max(scores[31:0]),.old_sum(k_scale[31:0]),
            .scores(scores),.k_scale(k_scale),.v_scale_half(v_scale_half),
            .wr_valid(o_write),.wr_state(o_wstate),.wr_row(o_wrow),
            .wr_beat(o_wbeat),.wr_ready(write_ready),
            .new_max(o_max),.alpha(o_alpha_v),.new_sum(o_sum),
            .done(o_done),.busy(o_busy),.ce(o_ce),
            .fa(o_fa),.fb(o_fb),.fc(o_fc),.fy(fy),.probs(pout),
            .alpha_exp(alpha_y),.scaled(o_scaled),.beat_max(mt[1]),
            .exp_valid(o_exp),.alpha_valid(o_alpha));
        vec_smtile_p3 engine256(
            .clk(clk),.reset(reset),.start(start && fast_mode && !wide),
            .rows(overlap_rows),.rd_valid(p_read),.rd_state(p_rstate),
            .rd_row(p_rrow),.rd_beat(p_rbeat),.rd_ready(read_ready),
            .old_max(scores[31:0]),.old_sum(k_scale[31:0]),
            .scores(scores),.k_scale(k_scale),.v_scale_half(v_scale_half),
            .wr_valid(p_write),.wr_state(p_wstate),.wr_row(p_wrow),
            .wr_beat(p_wbeat),.wr_ready(write_ready),
            .new_max(p_max),.alpha(p_alpha_v),.new_sum(p_sum),
            .done(p_done),.busy(p_busy),.ce(p_ce),
            .fa(p_fa),.fb(p_fb),.fc(p_fc),.fy(fy),.probs(pout),
            .alpha_exp(alpha_y),.scaled(p_scaled),.beat_max(mt[1]),
            .exp_valid(p_exp),.alpha_valid(p_alpha));
        assign {fast_read,fast_write,fast_ce,fast_done,fast_busy,fast_exp_valid,fast_alpha_valid,read_state,write_state}=wide ?
            {o_read,o_write,o_ce,o_done,o_busy,o_exp,o_alpha,o_rstate,o_wstate} :
            {p_read,p_write,p_ce,p_done,p_busy,p_exp,p_alpha,p_rstate,p_wstate};
        assign {fast_rbeat,fast_wbeat,read_row,write_row,fast_max,fast_sum,fast_alpha}=wide ?
            {o_rbeat,o_wbeat,o_rrow,o_wrow,o_max,o_sum,o_alpha_v} : {p_rbeat,p_wbeat,p_rrow,p_wrow,p_max,p_sum,p_alpha_v};
        assign fast_fa=wide ? o_fa : p_fa;
        assign fast_fb=wide ? o_fb : p_fb;
        assign fast_fc=wide ? o_fc : p_fc;
        assign fast_scaled=wide ? o_scaled : p_scaled;
    end else begin : no_streaming
        assign {fast_read,fast_write,fast_ce,fast_done,fast_busy,fast_exp_valid,fast_alpha_valid}=0;
        assign {fast_rbeat,fast_wbeat,fast_max,fast_sum,fast_alpha}=0;
        assign {read_state,write_state,read_row,write_row}=0;
        assign fast_fa=0;
        assign fast_fb=0;
        assign fast_fc=0;
        assign fast_scaled=0;
    end

    // ------------------------------------------------------------------ tree and final
    reg [3:0] lvl, tcnt;
    reg [512*32-1:0] tvec;

    // ------------------------------------------------------------------ bank operands
    wire [W*32-1:0] score_beat = sbuf[sb[2:0]];
    wire [W*32-1:0] scale_beat = vsbuf[e1b];
    wire [W*32-1:0] sum_beat = acc[32*(e1b[1:0]*W)+:W*32];
    for (j = 0; j < 512; j = j + 1) begin : lane
        reg [31:0] a, b, c;
        always @* begin
            a = 0; b = ONE; c = 0;
            case (phase)
                P1: if (j < W) begin a = scores[32*j+:32]; b = k_scale[32*j+:32]; end
                P2: begin
                    if (j < W) begin a = score_beat[32*j+:32]; c = {~mn[31], mn[30:0]}; end
                    else if (j < 2*W) begin a = pout[32*(j-W)+:32]; b = scale_beat[32*(j-W)+:32]; end
                    else if (j < 3*W) begin a = prob[32*(j-2*W)+:32]; c = sum_beat[32*(j-2*W)+:32]; end
                    else if (j == 511) begin a = old_max_r; c = {~mn[31], mn[30:0]}; end
                end
                TREE: if (j < 256) begin a = tvec[32*(2*j)+:32]; c = tvec[32*(2*j+1)+:32]; end
                FIN: if (j == 0) begin a = old_sum_r; b = serial_alpha; c = tvec[31:0]; end
                default: ;
            endcase
        end
        assign fa_bus[32*j+:32] = fast_mode ? fast_fa[32*j+:32] : a;
        assign fb_bus[32*j+:32] = fast_mode ? fast_fb[32*j+:32] : b;
        assign fc_bus[32*j+:32] = fast_mode ? fast_fc[32*j+:32] : c;
        if (LOCAL_BANK) fp32_fma_pipe8_ce madd(clk, bank_ce, fa_bus[32*j+:32], fb_bus[32*j+:32], fc_bus[32*j+:32], fy[32*j+:32]);
        else assign fy[32*j+:32] = bank_result[32*j+:32];
    end

    integer k;
    always @(posedge clk) begin
        serial_done <= 0;
        if (reset) phase <= IDLE;
        else case (phase)
            IDLE: if (start && !fast_mode) begin
                old_max_r <= old_max; old_sum_r <= old_sum; rowmax <= NINF;
                rb <= 0; emerged <= 0; sb <= 0; alpha_issued <= 0; acc <= 0; phase <= P1;
            end
            P1: if (ce) begin
                if (p1_issue) begin vsbuf[rb[2:0]] <= v_scale; rb <= rb + 1'b1; end
                if (p1v) begin
                    sbuf[p1b] <= scaled; rowmax <= fmax(rowmax, mt[1]);
                    emerged <= emerged + 1'b1;
                    if (emerged + 1'b1 == beats) phase <= P2;
                end
            end
            P2: if (ce) begin
                if (p2_issue) sb <= sb + 1'b1;
                if (alpha_slot) alpha_issued <= 1;
                // first-half p goes straight into the lane sums; second-half sums return from the bank
                if (e1v && !second_half) acc[32*(e1b*W)+:W*32] <= prob;
                if (a1v) acc[32*((a1b % 4)*W)+:W*32] <= fy[2*W*32+:W*32];
                if (q1v && q1b == beats - 1) begin
                    // last beat written: its lane sum (if any) returns in the same clock
                    phase <= TREE; lvl <= 0; tcnt <= 0;
                end
            end
            // operands come from tvec; with the bank free-running, its output LAT + 1 edges after tvec
            // changes is that level's result
            TREE: begin
                if (tcnt == 0) begin tvec <= acc; tcnt <= 1; end
                else if (tcnt == LAT + 1) begin
                    tvec <= fy; tcnt <= 1;
                    if (lvl == 8) phase <= FIN; else lvl <= lvl + 1'b1;
                end else tcnt <= tcnt + 1'b1;
            end
            FIN: begin
                if (tcnt == LAT + 1) begin serial_sum <= fy[31:0]; serial_max <= mn; serial_done <= 1; phase <= IDLE; end
                else tcnt <= tcnt + 1'b1;
            end
        endcase
    end
    // alpha is ready ELAT + LAT clocks after its slot: latch it when its delay line says so
    wire av;
    vec_delay_ce #(1,ELAT) d6(clk, ce, s1a, av);
    always @(posedge clk) if (phase==P2 && ce && av) serial_alpha <= alpha_y;
endmodule
