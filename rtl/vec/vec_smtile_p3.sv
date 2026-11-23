// Broadcast-scale R7 at 256 positions, one row every 3 enabled clocks (vec_smtile_overlap took 8).
// Same ports and arithmetic as vec_smtile_overlap's 256-position calendar; every R7 op and R1 pair is
// in the one-row order. Offsets are enabled clocks from the row's state read at T:
//   T          read old max and sum                                   (read port: T, T+1, T+2)
//   T+1, T+2   read score beats 0, 1; s' = s * ks on lanes 384..511 (Q3)
//   T+9, T+10  s' returns: keep it, running max; mn = max(m_old, row max) at T+10
//   T+11, T+12 s' - mn on lanes 0..127 (Q0), into the 128-unit exp pool at T+19, T+20
//   T+12       m_old - mn on lane 511 (Q3 is free on this residue), into alpha's exp at T+20
//   T+83, T+84 p leaves the exp pool: p * vs on lanes 128..255 (Q1); p enters the R1 tree
//   T+91, T+92 p * vs returns, into the 3-stage quantizer
//   T+94, T+95 its codes are written with their group exponents              (write port)
//   the tree, all on lanes 256..383 (Q2): levels 1..7 of each 128-position beat on lane slices of
//   64, 32, 16, 8, 4, 2, 1 lanes; a beat entering at e is at level l at e + 8(l-1). Beat 0's sum
//   (T+139) is held one clock; level 8 (beat 0 + beat 1) issues on lane 383 at T+140
//   T+148      l = fma(l_old, alpha, sum) on lane 383 (level 8 uses it only on the other residue)
//   T+156      write new max, alpha and l                                 (write port)
// The level that would add the +0 upper half of the 512 lanes is skipped: the sum is an FMA result,
// so it is never -0 and never below 2^-126, and adding +0 changes nothing.
// Reads use residues 0, 1, 2 of T; writes 1, 2 (P, T+94 and T+95) and 0 (state): one transfer per port per clock.
module vec_smtile_p3(
    input wire clk, reset, start,
    input wire [15:0] rows,
    output wire rd_valid, rd_state,
    output wire [15:0] rd_row,
    output wire [2:0] rd_beat,
    input wire rd_ready,
    input wire [31:0] old_max, old_sum,
    input wire [4095:0] scores, k_scale,
    input wire [2047:0] v_scale_half,
    output wire wr_valid, wr_state,
    output wire [15:0] wr_row,
    output wire [2:0] wr_beat,
    input wire wr_ready,
    output wire [31:0] new_max, alpha, new_sum,
    output reg done,
    output wire busy,
    output wire ce,
    output reg [16383:0] fa, fb, fc,
    input wire [16383:0] fy,
    input wire [4095:0] probs,
    input wire [31:0] alpha_exp,
    output wire [4095:0] scaled,
    input wire [31:0] beat_max,
    output wire exp_valid, alpha_valid
);
    localparam [31:0] ONE=32'h3f800000, NINF=32'hff800000;
    localparam SW=6, SLOTS=64, LAST=156, PERIOD=3, TAGW=16+SW;
    localparam Q0=0, Q1=128*32, Q2=256*32, Q3=384*32;
    reg running;
    assign busy=running;
    reg [1:0] phase;
    reg [15:0] next_row;
    reg [SW-1:0] next_slot;
    reg [LAST:1] valid;
    wire [LAST:0] due={valid, running && phase==0 && next_row<rows};
    wire [TAGW-1:0] tag[0:LAST];
    assign tag[0]={next_row,next_slot};
    for(genvar t=1;t<=LAST;t=t+1) begin : calendar_tag
        reg [TAGW-1:0] row_tag;
        always @(posedge clk) if(ce && due[t-1]) row_tag<=tag[t-1];
        assign tag[t]=row_tag;
    end
    function automatic [31:0] fp_max(input [31:0] a,b);
        if(a[30:0]>31'h7f800000 || b[30:0]>31'h7f800000) fp_max=32'h7fc00000;
        else if(a[31]!=b[31]) fp_max=a[31] ? b : a;
        else fp_max=(a[30:0]>b[30:0]) ^ a[31] ? a : b;
    endfunction
    function automatic [SW-1:0] slot_of(input [TAGW-1:0] t); slot_of=t[SW-1:0]; endfunction
    function automatic [15:0] row_of(input [TAGW-1:0] t); row_of=t[TAGW-1:SW]; endfunction

    // ---- events (one of each kind per clock: rows start 3 clocks apart)
    wire p1_issue=due[1] || due[2];       wire rbeat=due[2];
    wire p1_return=due[9] || due[10];     wire sbeat=due[10];
    wire sub_issue=due[11] || due[12];    wire subbeat=due[12];
    wire alpha_issue=due[12];
    wire prob_issue=due[83] || due[84];   wire pbeat=due[84];
    wire p_write=due[94] || due[95];      wire wbeat=due[95];
    wire final_issue=due[148];
    wire state_event=due[156];
    wire [TAGW-1:0] rtag=rbeat ? tag[2] : tag[1];
    wire [TAGW-1:0] stag=sbeat ? tag[10] : tag[9];
    wire [TAGW-1:0] subtag=subbeat ? tag[12] : tag[11];
    wire [TAGW-1:0] ptag=pbeat ? tag[84] : tag[83];
    wire [TAGW-1:0] wtag=wbeat ? tag[95] : tag[94];

    // ---- SRAM ports, with split handshakes held as in vec_smtile_overlap
    reg read_taken, write_taken;
    reg [31:0] held_max, held_sum;
    reg [4095:0] held_scores, held_k;
    reg [2047:0] held_v;
    wire read_event=due[0] || p1_issue;
    wire write_event=state_event || p_write;
    assign rd_valid=running && read_event && !read_taken;
    assign wr_valid=running && write_event && !write_taken;
    assign rd_state=due[0];
    assign rd_row=rd_state ? next_row : row_of(rtag);
    assign rd_beat={2'b0,rbeat};
    assign wr_state=state_event;
    assign wr_row=wr_state ? row_of(tag[156]) : row_of(wtag);
    assign wr_beat={2'b0,wbeat};
    assign ce=running && (!read_event || read_taken || rd_ready) &&
                        (!write_event || write_taken || wr_ready);
    wire [4095:0] score_in=read_taken ? held_scores : scores;
    wire [4095:0] k_in=read_taken ? held_k : k_scale;
    wire [2047:0] v_in=read_taken ? held_v : v_scale_half;

    // ---- storage
    reg [4095:0] score_slot[0:3];                     // {row parity, beat}: kept two clocks for the max
    reg [2047:0] v_scale[0:1];                        // broadcast: row 0's fp16 V scales
    reg [31:0] old_m[0:SLOTS-1], old_l[0:SLOTS-1], maximum[0:SLOTS-1], alpha_slot[0:SLOTS-1];
    reg [31:0] running_max, beat0_sum;
    wire [31:0] next_max=fp_max(sbeat ? running_max : NINF, beat_max);
    wire [4095:0] vs;
    wire [2047:0] v_half=v_scale[pbeat];
    for(genvar lane=0;lane<128;lane=lane+1) begin : widen_scale
        vec_unpack unpack_v({16'd0,v_half[16*lane+:16]},3'd1,1'b0,1'b0,vs[32*lane+:32]);
    end
    assign scaled=fy[Q3+:4096];
    assign exp_valid=due[19] || due[20];
    assign alpha_valid=due[20];
    assign new_max=maximum[slot_of(tag[156])];
    assign alpha=alpha_slot[slot_of(tag[156])];
    assign new_sum=fy[383*32+:32];

    // ---- the spatial tree on Q2: level l (1..7) on lanes 256 + OFF(l) .. + 128 >> l
    function automatic integer off(input integer l); off=128-(256>>l); endfunction  // 0, 64, 96, ..., 126
    reg [4095:0] tree_a, tree_c;
    always @* begin
        tree_a=0; tree_c=0;
        for(integer i=0;i<64;i=i+1) begin                        // level 1 from the exp pool
            tree_a[32*i+:32]=probs[64*i+:32]; tree_c[32*i+:32]=probs[64*i+32+:32];
        end
        for(integer l=2;l<=7;l=l+1)
            for(integer i=0;i<(128>>l);i=i+1) begin                // level l from level l-1's results
                tree_a[32*(off(l)+i)+:32]=fy[Q2+32*(off(l-1)+2*i)+:32];
                tree_c[32*(off(l)+i)+:32]=fy[Q2+32*(off(l-1)+2*i+1)+:32];
            end
    end

    always @* begin
        fa=0;
        fb={512{ONE}};
        fc=0;
        if(p1_issue) begin
            fa[Q3+:4096]=score_in;
            for(integer i=0;i<128;i=i+1)
                fb[Q3+32*i+:32]=score_in[32*i+:32]==NINF ? ONE : k_in[32*i+:32];
        end
        if(sub_issue) begin
            fa[Q0+:4096]=score_slot[{row_of(subtag) & 16'd1,subbeat}];
            fc[Q0+:4096]={128{maximum[slot_of(subtag)] ^ 32'h80000000}};
        end
        if(alpha_issue) begin
            fa[511*32+:32]=old_m[slot_of(tag[12])];
            fc[511*32+:32]=maximum[slot_of(tag[12])] ^ 32'h80000000;
        end
        if(prob_issue) begin
            fa[Q1+:4096]=probs;
            fb[Q1+:4096]=vs;
        end
        fa[Q2+:127*32]=tree_a[127*32-1:0];                // free-running: idle levels compute unused values
        fc[Q2+:127*32]=tree_c[127*32-1:0];
        if(due[140]) begin                                // level 8: beat 0's sum + beat 1's sum
            fa[383*32+:32]=beat0_sum;
            fc[383*32+:32]=fy[382*32+:32];
        end
        if(final_issue) begin
            fa[383*32+:32]=old_l[slot_of(tag[148])];
            fb[383*32+:32]=alpha_slot[slot_of(tag[148])];
            fc[383*32+:32]=fy[383*32+:32];
        end
    end

    always @(posedge clk) begin
        done<=0;
        if(reset || start) begin
            running<=start && !reset;
            next_row<=0; next_slot<=0; phase<=0; valid<=0;
            read_taken<=0; write_taken<=0;
        end else if(ce) begin
            read_taken<=0; write_taken<=0;
            valid<={valid[LAST-1:1],due[0]};
            phase<=phase==PERIOD-1 ? 2'd0 : phase+1'b1;
            if(due[0]) begin
                old_m[next_slot]<=read_taken ? held_max : old_max;
                old_l[next_slot]<=read_taken ? held_sum : old_sum;
                next_row<=next_row+1'b1;
                next_slot<=next_slot+1'b1;
            end
            if(p1_issue && row_of(rtag)==0) v_scale[rbeat]<=v_in;
            if(p1_return) begin
                score_slot[{row_of(stag) & 16'd1,sbeat}]<=scaled;
                running_max<=next_max;
                if(sbeat) maximum[slot_of(stag)]<=fp_max(old_m[slot_of(stag)],next_max);
            end
            if(due[84]) alpha_slot[slot_of(tag[84])]<=alpha_exp;
            if(due[139]) beat0_sum<=fy[382*32+:32];
            if(state_event && row_of(tag[156])+17'd1==rows) begin running<=0; done<=1; end
        end else if(running) begin
            if(rd_valid && rd_ready) begin
                read_taken<=1;
                held_max<=old_max; held_sum<=old_sum;
                held_scores<=scores; held_k<=k_scale; held_v<=v_scale_half;
            end
            if(wr_valid && wr_ready) write_taken<=1;
        end
    end
endmodule
