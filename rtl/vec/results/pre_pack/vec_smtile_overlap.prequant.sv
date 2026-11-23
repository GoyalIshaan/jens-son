// Broadcast-scale R7 calendar. Offsets are enabled clocks from the row's
// state read. See design/calc/check_smtile_overlap.py for the reservations.
module vec_smtile_overlap(
    input wire clk, reset, start, full_width,
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
    localparam STATE_SLOTS=22, LAST=207;
    reg running;
    assign busy=running;
    reg [4:0] phase, next_slot;
    reg [15:0] next_row;
    reg [LAST:1] valid;
    wire [LAST:0] due={valid, running && phase==0 && next_row<rows};
    wire [20:0] tag[0:LAST];
    assign tag[0]={next_row,next_slot};
    for(genvar t=1;t<=LAST;t=t+1) begin : calendar_tag
        reg [20:0] row_tag;
        always @(posedge clk) if(ce && due[t-1]) row_tag<=tag[t-1];
        assign tag[t]=row_tag;
    end

    function automatic [31:0] fp_max(input [31:0] a,b);
        if(a[30:0]>31'h7f800000 || b[30:0]>31'h7f800000) fp_max=32'h7fc00000;
        else if(a[31]!=b[31]) fp_max=a[31] ? b : a;
        else fp_max=(a[30:0]>b[30:0]) ^ a[31] ? a : b;
    endfunction

    // One active event of each kind per clock. The per-beat offsets are
    // constant taps in each mode; phase stalls hold all taps together.
    wire [7:0] p1_issue, p1_return, subtract_issue, probability, p_write;
    for(genvar b=0;b<8;b=b+1) begin : beat
        assign p1_issue[b]=(full_width || b<2) && due[1+b];
        assign p1_return[b]=(full_width || b<2) && due[9+b];
        assign subtract_issue[b]=full_width ? due[17+b] : b<2 && due[11+b];
        assign probability[b]=full_width ? due[89+b] : b<2 && due[83+b];
        assign p_write[b]=full_width ? due[97+b] : b<2 && due[91+b];
    end

    function automatic [2:0] beat_index(input [7:0] events);
        beat_index=0;
        for(integer i=0;i<8;i=i+1)
            if(events[i]) beat_index=i[2:0];
    endfunction
    wire [2:0] rbeat=beat_index(p1_issue), sbeat=beat_index(p1_return);
    wire [2:0] subbeat=beat_index(subtract_issue), pbeat=beat_index(probability);
    wire [2:0] wbeat=beat_index(p_write);
    wire [20:0] rtag=tag[1+rbeat], stag=tag[9+sbeat];
    wire [20:0] subtag=full_width ? tag[17+subbeat] : tag[11+subbeat];
    wire [20:0] ptag=full_width ? tag[89+pbeat] : tag[83+pbeat];
    wire [20:0] wtag=full_width ? tag[97+wbeat] : tag[91+wbeat];
    wire [20:0] final_tag=full_width ? tag[196] : tag[166];
    wire [20:0] final_return=full_width ? tag[204] : tag[174];
    wire [20:0] state_tag=full_width ? tag[207] : tag[175];
    wire final_issue=full_width ? due[196] : due[166];

    // The SRAM ports are independent. A transfer accepted while the other
    // port stalls must not be issued again. Only an early read needs storage:
    // the frozen pipeline itself holds an early write's payload stable.
    reg read_taken, write_taken;
    reg [31:0] held_max, held_sum;
    reg [4095:0] held_scores, held_k;
    reg [2047:0] held_v;
    wire read_event=due[0] || (|p1_issue);
    wire state_event=full_width ? due[207] : due[175];
    wire write_event=state_event || (|p_write);
    assign rd_valid=running && read_event && !read_taken;
    assign wr_valid=running && write_event && !write_taken;
    assign rd_state=due[0];
    assign rd_row=rd_state ? next_row : rtag[20:5];
    assign rd_beat=rbeat;
    assign wr_state=state_event;
    assign wr_row=wr_state ? state_tag[20:5] : wtag[20:5];
    assign wr_beat=wbeat;
    assign ce=running && (!read_event || read_taken || rd_ready) &&
                        (!write_event || write_taken || wr_ready);
    wire [4095:0] score_in=read_taken ? held_scores : scores;
    wire [4095:0] k_in=read_taken ? held_k : k_scale;
    wire [2047:0] v_in=read_taken ? held_v : v_scale_half;

    reg [4095:0] score_slot[0:15];
    reg [2047:0] v_scale[0:7];
    reg [4095:0] sums[0:7];
    reg [31:0] old_m[0:STATE_SLOTS-1], old_l[0:STATE_SLOTS-1];
    reg [31:0] maximum[0:STATE_SLOTS-1], alpha_slot[0:STATE_SLOTS-1], l_slot[0:STATE_SLOTS-1];
    reg [31:0] running_max[0:1];
    wire [31:0] next_max=fp_max(running_max[stag[5]],beat_max);
    wire sum_p_slot=full_width && ptag[5];
    wire [4095:0] vs;
    wire [2047:0] v_half=v_scale[pbeat];
    for(genvar lane=0;lane<128;lane=lane+1) begin : widen_scale
        vec_unpack unpack_v({16'd0,v_half[16*lane+:16]},3'd1,1'b0,1'b0,vs[32*lane+:32]);
    end
    assign scaled=fy[16383:12288];
    assign exp_valid=full_width ? |due[32:25] : |due[20:19];
    assign alpha_valid=full_width ? due[25] : due[19];
    assign new_max=maximum[state_tag[4:0]];
    assign alpha=alpha_slot[state_tag[4:0]];
    assign new_sum=l_slot[state_tag[4:0]];

    wire [8:0] tree_issue;
    wire [8191:0] tree_left[0:8],tree_right[0:8];
    wire [16383:0] tree_input[0:8];
    wire [16383:0] tree_result[0:8];
    wire sum_tree_slot=full_width && tag[114][5];
    assign tree_input[0]=full_width ?
        {sums[{sum_tree_slot,2'd3}],sums[{sum_tree_slot,2'd2}],sums[{sum_tree_slot,2'd1}],sums[{sum_tree_slot,2'd0}]} :
        {8192'd0,sums[1],sums[0]};
    for(genvar level=0;level<9;level=level+1) begin : reduce
        localparam SHORT=85+9*level, LONG=114+9*level;
        localparam OUTPUTS=256>>level, BITS=32*OUTPUTS;
        localparam FIRST_LANE=level==0 ? 0 : 256;
        reg [BITS-1:0] result;
        assign tree_issue[level]=full_width ? due[LONG] : due[SHORT];
        if(level!=0) assign tree_input[level]=tree_result[level-1];
        for(genvar lane=0;lane<OUTPUTS;lane=lane+1) begin : pairs
            assign tree_left[level][32*lane+:32]=tree_input[level][64*lane+:32];
            assign tree_right[level][32*lane+:32]=tree_input[level][64*lane+32+:32];
        end
        if(BITS<8192) begin : padding
            assign tree_left[level][8191:BITS]=0;
            assign tree_right[level][8191:BITS]=0;
        end
        always @(posedge clk) if(ce && (full_width ? due[LONG+8] : due[SHORT+8]))
            result<=fy[FIRST_LANE*32+:BITS];
        assign tree_result[level]={{(16384-BITS){1'b0}},result};
    end

    always @* begin
        fa=0;
        fb={512{ONE}};
        fc=0;
        if(|p1_issue) begin
            fa[12288+:4096]=score_in;
            for(integer i=0;i<128;i=i+1)
                fb[12288+32*i+:32]=score_in[32*i+:32]==NINF ? ONE : k_in[32*i+:32];
        end
        if(|subtract_issue) begin
            fa[0+:4096]=score_slot[{subtag[5],subbeat}];
            fc[0+:4096]={128{maximum[subtag[4:0]] ^ 32'h80000000}};
        end
        if(subtract_issue[0]) begin
            fa[511*32+:32]=old_m[subtag[4:0]];
            fc[511*32+:32]=maximum[subtag[4:0]] ^ 32'h80000000;
        end
        if(|probability) begin
            fa[4096+:4096]=probs;
            fb[4096+:4096]=vs;
            if(pbeat[2]) begin
                fa[8192+:4096]=probs;
                fc[8192+:4096]=sums[{sum_p_slot,pbeat[1:0]}];
            end
        end
        if(tree_issue[0]) begin
            fa[0+:8192]=tree_left[0];
            fc[0+:8192]=tree_right[0];
        end
        for(integer level=1;level<9;level=level+1)
            if(tree_issue[level]) begin
                fa[8192+:4096]=tree_left[level][4095:0];
                fc[8192+:4096]=tree_right[level][4095:0];
            end
        if(final_issue) begin
            fa[511*32+:32]=old_l[final_tag[4:0]];
            fb[511*32+:32]=alpha_slot[final_tag[4:0]];
            fc[511*32+:32]=tree_result[8][31:0];
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
            phase<=phase==(full_width ? 5'd16 : 5'd7) ? 5'd0 : phase+1'b1;
            if(due[0]) begin
                old_m[next_slot]<=read_taken ? held_max : old_max;
                old_l[next_slot]<=read_taken ? held_sum : old_sum;
                running_max[next_row[0]]<=NINF;
                next_row<=next_row+1'b1;
                next_slot<=next_slot==STATE_SLOTS-1 ? 5'd0 : next_slot+1'b1;
            end
            if((|p1_issue) && rtag[20:5]==0) v_scale[rbeat]<=v_in;
            if(|p1_return) begin
                score_slot[{stag[5],sbeat}]<=scaled;
                running_max[stag[5]]<=next_max;
                if(sbeat==(full_width ? 3'd7 : 3'd1))
                    maximum[stag[4:0]]<=fp_max(old_m[stag[4:0]],next_max);
            end
            if((|probability) && !pbeat[2]) sums[{sum_p_slot,pbeat[1:0]}]<=probs;
            if(full_width && (|due[104:101])) begin : accumulate_return
                // Stored result returns eight clocks after the paired addition.
                reg [1:0] b;
                b=due[101] ? 0 : due[102] ? 1 : due[103] ? 2 : 3;
                sums[{tag[101+b][5],b}]<=fy[8192+:4096];
            end
            if(probability[0]) alpha_slot[ptag[4:0]]<=alpha_exp;
            if(full_width ? due[204] : due[174]) l_slot[final_return[4:0]]<=fy[511*32+:32];
            if(state_event && wr_row+17'd1==rows) begin running<=0; done<=1; end
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
