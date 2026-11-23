// Decoded V-queue entry to bank transactions. The queue holds the resolved
// descriptor stable until done. Addresses are byte addresses; SRAM gathers
// return raw words with the addressed byte in the low bits. Writes carry a
// format tag: F32, F16, I8, U32 or E8. For QKV scales and codes and QUANT
// exponents, wr_raw selects an already packed value. No physical SRAM macro
// or scalar/descriptor RAM is assumed at this boundary.
// The unbuffered SRAM commits a write on wr_valid && wr_ready. A buffered
// client sets SRAM_WRITES_IMMEDIATE=0 and asserts wr_visible after the tile's
// accepted writes have reached the SRAM visibility point.
module vec_unit #(parameter LANES=512, DIM=LANES/4, AUXW=128, SRAM_WRITES_IMMEDIATE=1)(
    input wire clk,reset,start,output wire ready,output reg done,fault,
    input wire [4:0] op,input wire [15:0] n,m,p,
    input wire [22:0] base[0:3],input wire signed [19:0] sn[0:3],sm[0:3],sp[0:3],
    input wire [2:0] dtype[0:3],input wire has_b,has_c,negate,
    input wire signed [9:0] exp_bias,input wire [15:0] rot,
    input wire [31:0] position,limit,causal_step,input wire save_window,input wire [22:0] window_base,
    input wire [15:0] head,key_head,key_heads,value_heads,replay_count,token_count,
    input wire [1:0] dn_extra_heads,input wire commit_spec,input wire [22:0] work_base,
    input wire [22:0] vs_offset,e_offset,e_stride,mlb,
    output reg rd_valid,input wire rd_ready,
    output wire [22:0] rd_addr[0:2][0:LANES-1],
    output wire [2:0] rd_type[0:2][0:LANES-1],
    output wire [LANES-1:0] rd_mask[0:2],
    input wire [31:0] rd_data[0:2][0:LANES-1],
    output reg wr_valid,input wire wr_ready,
    output wire [LANES-1:0] wr_mask,output wire [22:0] wr_addr[0:LANES-1],
    output wire [31:0] wr_data[0:LANES-1],output wire [2:0] wr_type[0:LANES-1],
    output wire [LANES-1:0] wr_raw,input wire wr_visible
);
    localparam F32=0,F16=1,I8=2,U32=3,E8=4;
    localparam IDLE=0,LAUNCH=1,RUN=2,DN_FETCH=3,TILE_LOAD=4,ROW_RESULT=5,
               EXP_WRITE=6,HEAD_RESULT=7,TILE_DRAIN=8;
    reg [3:0] phase;
    reg [15:0] row_idx,plane_idx,col_idx,token;
    reg from_log;
    reg [31:0] old_max,old_sum,scalar_y;
    reg [15:0] last_aux_col;
    reg [9:0] saved_exp;
    reg [15:0] saved_scale;
    reg [4*DIM*32-1:0] k,q,v,head_buf;
    reg [127:0] beta,decay;
    reg [1:0] transfer_slot;
    wire [1:0] dn_slot, dn_wslot;
    wire dn_read_dst;
    wire [$clog2(DIM/4)-1:0] dn_wgroup;
    wire [15:0] value_head=head+transfer_slot;
    // The decoder supplies the first key head. Advance at a GQA boundary
    // while gathering consecutive value heads, including unaligned groups.
    reg [15:0] fetch_key;
    wire [31:0] next_value_scaled=({1'b0,value_head}+1'b1)*key_heads;
    wire [31:0] next_key_scaled=({1'b0,fetch_key}+1'b1)*value_heads;
    wire is_dn=op==22,is_tile=op==24,is_red=op==13 || op==14;
    wire is_aux=(op>=7 && op<=12) || (op>=15 && op<=18) || op==23;
    wire is_box=!is_dn && !is_tile && !is_red && !is_aux && op!=0;
    // Row ops stream every row and plane in one launch (vec_row_ops tags each beat with its row).
    wire aux_stream=is_aux;
    wire [15:0] aux_rrow,aux_rplane,aux_wrow,aux_wplane;
    wire [15:0] red_rrow,red_rplane,red_col,red_wrow,red_wplane;
    wire red_r,red_w;
    wire [LANES-1:0] red_rmask,red_wmask; wire [LANES*32-1:0] red_bus; wire [3:0] red_lg; wire red_pp;
    wire rpk=is_red && red_lg!=0;                  // packed R1: lane j belongs to row slot j >> red_lg
    wire apk=rpk || wpk; wire [3:0] alg=rpk ? red_lg : wide_lg; wire app=rpk ? red_pp : wide_pp;
    wire [15:0] rrow=aux_stream ? aux_rrow : is_red ? red_rrow : row_idx, rplane=aux_stream ? aux_rplane : is_red ? red_rplane : plane_idx;
    wire [15:0] wrow=aux_stream ? aux_wrow : is_red ? red_wrow : row_idx, wplane=aux_stream ? aux_wplane : is_red ? red_wplane : plane_idx;
    wire box_r,box_w,tile_r,tile_w,state_r,state_w,head_valid;
    wire exec_ready,exec_done,exec_fault,aux_r,aux_w;
    wire [LANES-1:0] box_mask, box_read_mask;
    wire [22:0] ba[0:LANES-1],bb[0:LANES-1],bc[0:LANES-1],by[0:LANES-1];
    wire [31:0] box_y[0:LANES-1],red_y,mn,ln,alpha;
    localparam AG=AUXW/32;
    wire [AUXW*32-1:0] aux_y;
    wire [AUXW-1:0] aux_mask;wire [15:0] aux_col,aux_wcol,aux_scale;
    wire [AG*10-1:0] aux_exp,tile_exp;wire [AUXW*8-1:0] tile_q;
    wire [2:0] tile_wbeat;wire [AUXW-1:0] tile_wmask;
    wire [2:0] tile_idx;
    wire tile_read_state,tile_write_state;
    wire [15:0] tile_read_row,tile_write_row;
    wire [AUXW*16-1:0] tile_v_half;
    wire [$clog2(DIM/4)-1:0] group_idx;
    wire [LANES*32-1:0] state_out,read0,red_x;
    wire [AUXW*32-1:0] read1,read2;
    // QUANT and DEQ at full width (vec_rows_wide): QUANT needs contiguous int8 codes (4 per packed word)
    wire wide_on; wire [LANES-1:0] wide_wmask; wire [LANES*32-1:0] wide_y,read1w; wire [(LANES/32)*10-1:0] wide_exp;
    wire wide_ok=LANES>=512 && AUXW==LANES/4 && (op==18 || (op==16 && sn[0]==20'sd1 && dtype[0]==3'd2 && n[1:0]==2'd0) ||
                 (op==17 && sn[0]==20'sd1 && dtype[0]==3'd2 && (n==32 || n==64 || n==128 || n==256) && (m>1 || p>1)));
    wire wq=wide_on && (op==16 || op==17);   // codes leave 4 to a word
    wire [LANES-1:0] wide_rmask; wire [3:0] wide_lg; wire wide_pp;
    wire wpk=wide_on && wide_lg!=0;                // packed QUANT/DEQ: lane j belongs to slot j >> wide_lg
    wire [4*DIM*32-1:0] head_y;
    // Conservative broadcast fast path. Unsigned interval arithmetic is
    // used only for the dense, positive-stride layouts enabled below. The
    // high-exclusive endpoint must fit SRAM, including the last row.
    function automatic [79:0] tile_range(input [39:0] low,size);
        tile_range={low,low+size};
    endfunction
    function automatic disjoint(input [79:0] a,b);
        disjoint=a[39:0]<=b[79:40] || b[39:0]<=a[79:40];
    endfunction
    wire [39:0] last_row={24'd0,m}-40'd1;
    wire [39:0] score_bytes=last_row*$unsigned(sm[1])+({24'd0,n}<<2);
    wire [39:0] scale_bytes={24'd0,n}<<1;
    wire [39:0] code_bytes=last_row*$unsigned(sm[0])+{24'd0,n};
    wire [39:0] exp_bytes=last_row*e_stride+({24'd0,n}>>5);
    wire [39:0] stat_bytes={24'd0,m}<<2;
    wire [39:0] dst_base={17'd0,base[0]}, scale_base={17'd0,base[2]}, state_base={17'd0,base[3]};
    wire [79:0] inputs[0:2],outputs[0:4];
    assign inputs[0]=tile_range({17'd0,base[1]},score_bytes);
    assign inputs[1]=tile_range(scale_base,scale_bytes);
    assign inputs[2]=tile_range(scale_base+vs_offset,scale_bytes);
    assign outputs[0]=tile_range(dst_base,code_bytes);
    assign outputs[1]=tile_range(dst_base+e_offset,exp_bytes);
    assign outputs[2]=tile_range(state_base,stat_bytes);
    assign outputs[3]=tile_range(state_base+({17'd0,mlb}<<2),stat_bytes);
    assign outputs[4]=tile_range(state_base+40'd5*mlb,stat_bytes);
    wire [2:0] input_bounds;
    wire [4:0] output_bounds, input_disjoint;
    wire [24:0] output_disjoint;
    for(genvar i=0;i<3;i=i+1) begin : tile_input_range
        assign input_bounds[i]=inputs[i][39:0]<=40'd8388608;
    end
    for(genvar i=0;i<5;i=i+1) begin : tile_output_range
        assign output_bounds[i]=outputs[i][39:0]<=40'd8388608;
        assign input_disjoint[i]=disjoint(outputs[i],inputs[0]) &&
                                 disjoint(outputs[i],inputs[1]) &&
                                 disjoint(outputs[i],inputs[2]);
        for(genvar j=0;j<5;j=j+1) begin : other_output
            assign output_disjoint[5*i+j]=j<=i || disjoint(outputs[i],outputs[j]);
        end
    end
    wire fast_tile=LANES==512 && AUXW==128 && is_tile && m>1 && p==1 &&
        (n==256 || n==1024) && dtype[0]==I8 && dtype[1]==F32 && dtype[2]==F16 && dtype[3]==F32 &&
        sn[0]==1 && sn[1]==4 && sn[2]==2 && sm[2]==0 && sp[2]==0 && sm[3]==4 &&
        $signed(sm[0]) >= $signed({4'd0,n}) &&
        $signed(sm[1]) >= $signed({2'd0,n,2'b00}) && e_stride>=n/32 &&
        (&input_bounds) && (&output_bounds) && (&input_disjoint) && (&output_disjoint);
    wire tile_state_read=phase==TILE_LOAD || (fast_tile && tile_read_state);
    wire tile_state_write=phase==ROW_RESULT && is_tile || (fast_tile && tile_write_state);
    wire [15:0] tile_addr_read_row=fast_tile ? tile_read_row : row_idx;
    wire [15:0] tile_addr_write_row=fast_tile ? tile_write_row : row_idx;
    reg run_finished;
    wire token_advances=token+17'd1<(from_log ? replay_count : token_count);
    wire dn_work_read=!from_log && !commit_spec && token!=0;
    wire dn_work_write=!from_log && !commit_spec;
    wire [22:0] token_base=from_log ? base[2]+token*{{3{sp[2][19]}},sp[2]} : base[1]+token*{{3{sp[1][19]}},sp[1]};
    wire [22:0] qkv_width=(2*key_heads+value_heads)*DIM;
    wire [15:0] inner_col=is_tile ? {6'd0,tile_idx,7'd0} : is_red ? red_col : aux_col;
    wire [15:0] wcol=is_tile ? {6'd0,tile_wbeat,7'd0} : aux_wcol;   // row-op and tile writes trail their reads
    // Only the optional elementwise operands broadcast a power-of-two scale.
    wire elementwise=(op>=1 && op<=12);
    wire [3:0] grouped={elementwise && dtype[3]==E8,elementwise && dtype[2]==E8,2'b00};
    wire [31:0] core_a[0:LANES-1],core_b[0:LANES-1],core_c[0:LANES-1];
    genvar j;
    for(j=0;j<LANES;j=j+1) begin : ports
        vec_unpack decode_a(rd_data[0][j],rd_type[0][j],grouped[1],op==23,core_a[j]);
        vec_unpack decode_b(rd_data[1][j],rd_type[1][j],grouped[2],op==23,core_b[j]);
        vec_unpack decode_c(rd_data[2][j],rd_type[2][j],grouped[3],1'b0,core_c[j]);
        assign read0[32*j+:32]=core_a[j];
        assign read1w[32*j+:32]=core_b[j];
        assign red_x[32*j+:32]=core_a[j] ^ {negate,31'd0};      // the reducer zeroes lanes outside its mask
        if(j<AUXW) begin
            assign read1[32*j+:32]=core_b[j];
            assign read2[32*j+:32]=core_c[j];
            assign tile_v_half[16*j+:16]=rd_data[2][j][15:0];
        end
    end
    // DN is split at token boundaries so token operands are fetched through
    // the ordinary bank interface and output writes can be backpressured.
    vector_exec #(LANES,DIM,AUXW) execution(.clk(clk),.reset(reset),.start(phase==LAUNCH && exec_ready),.opcode(op),
        .ready(exec_ready),.done(exec_done),.fault(exec_fault),.cols(n),.rows(m),.planes(p),.rot(rot),
        .base(base),.sn(sn),.sm(sm),.sp(sp),.grouped(grouped),.negate(negate),.position(position),
        .limit(limit),.causal_step(causal_step),.save_window(save_window),.window_base(window_base),
        .box_read(box_r),.box_read_ready(rd_ready),.addr_a(ba),.addr_b(bb),.addr_c(bc),
        .data_a(core_a),.data_b(core_b),.data_c(core_c),.box_write(box_w),.box_write_ready(wr_ready),
        .addr_y(by),.data_y(box_y),.box_mask(box_mask),.box_read_mask(box_read_mask),
        .red_read(red_r),.red_read_ready(phase==RUN && rd_ready),.red_rrow(red_rrow),.red_rplane(red_rplane),.red_col(red_col),
        .red_x(red_x),.red_write(red_w),.red_write_ready(phase==RUN && wr_ready),.red_wrow(red_wrow),.red_wplane(red_wplane),
        .red_result(red_y),.red_rmask(red_rmask),.red_wmask(red_wmask),.red_bus(red_bus),.red_lg(red_lg),.red_pp(red_pp),
        .dn_extra_heads(dn_extra_heads),.dn_slot(dn_slot),.dn_read_dst(dn_read_dst),.dn_wslot(dn_wslot),.dn_wgroup(dn_wgroup),.commit_spec(1'b1),.replay_count({15'd0,from_log}),.token_count({15'd0,!from_log}),
        .k(k),.q(q),.v(v),.beta(beta),.decay(decay),.row_group(group_idx),.state_read(state_r),
        .state_read_ready(rd_ready),.state_in(read0),.state_write(state_w),.state_write_ready(wr_ready),
        .state_out(state_out),.head_out(head_y),.head_valid(head_valid),
        .tile_groups(n[10:5]),.old_max(old_max),.old_sum(old_sum),.tile_index(tile_idx),
        .tile_read(tile_r),.tile_read_ready(phase==RUN && rd_ready),.scores(read0[AUXW*32-1:0]),.k_scale(read1),.v_scale(read2),
        .tile_write(tile_w),.tile_write_ready(phase==RUN && wr_ready),.tile_wbeat(tile_wbeat),.tile_wmask(tile_wmask),
        .tile_codes(tile_q),.tile_exp(tile_exp),
        .new_max(mn),.new_sum(ln),.alpha(alpha),
        .tile_overlap(fast_tile),.tile_v_half(tile_v_half),.tile_read_state(tile_read_state),
        .tile_write_state(tile_write_state),.tile_read_row(tile_read_row),.tile_write_row(tile_write_row),
        .has_b(has_b),.has_c(has_c),.exp_bias(exp_bias),
        .aux_col(aux_col),.aux_rrow(aux_rrow),.aux_rplane(aux_rplane),.aux_wrow(aux_wrow),.aux_wplane(aux_wplane),.aux_read(aux_r),.aux_read_ready(phase==RUN && rd_ready),.aux_a(read0[AUXW*32-1:0]),
        .aux_b(read1),.aux_c(read2),.aux_write(aux_w),.aux_write_ready(phase==RUN && wr_ready),
        .aux_mask(aux_mask),.aux_wcol(aux_wcol),.aux_data(aux_y),.aux_e(aux_exp),.aux_scale(aux_scale),
        .wide_rows(wide_ok),.wide_a(read0),.wide_b(read1w),.wide_on(wide_on),.wide_mask(wide_wmask),.wide_data(wide_y),.wide_e(wide_exp),
        .wide_rmask(wide_rmask),.wide_lg(wide_lg),.wide_pp(wide_pp));

    wire [22:0] stride_m3={{3{sm[3][19]}},sm[3]}, stride_p3={{3{sp[3][19]}},sp[3]};
    wire [22:0] stride_n3={{3{sn[3][19]}},sn[3]}, stride_m0={{3{sm[0][19]}},sm[0]};
    wire [22:0] stride_p0={{3{sp[0][19]}},sp[0]};
    assign ready=phase==IDLE;
    always @* begin
        rd_valid=phase==DN_FETCH || phase==TILE_LOAD ||
                 phase==RUN && (box_r || state_r || tile_r || aux_r || (is_red && red_r));
        wr_valid=phase==EXP_WRITE || phase==ROW_RESULT || phase==HEAD_RESULT ||
                 phase==RUN && (box_w || state_w || tile_w || aux_w || (is_red && red_w));
    end
    for(j=0;j<LANES;j=j+1) begin : address_lane
        wire [15:0] rslot=j>>alg;
        wire [15:0] col=apk ? (j & (n-16'd1)) : inner_col+j;
        wire [15:0] lrrow=apk && !app ? rrow+rslot : rrow, lrplane=apk && app ? rplane+rslot : rplane;
        wire [22:0] col_addr[0:3];
        vec_box_addr op0(base[0],sn[0],sm[0],sp[0],col,row_idx,plane_idx,
                         1'b0,col_addr[0]);
        vec_box_addr op1(base[1],sn[1],sm[1],sp[1],col,is_tile ? tile_addr_read_row : lrrow,lrplane,
                         grouped[1],col_addr[1]);
        vec_box_addr op2(base[2],sn[2],sm[2],sp[2],col,lrrow,lrplane,
                         grouped[2] || op==18,col_addr[2]);
        vec_box_addr op3(base[3],sn[3],sm[3],sp[3],col,rrow,rplane,
                         grouped[3],col_addr[3]);
        // Four consecutive state rows, across all columns.
        // Reads (S for S.k, then d*S for the update) and writes (d*S, then the new S) address different heads
        // and row groups in the same clock. The update reads d*S back from the destination bank.
        wire [22:0] state_offset=(dn_slot*DIM+4*group_idx+j/DIM)*stride_m3+(j%DIM)*4;
        wire [22:0] state_woffset=(dn_wslot*DIM+4*dn_wgroup+j/DIM)*stride_m3+(j%DIM)*4;
        wire [22:0] state_src=((dn_read_dst ? dn_work_write : dn_work_read) ? work_base : base[3])+state_offset;
        wire [22:0] state_dst=(dn_work_write ? work_base : base[3])+state_woffset;
        wire [22:0] stat_read_base=base[3]+tile_addr_read_row*stride_m3+plane_idx*stride_p3;
        wire [22:0] stat_write_base=base[3]+tile_addr_write_row*stride_m3+plane_idx*stride_p3;
        wire [22:0] qkv_addr=token_base+(j==DIM ? qkv_width+value_head : fetch_key*DIM+j)*4;
        wire [22:0] key_addr=token_base+(j==DIM ? qkv_width+value_heads+value_head : (key_heads+fetch_key)*DIM+j)*4;
        wire [22:0] val_addr=token_base+((2*key_heads+value_head)*DIM+j)*4;
        assign rd_addr[0][j]=phase==DN_FETCH ? qkv_addr : tile_state_read ? stat_read_base :
                              is_box ? ba[j] : is_dn ? state_src : col_addr[1];
        assign rd_addr[1][j]=phase==DN_FETCH ? key_addr : tile_state_read ? stat_read_base+5*mlb :
                              is_box ? bb[j] : col_addr[2];
        assign rd_addr[2][j]=phase==DN_FETCH ? val_addr : is_box ? bc[j] :
                              is_tile ? col_addr[2]+vs_offset : col_addr[3];
        wire main_lane=({1'b0,inner_col}+j<n);
        wire aux_lane=wide_on ? wide_rmask[j] : j<AUXW && main_lane;
        assign rd_mask[0][j]=phase==DN_FETCH ? j<=DIM : tile_state_read ? j==0 :
                             is_box ? box_read_mask[j] : is_dn ? 1'b1 : is_red ? red_rmask[j] : aux_lane;
        assign rd_mask[1][j]=phase==DN_FETCH ? j<=DIM : tile_state_read ? j==0 :
                             is_box ? box_read_mask[j] && ((has_b && op!=6 && op!=20) || op==19 || op==21) :
                             aux_lane && (is_tile || (is_aux && (has_b || op==18 || op==23)));
        assign rd_mask[2][j]=phase==DN_FETCH ? j<DIM : tile_state_read ? 0 :
                             is_box ? box_read_mask[j] && op==5 && has_c : aux_lane && ((is_tile && (!fast_tile || tile_read_row==0)) || op>=7 && op<=12 && has_c);
        assign rd_type[0][j]=is_dn || tile_state_read ? F32 : dtype[1];
        assign rd_type[1][j]=is_dn || tile_state_read ? F32 : dtype[2];
        assign rd_type[2][j]=is_dn ? F32 : is_tile ? dtype[2] : dtype[3];
        wire scalar_aux=op==15 || op==23;
        wire aux_side=op==16 || op==17;
        // row-op writes trail their reads: address them from the write column
        wire [22:0] aux_dst;
        // packed QUANT/DEQ: write element e (code 4j for QUANT's packed words, else j) is column e % n of slot e / n
        wire [15:0] we=wq ? 16'(4*j) : 16'(j), wslot=we>>wide_lg;
        wire [15:0] write_col=wpk ? (we & (n-16'd1)) : wcol+we;
        wire [15:0] lwrow=wpk && !wide_pp ? wrow+wslot : wrow, lwplane=wpk && wide_pp ? wplane+wslot : wplane;
        vec_box_addr opw(base[0],sn[0],sm[0],sp[0],write_col,is_tile ? tile_addr_write_row : lwrow,lwplane,1'b0,aux_dst);
        // exponent lane 128 + g: group g of the beat, element 32 g
        wire [15:0] ge=16'(32*(j-AUXW)), gslot=op==17 ? 16'(j-AUXW) : ge>>wide_lg;   // QKV: lane 128 + s is slot s
        wire [15:0] grow=wpk && !wide_pp ? wrow+gslot : wrow, gplane=wpk && wide_pp ? wplane+gslot : wplane;
        wire [22:0] aux_side_base=base[3]+grow*stride_m3+gplane*stride_p3;
        // sideband lanes: QUANT's and the tile's group exponents, QKV's row scale
        wire side_lane=(is_aux || is_tile) && j>=AUXW && j<AUXW+(wq ? LANES/32 : AG) && (op==16 || is_tile || (op==17 && (wide_on || j==AUXW)));
        wire [15:0] side_g=j-AUXW;
        wire side_on=wide_on ? wide_wmask[j] : op==17 || {1'b0,wcol}+32*side_g<n;
        wire [22:0] side_dst=is_tile ? base[0]+e_offset+tile_addr_write_row*e_stride+(wcol>>5)+side_g :
                             aux_side_base+(op==16 ? (wpk ? (ge & (n-16'd1))>>5 : (wcol>>5)+side_g)*stride_n3 : 0);
        wire [31:0] side_word=op==17 ? (wide_on ? wide_y[32*j+:32] : {16'd0,aux_scale}) :
                              {22'd0,is_tile ? tile_exp[10*(side_g%AG)+:10] : wq ? wide_exp[10*(side_g%(LANES/32))+:10] : aux_exp[10*(side_g%AG)+:10]};
        wire [22:0] side_addr=is_tile ? base[0]+e_offset+row_idx*e_stride+(last_aux_col>>5) :
                               base[3]+row_idx*stride_m3+plane_idx*stride_p3+
                               (op==16 ? (last_aux_col>>5)*stride_n3 : 0);
        assign wr_mask[j]=phase==HEAD_RESULT ? j<DIM : phase==EXP_WRITE ? j==0 :
                          tile_state_write ? j<3 : phase==ROW_RESULT ? j==0 : is_red ? red_wmask[j] :
                          is_dn ? 1 : is_box ? box_mask[j] : side_lane ? side_on :
                          is_tile ? j<AUXW && tile_wmask[j%AUXW] : scalar_aux ? j==0 && aux_mask[0] : wide_on ? wide_wmask[j] : j<AUXW && aux_mask[j%AUXW];
        assign wr_addr[j]=phase==HEAD_RESULT ? base[0]+token*stride_p0+(value_head*DIM+j)*4 :
                          phase==EXP_WRITE ? side_addr : tile_state_write ?
                          stat_write_base+(j==0 ? 0 : j==1 ? 4*mlb : 5*mlb) :
                          is_dn ? state_dst : is_box ? by[j] :
                          is_red ? base[0]+(rpk && !red_pp ? wrow+j : wrow)*stride_m0+(rpk && red_pp ? wplane+j : wplane)*stride_p0 :
                          scalar_aux ? base[0]+wrow*stride_m0+wplane*stride_p0 :
                          (is_aux || is_tile) ? (side_lane ? side_dst : aux_dst) : col_addr[0];
        wire [31:0] result_word=phase==HEAD_RESULT ? head_buf[32*(transfer_slot*DIM+j%DIM)+:32] :
                          phase==EXP_WRITE ? (op==17 ? {16'd0,saved_scale} : {22'd0,saved_exp}) :
                          tile_state_write ? (j==0 ? mn : j==1 ? alpha : ln) : phase==ROW_RESULT ? scalar_y : is_red ? red_bus[32*j+:32] :
                          is_box ? box_y[j] : is_dn ? state_out[j*32+:32] :
                          side_lane ? side_word : is_tile ? {24'd0,tile_q[8*(j%AUXW)+:8]} : wide_on ? wide_y[32*j+:32] : aux_y[32*(j%AUXW)+:32];
        assign wr_type[j]=phase==EXP_WRITE ? (op==17 ? F16 : E8) : side_lane ? (op==17 ? F16 : E8) :
                          phase==ROW_RESULT || tile_state_write || is_dn || is_red || (is_box && op==21) ? F32 :
                          wq && j<LANES/4 ? U32 : dtype[0];
        vec_pack encode_write(result_word,wr_type[j],wr_raw[j],wr_data[j]);
        assign wr_raw[j]=phase==EXP_WRITE || is_tile && !tile_state_write || is_aux && (aux_side || op==23);
    end

    task next_row;
        begin
            col_idx<=0;
            if(row_idx==m-1'b1 && plane_idx==p-1'b1) begin phase<=IDLE;done<=1;end
            else begin
                if(row_idx==m-1'b1) begin row_idx<=0;plane_idx<=plane_idx+1'b1;end
                else row_idx<=row_idx+1'b1;
                phase<=is_tile ? TILE_LOAD : LAUNCH;
            end
        end
    endtask
    task finish_tile;
        begin
            if(SRAM_WRITES_IMMEDIATE) begin phase<=IDLE;done<=1;end
            else phase<=TILE_DRAIN;
        end
    endtask
    task next_token;
        begin
            transfer_slot<=0;fetch_key<=key_head;
            if(token_advances) begin token<=token+1'b1;phase<=DN_FETCH;end
            else if(from_log && token_count!=0) begin token<=0;from_log<=0;phase<=DN_FETCH;end
            else begin phase<=IDLE;done<=1;end
        end
    endtask
    always @(posedge clk) begin
        done<=0;fault<=0;
        if(reset) begin phase<=IDLE;run_finished<=0;end
        else begin
            if(exec_done) run_finished<=1;
            case(phase)
                IDLE: if(start) begin
                    row_idx<=0;plane_idx<=0;col_idx<=0;token<=0;transfer_slot<=0;fetch_key<=key_head;from_log<=replay_count!=0;run_finished<=0;
                    if(is_dn) begin
                        if(({1'b0,head}+dn_extra_heads)>=value_heads || key_heads==0 ||
                           (dn_extra_heads!=0 && !commit_spec)) fault<=1;
                        else if(token_count==0 && replay_count==0) done<=1;
                        else phase<=DN_FETCH;
                    end else if(is_tile && (n==0 || n>1024 || n[4:0]!=0)) fault<=1;
                    else if(n==0 || m==0 || p==0) done<=1;
                    else phase<=is_tile && !fast_tile ? TILE_LOAD : LAUNCH;
                end
                DN_FETCH: if(rd_ready) begin
                    k[transfer_slot*DIM*32+:DIM*32]<=read_vector_key();
                    q[transfer_slot*DIM*32+:DIM*32]<=read0[32*DIM-1:0];
                    v[transfer_slot*DIM*32+:DIM*32]<=read_vector_value();
                    beta[32*transfer_slot+:32]<=core_a[DIM];
                    decay[32*transfer_slot+:32]<=core_b[DIM];
                    if(transfer_slot==dn_extra_heads) begin transfer_slot<=0;phase<=LAUNCH;end
                    else begin
                        transfer_slot<=transfer_slot+1'b1;
                        if(next_value_scaled==next_key_scaled) fetch_key<=fetch_key+1'b1;
                    end
                end
                TILE_LOAD: if(rd_ready) begin old_max<=core_a[0];old_sum<=core_b[0];phase<=LAUNCH;end
                LAUNCH: if(exec_ready) begin
                    run_finished<=0;
                    if(exec_fault) begin phase<=IDLE;fault<=1;end else phase<=RUN;
                end
                RUN: begin
                    if(head_valid) head_buf<=head_y;
                    if(1'b0) begin                          // (sideband exponents: no separate write cycle)
                        last_aux_col<=inner_col;saved_exp<=is_tile ? tile_exp : aux_exp;saved_scale<=aux_scale;
                        phase<=EXP_WRITE;
                    end else if(exec_done || run_finished) begin
                        run_finished<=0;
                        if(is_dn && !from_log) begin head_buf<=head_y;phase<=HEAD_RESULT;end
                        else if(is_dn) next_token;
                        else if(fast_tile) finish_tile;
                        else if(is_tile) begin scalar_y<=red_y;phase<=ROW_RESULT;end
                        else if(is_aux && !aux_stream) next_row;
                        else begin done<=1;phase<=IDLE;end
                    end
                end
                EXP_WRITE: if(wr_ready) phase<=RUN;
                ROW_RESULT: if(wr_ready) begin
                    if(is_tile && row_idx==m-1'b1 && plane_idx==p-1'b1) finish_tile;
                    else next_row;
                end
                TILE_DRAIN: if(wr_visible) begin phase<=IDLE;done<=1;end
                HEAD_RESULT: if(wr_ready) begin
                    if(transfer_slot==dn_extra_heads) next_token;
                    else transfer_slot<=transfer_slot+1'b1;
                end
            endcase
        end
    end
    function automatic [DIM*32-1:0] read_vector_key;
        for(integer i=0;i<DIM;i=i+1) read_vector_key[32*i+:32]=core_b[i];
    endfunction
    function automatic [DIM*32-1:0] read_vector_value;
        for(integer i=0;i<DIM;i=i+1) read_vector_value[32*i+:32]=core_c[i];
    endfunction
endmodule
