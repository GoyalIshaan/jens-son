// Execution cluster behind the V command queue. The request and its decoded
// descriptor stay stable through done. Box commands expose byte addresses.
// R4 uses a head-state bank port, R7 a row port, and R1 a masked beat stream.
// An SRAM frontend presents those ports with converted operands. Reduction
// commands cover every row and plane. SMTILE can cover an entire broadcast-scale descriptor.
module vector_exec #(parameter LANES=512, parameter DIM=LANES/4, parameter AUXW=128)(
    input wire clk,reset,start,input wire [4:0] opcode,
    output wire ready,done,fault,
    input wire [15:0] cols,rows,planes,rot,
    input wire [22:0] base[0:3], input wire signed [19:0] sn[0:3],sm[0:3],sp[0:3],
    input wire [3:0] grouped,input wire negate,
    input wire [31:0] position,limit,causal_step,
    input wire save_window,input wire [22:0] window_base,
    output wire box_read, input wire box_read_ready,
    output wire [22:0] addr_a[0:LANES-1],addr_b[0:LANES-1],addr_c[0:LANES-1],
    input wire [31:0] data_a[0:LANES-1],data_b[0:LANES-1],data_c[0:LANES-1],
    output wire box_write,input wire box_write_ready,
    output wire [22:0] addr_y[0:LANES-1],output wire [31:0] data_y[0:LANES-1],
    output wire [LANES-1:0] box_mask, box_read_mask,
    // R1 walks all rows and planes of the command, several rows in flight
    output wire red_read,input wire red_read_ready,output wire [15:0] red_rrow,red_rplane,red_col,
    input wire [LANES*32-1:0] red_x,
    output wire red_write,input wire red_write_ready,output wire [15:0] red_wrow,red_wplane,
    output wire [31:0] red_result,
    // packed R1: per-lane read mask, one result lane per packed row, pack shape (log2 cols, along planes)
    output wire [LANES-1:0] red_rmask, red_wmask, output wire [LANES*32-1:0] red_bus,
    output wire [3:0] red_lg, output wire red_pp,
    input wire commit_spec,input wire [15:0] replay_count,token_count,
    output wire replay,output wire [15:0] token_index,
    input wire [1:0] dn_extra_heads,output wire [1:0] dn_slot,
    output wire dn_read_dst,output wire [1:0] dn_wslot,output wire [$clog2(DIM/4)-1:0] dn_wgroup,
    input wire [4*DIM*32-1:0] k,q,v,input wire [127:0] beta,decay,
    output wire [$clog2(DIM/4)-1:0] row_group,
    output wire read_work,write_work,state_read,input wire state_read_ready,
    input wire [4*DIM*32-1:0] state_in,
    output wire state_write,input wire state_write_ready,
    output wire [4*DIM*32-1:0] state_out,output wire [4*DIM*32-1:0] head_out,
    output wire head_valid,output wire [15:0] head_index,
    input wire [5:0] tile_groups,input wire [31:0] old_max,old_sum,
    output wire [2:0] tile_index,output wire tile_read,input wire tile_read_ready,
    input wire [AUXW*32-1:0] scores,k_scale,v_scale,
    output wire tile_write,input wire tile_write_ready,output wire [2:0] tile_wbeat,output wire [AUXW-1:0] tile_wmask,
    output wire [AUXW*8-1:0] tile_codes,output wire [(AUXW/32)*10-1:0] tile_exp,
    output wire [31:0] new_max,new_sum,alpha,
    input wire tile_overlap,
    input wire [AUXW*16-1:0] tile_v_half,
    output wire tile_read_state,tile_write_state,
    output wire [15:0] tile_read_row,tile_write_row,
    input wire has_b,has_c,input wire signed [9:0] exp_bias,
    output wire [15:0] aux_col,aux_rrow,aux_rplane,aux_wrow,aux_wplane,output wire aux_read,input wire aux_read_ready,
    input wire [AUXW*32-1:0] aux_a,aux_b,aux_c,
    output wire aux_write,input wire aux_write_ready,output wire [AUXW-1:0] aux_mask,output wire [15:0] aux_wcol,
    output wire [AUXW*32-1:0] aux_data,output wire [(AUXW/32)*10-1:0] aux_e,output wire [15:0] aux_scale
);
    localparam NONE=0, BOX=1, RED=2, DN=3, TILE=4, AUX=5, NOP=6;
    reg [2:0] owner;
    wire [2:0] selected = opcode==13 || opcode==14 ? RED : opcode==22 ? DN :
                          opcode==24 ? TILE : opcode==0 ? NOP :
                          (opcode>=7 && opcode<=12) || (opcode>=15 && opcode<=18) || opcode==23 ? AUX : BOX;
    assign ready=owner==NONE;
    wire accept=start && ready && !reset;
    wire box_done,box_fault,red_done,dn_done,tile_done,aux_done,red_ce;
    wire box_ce;                        // a streaming box command freezes the bank on a stall
    // R7's sum is an R1 tree over 512 lanes: the tile needs the full 512-lane bank
    wire tile_fault=accept && selected==TILE && (tile_groups==0 || tile_groups>32 || LANES<512);
    // The descriptor frontend gathers each grouped token separately and
    // checkpoints tentative groups. Only legacy single-head launches may
    // stream multiple token rows directly through this internal interface.
    wire dn_fault=accept && selected==DN && dn_extra_heads!=0 &&
                  (!commit_spec || {1'b0,replay_count}+{1'b0,token_count}>1);
    assign fault=box_fault || tile_fault || dn_fault;
    assign done=owner==BOX ? box_done : owner==RED ? red_done :
                owner==DN ? dn_done : owner==TILE ? tile_done : owner==AUX ? aux_done : owner==NOP;
    always @(posedge clk) begin
        if(reset || done) owner<=NONE;
        else if(accept && !fault) owner<=selected;
    end
    localparam W=32*LANES;
    wire [W-1:0] a_box,b_box,c_box,a_red,b_red,c_red,a_dn,b_dn,c_dn,fy;
    wire [W-1:0] a_tile,b_tile,c_tile;
    wire tile_ce;
    vec_box_command #(LANES,0) boxes(clk,reset,accept && selected==BOX,opcode,
        cols,rows,planes,base,sn,sm,sp,grouped,negate,rot,position,limit,causal_step,save_window,window_base,
        box_done,,box_fault,box_read,box_read_ready,addr_a,addr_b,addr_c,data_a,data_b,data_c,
        box_write,box_write_ready,addr_y,data_y,box_mask,a_box,b_box,c_box,fy,box_ce,box_read_mask);
    vec_reduce_rows #(LANES,4) reduce(.clk(clk),.reset(reset),.start(accept && selected==RED),.square(opcode==14),
        .cols(cols),.rows(rows),.planes(planes),.read_valid(red_read),.read_ready(red_read_ready),
        .read_row(red_rrow),.read_plane(red_rplane),.read_col(red_col),.x(red_x),
        .write_valid(red_write),.write_ready(red_write_ready),.write_row(red_wrow),.write_plane(red_wplane),
        .read_mask(red_rmask),.write_mask(red_wmask),.result(red_bus),.pack_lg(red_lg),.pack_p(red_pp),.done(red_done),.busy(),.fa_bus(a_red),.fb_bus(b_red),.fc_bus(c_red),
        .bank_result(fy),.bank_ce(red_ce));
    // the last row's sum, held after its write (the bus itself follows the bank)
    reg [31:0] red_last;
    always @(posedge clk) if (red_write && red_write_ready) red_last <= red_bus[31:0];
    assign red_result = red_write ? red_bus[31:0] : red_last;
    vec_dn_group #(DIM) dn(clk,reset,accept && selected==DN && !dn_fault,commit_spec,dn_extra_heads,replay_count,token_count,
        replay,token_index,k,q,v,beta,decay,dn_slot,row_group,dn_read_dst,read_work,write_work,state_read,state_read_ready,
        state_in,dn_wslot,dn_wgroup,state_write,state_write_ready,state_out,head_out,head_valid,head_index,dn_done,a_dn,b_dn,c_dn,fy);
    if (LANES >= 512) begin : tile_unit
        vec_smtile #(0,AUXW,LANES==512 && AUXW==128) tile(.clk(clk),.reset(reset),.start(accept && selected==TILE && !tile_fault),.groups(tile_groups),
            .old_max(old_max),.old_sum(old_sum),.read_beat(tile_index),.read_valid(tile_read),.read_ready(tile_read_ready),
            .scores(scores),.k_scale(k_scale),.v_scale(v_scale),.write_valid(tile_write),.write_ready(tile_write_ready),
            .write_beat(tile_wbeat),.codes(tile_codes),.exponents(tile_exp),.write_mask(tile_wmask),
            .new_max(new_max),.new_sum(new_sum),.alpha(alpha),.done(tile_done),.busy(),
            .overlap(tile_overlap),.overlap_rows(rows),.v_scale_half(tile_v_half),
            .read_state(tile_read_state),.write_state(tile_write_state),
            .read_row(tile_read_row),.write_row(tile_write_row),
            .fa_bus(a_tile[512*32-1:0]),.fb_bus(b_tile[512*32-1:0]),.fc_bus(c_tile[512*32-1:0]),.bank_result(fy[512*32-1:0]),.bank_ce(tile_ce));
    end else begin : no_tile
        assign {tile_index,tile_read,tile_write,tile_wbeat,tile_wmask,tile_codes,tile_exp,tile_done,tile_ce}=0;
        assign {tile_read_state,tile_write_state,tile_read_row,tile_write_row}=0;
        assign {new_max,new_sum,alpha}=0; assign {a_tile,b_tile,c_tile}=0;
    end

    wire [AUXW*32-1:0] a_aux,b_aux,c_aux;
    wire aux_ce;
    vec_row_ops #(AUXW) aux(.clk(clk),.reset(reset),.start(accept && selected==AUX),.opcode(opcode),.count(cols),.rows(rows),.planes(planes),
        .read_row(aux_rrow),.read_plane(aux_rplane),.write_row(aux_wrow),.write_plane(aux_wplane),
        .has_b(has_b),.has_c(has_c),.negate(negate),.exp_bias(exp_bias),
        .col(aux_col),.read_valid(aux_read),.read_ready(aux_read_ready),.a(aux_a),.b(aux_b),.c(aux_c),
        .write_valid(aux_write),.write_ready(aux_write_ready),.lane_mask(aux_mask),.write_col(aux_wcol),
        .data(aux_data),.group_exp(aux_e),.kv_scale(aux_scale),.done(aux_done),.busy(),
        .fa(a_aux),.fb(b_aux),.fc(c_aux),.fy(fy[AUXW*32-1:0]),.bank_ce(aux_ce));
    wire [W-1:0] a_mux=owner==RED ? a_red : owner==DN ? a_dn : owner==TILE ? a_tile : owner==AUX ? a_aux : a_box;
    wire [W-1:0] b_mux=owner==RED ? b_red : owner==DN ? b_dn : owner==TILE ? b_tile : owner==AUX ? b_aux : b_box;
    wire [W-1:0] c_mux=owner==RED ? c_red : owner==DN ? c_dn : owner==TILE ? c_tile : owner==AUX ? c_aux : c_box;
    wire bank_ce = owner==BOX ? box_ce : owner==AUX ? aux_ce : owner==TILE ? tile_ce : owner==RED ? red_ce : 1'b1;
    genvar j;
    for(j=0;j<LANES;j=j+1) begin : arithmetic
        fp32_fma_pipe8_ce madd(clk,bank_ce,a_mux[32*j+:32],b_mux[32*j+:32],c_mux[32*j+:32],fy[32*j+:32]);
    end
endmodule
