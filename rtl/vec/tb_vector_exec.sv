// Switch owners without reset, using the actual shared pipeline. DIM=16
// makes the head-state contents small enough to inspect exhaustively here.
module tb_vector_exec;
    localparam N=64,D=N/4;
    reg clk=0,reset=1,start=0;
    reg [4:0] op;
    wire ready,done,fault;
    reg [15:0] cols=2,rows=1,planes=1;
    reg [22:0] base[0:3];reg signed [19:0] sn[0:3],sm[0:3],sp[0:3];
    wire [31:0] a[0:N-1],b[0:N-1],c[0:N-1],y[0:N-1];
    wire box_write,tile_write,state_write,head_valid,red_read,red_write;
    wire [N-1:0] mask;
    wire [31:0] sum,m,l,alpha;
    wire [255:0] codes;
    wire [9:0] e;
    wire [N*32-1:0] state_y;
    wire [4*D*32-1:0] heads;
    wire [D*32-1:0] head=heads[D*32-1:0];
    wire read_work,write_work,replay;
    wire [15:0] token_index,out_index;
    reg [15:0] nrep=0,ntok=1;
    wire [$clog2(D/4)-1:0] group_idx, wgroup;
    reg [N*32-1:0] banks[0:7];
    integer i,writes=0,outputs=0,cycles=0,cold_writes=0,sigmoid_writes=0;
    genvar j;
    for(j=0;j<N;j=j+1) begin
        assign a[j]=32'h3f800000,b[j]=32'h40000000,c[j]=32'h40800000;
    end
    wire aux_write;
    wire [15:0] aux_scale,aux_col;
    wire [1023:0] aux_data;
    wire [31:0] aux_mask;
    wire [9:0] aux_e;
    reg [1023:0] aux_a,aux_b;
    reg has_b=0;
    vector_exec #(N, N/4, 32) dut(.wide_rows(1'b0),.clk(clk),.reset(reset),.start(start),.opcode(op),.ready(ready),.done(done),.fault(fault),
      .cols(cols),.rows(rows),.planes(planes),.rot(16'd64),.base(base),.sn(sn),.sm(sm),.sp(sp),.grouped(4'd0),
      .negate(1'b0),.position(0),.limit(0),.causal_step(1),.save_window(1'b0),.window_base(23'd0),
      .box_read_ready(1'b1),.data_a(a),.data_b(b),.data_c(c),.box_write(box_write),.box_write_ready(1'b1),
      .data_y(y),.box_mask(mask),.red_read(red_read),.red_read_ready(1'b1),.red_x({N{32'h40000000}}),
      .red_write(red_write),.red_write_ready(1'b1),.red_result(sum),.dn_extra_heads(2'd0),.commit_spec(1'b0),.replay_count(nrep),.token_count(ntok),
      .replay(replay),.token_index(token_index),.k({(4*D){32'h3f800000}}),.q({(4*D){32'h3f800000}}),.v({(4*D){32'h3f800000}}),
      .beta({4{32'h3f800000}}),.decay({4{32'h3f800000}}),.row_group(group_idx),.dn_wgroup(wgroup),.read_work(read_work),.write_work(write_work),
      .state_read_ready(1'b1),.state_in(banks[{read_work,group_idx}]),.state_write(state_write),.state_write_ready(1'b1),
      .state_out(state_y),.head_out(heads),.head_valid(head_valid),.head_index(out_index),.tile_groups(6'd1),
      .old_max(32'hff800000),.old_sum(0),.tile_read_ready(1'b1),.scores(1024'd0),
      .k_scale({32{32'h3f800000}}),.v_scale({32{32'h3f800000}}),.tile_write(tile_write),
      .tile_write_ready(1'b1),.tile_codes(codes),.tile_exp(e),.new_max(m),.new_sum(l),.alpha(alpha),.has_b(has_b),.has_c(1'b0),.exp_bias(10'sd0),
      .aux_a(aux_a),.aux_b(aux_b),.aux_c(1024'd0),.aux_read_ready(1'b1),.aux_write_ready(1'b1),
      .aux_write(aux_write),.aux_col(aux_col),.aux_data(aux_data),.aux_mask(aux_mask),.aux_e(aux_e),.aux_scale(aux_scale));
    task tick;begin
        if(state_write) banks[{write_work,wgroup}] = state_y;
        if(box_write) begin
            if(mask!==3 || y[0]!==32'h40400000 || y[1]!==32'h40400000) $fatal(1,"box add");
            writes=writes+1;
        end
        #1 clk=1;#1 clk=0;cycles=cycles+1;
        if(head_valid) begin
            // Starting from zero: replay creates all ones. Token 0 then
            // gives delta=-15, state=-14, output=16-240=-224. Token 1
            // gives delta=225, state=211, output=-224+3600=3376.
            if(out_index!=outputs || head!=={D{outputs==0 ? 32'hc3600000 : 32'h45530000}})
                $fatal(1,"head %0d %h",out_index,head[31:0]);
            outputs=outputs+1;
        end
        if(cycles>20000) $fatal(1,"command timeout");
    end endtask
    task issue(input [4:0] code);begin
        while(!ready) tick;
        op=code;start=1;tick;start=0;
    end endtask
    initial begin
        tick;reset=0;
        for(i=0;i<4;i=i+1) begin base[i]=0;sn[i]=4;sm[i]=0;sp[i]=0;end
        for(i=0;i<8;i=i+1) banks[i]=0;
        // The first command must retire even with an empty SFU tag pipeline.
        aux_a=0;
        cols=31;
        issue(7);
        while(!done) begin
            if(aux_write) begin
                if(aux_mask!==32'h7fffffff || aux_data[31:0]!==32'h3f800000) $fatal(1,"cold EXP");
                cold_writes=cold_writes+1;
            end
            tick;
        end
        if(cold_writes!=1) $fatal(1,"cold EXP wrote %0d times",cold_writes);
        // The same column needs two sub-beats at the sigmoid pool's width.
        issue(8);
        while(!done) begin
            if(aux_write) begin
                if(aux_mask!==32'h7fffffff || aux_data[31*32-1:0]!=={31{32'h3f000000}})
                    $fatal(1,"SIGMOID after EXP");
                sigmoid_writes=sigmoid_writes+1;
            end
            tick;
        end
        if(sigmoid_writes!=1) $fatal(1,"SIGMOID wrote %0d times",sigmoid_writes);
        cols=2;
        issue(1);while(!done) tick;if(writes!=1) $fatal;
        begin : one_row_rsumsq
            integer red_reads, red_writes; red_reads=0; red_writes=0;
            issue(14);
            while(!done) begin red_reads=red_reads+red_read; red_writes=red_writes+red_write; tick; end
            // cols = 2: the reducer masks lanes past cols itself, so the sum is 2 x 2.0^2 = 8
            if(sum!==32'h41000000 || red_reads!=1 || red_writes!=1) $fatal(1,"sum of squares %h %0d %0d",sum,red_reads,red_writes);
        end
        nrep=1;ntok=2;issue(22);while(!done) tick;
        for(i=0;i<4;i=i+1) if(banks[i]!=={N{32'h3f800000}} || banks[i+4]!=={N{32'h43530000}})
            $fatal(1,"replay vs working state");
        // (the R7 tile needs the full 512-lane bank: tb_smtile and test_vec_unit.py check it)
        if(writes!=1 || outputs!=2) $fatal(1,"after DN: writes %0d outputs %0d",writes,outputs);
        cols=37;
        aux_a={32{32'h3f800000}};aux_b={32{32'h40000000}};
        issue(17);
        while(!done) begin
            if(aux_write && (aux_data[7:0]!==127 || aux_scale!==16'h2008)) $fatal(1,"QKV %h %h",aux_scale,aux_data[7:0]);
            tick;
        end
        issue(16);while(!done) begin if(aux_write && (aux_e!==-10'sd6 || aux_data[7:0]!==64)) $fatal(1,"QUANT");tick;end
        has_b=1;issue(18);while(!done) begin if(aux_write && aux_data[31:0]!==32'h40000000) $fatal(1,"DEQ");tick;end
        has_b=0;issue(15);while(!done) begin if(aux_write && aux_data[31:0]!==32'h3f800000) $fatal(1,"RMAX");tick;end
        aux_b=aux_a;aux_b[32*7+:32]=0;
        issue(23);while(!done) begin if(aux_write && aux_data[31:0]!==7) $fatal(1,"MATCH");tick;end
        aux_a=0;issue(7);while(!done) begin if(aux_write && aux_data[31:0]!==32'h3f800000) $fatal(1,"EXP");tick;end
        $display("PASS shared execution bank, command switching, replay and two speculative tokens (no tile: 64 lanes)");
    end
endmodule
