module tb_vec_unit #(parameter BUFFERED=0);
    localparam LANES=512;
    reg clk=0,reset=1,start=0;
    wire ready,done,fault,rd_valid,wr_valid;
    reg [4:0] op;
    reg [15:0] n,m,p,rot,head,key_head,key_heads,value_heads,replay_count,token_count;
    reg [22:0] base[0:3],work_base,vs_offset,e_offset,e_stride,mlb;
    reg signed [19:0] sn[0:3],sm[0:3],sp[0:3];
    reg [2:0] dtype[0:3];
    reg [1:0] dn_extra_heads;
    reg has_b,has_c,negate,commit_spec;
    reg signed [9:0] exp_bias;
    reg [31:0] position,limit,causal_step;
    wire [22:0] rd_addr[0:2][0:LANES-1],wr_addr[0:LANES-1];
    wire [2:0] rd_type[0:2][0:LANES-1],wr_type[0:LANES-1];
    wire [LANES-1:0] rd_mask[0:2],wr_mask,wr_raw;
    reg [31:0] rd_data[0:2][0:LANES-1];
    wire [31:0] wr_data[0:LANES-1];
    reg [7:0] mem[0:524287],gold[0:524287];
    integer started_at;
    integer cycle=0,j,s,l,bytes,count,fd,ef,cases,used,cs,rc,changes,addr,val;
    reg [31:0] stalls=32'hf193b42d;
    integer tile_controls=0, profile_ready=0, profile_fast=0;
    integer reads,writes,visible_at=0;
    wire rd_ready=tile_controls ? profile_ready || stalls[0] || stalls[3] : cycle%7!=1;
    wire wr_ready=tile_controls ? profile_ready || stalls[9] || stalls[14] : cycle%5!=2;
    reg save_window;reg [22:0] window_base;
    wire wr_visible=!BUFFERED || cycle>=visible_at;
    vec_unit #(.SRAM_WRITES_IMMEDIATE(!BUFFERED)) dut(.*);
    function integer size(input [2:0] t);
        size=t==1 ? 2 : t==2 || t==4 ? 1 : 4;
    endfunction
    task tick;
        begin
            #1;
            if(rd_valid && rd_ready) reads=reads+1;
            if(rd_valid && rd_ready) for(s=0;s<3;s=s+1) for(j=0;j<LANES;j=j+1)
                if(rd_mask[s][j]) begin
                    addr=rd_addr[s][j];bytes=size(rd_type[s][j]);
                    if(addr+bytes>used) $fatal(1,"read outside image op %0d %0d",op,addr);
                    rd_data[s][j]=0;
                    for(l=0;l<bytes;l=l+1) rd_data[s][j][8*l+:8]=mem[addr+l];
                end
            #1;
            if(wr_valid && wr_ready) begin
                writes=writes+1;
                visible_at=cycle+6;
            end
            if(wr_valid && wr_ready) for(j=0;j<LANES;j=j+1)
                if(wr_mask[j]) begin
                    addr=wr_addr[j];bytes=size(wr_type[j]);
                    if(addr+bytes>used) $fatal(1,"write outside image");
                    for(l=0;l<bytes;l=l+1) mem[addr+l]=wr_data[j][8*l+:8];
                end
            #1 clk=1;#1 clk=0;cycle=cycle+1;
            stalls={stalls[30:0],stalls[31]^stalls[21]^stalls[1]^stalls[0]};
            if(BUFFERED && op==24 && done && !wr_visible) $fatal(1,"completed before SRAM visibility");
            if(fault || cycle>200000) $fatal(1,"op %0d fault/timeout",op);
        end
    endtask
    initial begin
        if(!$value$plusargs("bytes=%d",used) || !$value$plusargs("cases=%d",cases)) $fatal;
        $readmemh("start.hex",mem,0,used-1);
        for(j=0;j<used;j=j+1) gold[j]=mem[j];
        if($test$plusargs("tile_checks")) begin
            tile_controls=$fopen("controls.txt","r");
            if(tile_controls==0) $fatal(1,"missing tile controls");
        end
        fd=$fopen("cases.txt","r");ef=$fopen("expected.txt","r");tick;reset=0;
        for(cs=0;cs<cases;cs=cs+1) begin
            rc=$fscanf(fd,"%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",op,n,m,p,
              has_b,has_c,negate,exp_bias,rot,position,limit,causal_step,head,key_head,key_heads,value_heads,
              replay_count,token_count,commit_spec,work_base,vs_offset,e_offset,e_stride,mlb);
            if(rc!=24) $fatal;
            for(s=0;s<4;s=s+1) rc=$fscanf(fd,"%d%d%d%d%d",base[s],sn[s],sm[s],sp[s],dtype[s]);
            rc=$fscanf(fd,"%d%d%d",save_window,window_base,dn_extra_heads);
            if(rc!=3) $fatal;
            if(tile_controls) begin
                if($fscanf(tile_controls,"%d%d",profile_fast,profile_ready)!=2) $fatal;
                #1;
                if(dut.fast_tile !== (profile_fast!=0)) $fatal(1,"case %0d fast-path selection",cs);
                if(profile_fast) begin : shifted_alias
                    reg [22:0] destination;
                    destination=base[0];base[0]=base[1]+sm[1];#1;
                    if(dut.fast_tile!==0) $fatal(1,"shifted score alias entered fast path");
                    base[0]=destination;#1;
                end
            end
            reads=0;writes=0;
            started_at=cycle;start=1;tick;start=0;while(!done) tick;
            if(profile_fast && (reads!=m*(1+n/128) || writes!=reads))
                $fatal(1,"case %0d dropped or repeated a transfer: %0d reads, %0d writes",cs,reads,writes);
            rc=$fscanf(ef,"%d",changes);
            for(j=0;j<changes;j=j+1) begin rc=$fscanf(ef,"%d%d",addr,val);gold[addr]=val;end
            for(j=0;j<used;j=j+1)
                if(mem[j]!==gold[j])
                    $fatal(1,"case %0d op %0d addr %0d got %h expected %h",cs,op,j,mem[j],gold[j]);
            $display("PASS descriptor %0d, V opcode %0d, %0d clocks",cs,op,cycle-started_at);
            $fflush();
        end
        $finish;
    end
endmodule
