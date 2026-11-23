module tb_dn_head;
    localparam D=128;
    reg clk=0,reset=1,start=0,commit;
    reg [15:0] nrep,ntok;
    reg [D*32-1:0] k,q,v;
    reg [31:0] beta,g;
    wire [31:0] decay;
    sfu_exp exp_decay(g,decay);
    reg [D*4*32-1:0] state_mem[0:63],expected[0:31],original[0:31];
    wire [D*4*32-1:0] state_y;
    wire [4:0] row, wrow;
    wire rwork,wwork,rv,wv,ov,done,busy,replay;
    wire [15:0] idx,out_index;
    wire [D*32-1:0] out_row;
    reg [D*32-1:0] expected_row;
    integer cycle=0,file,cs,i,outputs,writes;
    wire rr=cycle%7!=1,wr=cycle%5!=2;
    vec_dn_head dut(.clk(clk),.reset(reset),.start(start),.commit_spec(commit),.replay_count(nrep),.token_count(ntok),
        .replay(replay),.token_index(idx),.k(k),.q(q),.v(v),.beta(beta),.decay(decay),
        .rd_group(row),.wr_group(wrow),.read_work(rwork),.write_work(wwork),.read_valid(rv),.read_ready(rr),
        .state_in(state_mem[{rwork,row}]),.write_valid(wv),.write_ready(wr),.state_out(state_y),
        .head_out(out_row),.out_valid(ov),.out_index(out_index),.done(done),.busy(busy));
    task tick;
        begin
            if(wv && wr) begin state_mem[{wwork,wrow}]=state_y;writes=writes+1;end
            #1 clk=1;#1 clk=0;cycle=cycle+1;
            if(ov) begin
                if(out_row!==expected_row || out_index!=0) $fatal(1,"R4 output case %0d",cs);
                outputs=outputs+1;
            end
        end
    endtask
    reg [31:0] value;
    reg [7:0] bar;
    task word; if($fscanf(file,"%h",value)!=1) $fatal(1,"bad vector");endtask
    task sep; if($fscanf(file," %c",bar)!=1 || bar!="|") $fatal(1,"separator");endtask
    initial begin
        tick;reset=0;file=$fopen("../../golden/vectors/dn_token.hex","r");
        for(cs=0;cs<8;cs=cs+1) begin
            for(i=0;i<D*D;i=i+1) begin word;original[i/512][32*(i%512)+:32]=value;end
            for(i=0;i<32;i=i+1) begin state_mem[i]=original[i];state_mem[32+i]=0;end
            sep;
            for(i=0;i<D;i=i+1) begin word;k[32*i+:32]=value;end
            for(i=0;i<D;i=i+1) begin word;q[32*i+:32]=value;end
            for(i=0;i<D;i=i+1) begin word;v[32*i+:32]=value;end
            word;beta=value;word;g=value;sep;
            for(i=0;i<D*D;i=i+1) begin word;expected[i/512][32*(i%512)+:32]=value;end
            sep;
            for(i=0;i<D;i=i+1) begin word;expected_row[32*i+:32]=value;end
            commit=cs%3==0;nrep=cs%3==2;ntok=nrep==0;writes=0;outputs=0;
            start=1;tick;start=0;while(!done) tick;
            if(writes!=64 || outputs!=ntok) $fatal(1,"completion: %0d writes (32 d*S + 32 S), %0d outputs",writes,outputs);
            for(i=0;i<32;i=i+1) begin
                if(state_mem[(commit || nrep ? 0 : 32)+i]!==expected[i]) $fatal(1,"R4 state case %0d rowgroup %0d",cs,i);
                if(!commit && !nrep && state_mem[i]!==original[i]) $fatal(1,"spec commit leak");
            end
        end
        $display("PASS 8 R4 single-head vectors (commit, speculative copy and replay), last case %0d clocks",cycle);
    end
endmodule
