// Cycle/handshake regression for the 128-wide R4 controller. Arithmetic is
// data-independent here; test_vec_unit.py checks the real FMA results.
module tb_dn_interleave;
    localparam D=128,W=4*D*32;
    reg clk=0,reset=1,start=0,stalls=0;
    reg [1:0] extra=0;
    reg [15:0] nrep=0,ntok=1;
    wire [1:0] slot, wslot;
    wire [4:0] row, wrow;
    wire rv,wv,done;
    integer cycle=0,writes[0:3],reads[0:3],base_cycles,group_cycles;
    reg held=0;
    reg [7:0] held_request; reg wheld=0; reg [6:0] wheld_request;
    wire rd_ready=!stalls || cycle%7>=3,wr_ready=!stalls || cycle%9>=4;
    wire [W-1:0] zeros={W{1'b0}};
    vec_dn_group #(D) dut(.clk(clk),.reset(reset),.start(start),.commit_spec(1'b1),
        .extra_heads(extra),.replay_count(nrep),.token_count(ntok),.k(zeros),.q(zeros),.v(zeros),
        .beta(128'd0),.decay(128'd0),.slot(slot),.row_group(row),.wslot(wslot),.wrow_group(wrow),
        .state_read(rv),.read_ready(rd_ready),.state_in(zeros),.state_write(wv),.write_ready(wr_ready),
        .done(done),.fy(zeros));
    task tick;
        begin
            #1;
            if(held && (!rv || {1'b1,slot,row}!==held_request)) $fatal(1,"stalled read changed");
            if(wheld && (!wv || {wslot,wrow}!==wheld_request)) $fatal(1,"stalled write changed");
            held=rv && !rd_ready; held_request={1'b1,slot,row};
            wheld=wv && !wr_ready; wheld_request={wslot,wrow};
            if(rv && rd_ready) reads[slot]=reads[slot]+1;
            if(wv && wr_ready) writes[wslot]=writes[wslot]+1;
            #1 clk=1;#1 clk=0;cycle=cycle+1;
        end
    endtask
    task run(input integer heads,input bit replay_only,output integer elapsed);
        integer h,arrival,accepted;
        reg [3:0] early_issues;
        begin
            extra=heads-1;nrep=replay_only;ntok=!replay_only;early_issues=0;accepted=0;
            for(h=0;h<4;h=h+1) begin reads[h]=0;writes[h]=0;end
            arrival=cycle;start=1;tick;start=0;
            while(!done) begin
                #1;
                if(accepted<heads && dut.issue_ok) begin
                    early_issues[slot]=1;accepted=accepted+1;
                end
                tick;
                if(cycle-arrival>3000) $fatal(1,"group timeout");
            end
            elapsed=cycle-arrival;
            if(early_issues!==((1<<heads)-1)) $fatal(1,"heads did not overlap");
            for(h=0;h<4;h=h+1) begin
                if(writes[h]!=(h<heads ? 64 : 0) || reads[h]!=(h<heads ? 64 : 0))
                    $fatal(1,"head %0d: reads %0d writes %0d",h,reads[h],writes[h]);
            end
            tick;
        end
    endtask
    integer replay_mode,count;
    initial begin
        tick;reset=0;
        for(replay_mode=0;replay_mode<2;replay_mode=replay_mode+1) begin
            stalls=0;
            run(1,replay_mode,base_cycles);
            run(4,replay_mode,group_cycles);
            if(group_cycles*2>=base_cycles*4) $fatal(1,"less than 2x speedup");
            $display("R4 replay=%0d: four serial launches %0d clocks, four interleaved %0d",
                     replay_mode,4*base_cycles,group_cycles);
            stalls=1;
            for(count=1;count<=4;count=count+1) run(count,replay_mode,group_cycles);
        end
        $display("PASS R4 interleaving, tail groups, and SRAM backpressure");
        $finish;
    end
endmodule
