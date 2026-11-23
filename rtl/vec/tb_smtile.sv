// The streaming R7 tile against the 64 golden rows (golden/vectors/smtile_row.hex), with read and write stalls.
// Each beat carries 128 positions: 128 P codes and 4 group exponents per write.
module tb_smtile;
    localparam W=128;
    reg clk=0,reset=1,start=0;
    reg [31:0] s[0:1023],ks[0:1023],vs[0:1023],m,l;
    reg [7:0] expected_q[0:1023];
    reg [9:0] expected_e[0:31];
    reg [31:0] expected_m,expected_l,expected_alpha;
    reg [5:0] groups=32;
    reg stall=0;
    wire [2:0] rbeat, wbeat;
    wire [W*32-1:0] scores,k_scale,v_scale;
    wire rv,wv,done,busy;
    wire [W*8-1:0] codes;
    wire [(W/32)*10-1:0] e;
    wire [W-1:0] wmask;
    wire [31:0] mn,ln,alpha;
    integer cycles=0,limit,cs,i,file,rc,writes=0,row_start,total=0;
    wire rr=!stall || cycles%5!=1, wr=!stall || cycles%7!=3;
    genvar j;
    for(j=0;j<W;j=j+1) begin
        wire [9:0] pos={rbeat,7'b0}+j;
        assign scores[32*j+:32]=pos>limit ? 32'hff800000 : s[pos];
        assign k_scale[32*j+:32]=ks[pos],v_scale[32*j+:32]=vs[pos];
    end
    vec_smtile dut(.clk(clk),.reset(reset),.start(start),.groups(groups),.old_max(m),.old_sum(l),
        .read_beat(rbeat),.read_valid(rv),.read_ready(rr),.scores(scores),.k_scale(k_scale),.v_scale(v_scale),
        .write_valid(wv),.write_ready(wr),.write_beat(wbeat),.codes(codes),.exponents(e),.write_mask(wmask),
        .new_max(mn),.new_sum(ln),.alpha(alpha),.done(done),.busy(busy),.fa_bus(),.fb_bus(),.fc_bus(),
        .bank_result({(512*32){1'b0}}),.bank_ce());
    task tick;
        begin
            #1;
            if(wv && wr) begin
                for(i=0;i<W;i=i+1) begin
                    if(wmask[i] !== (W*wbeat+i < 32*groups)) $fatal(1,"row %0d write mask",cs);
                    if(wmask[i] && codes[8*i+:8]!==expected_q[W*wbeat+i])
                        $fatal(1,"row %0d beat %0d lane %0d: P %h want %h",cs,wbeat,i,codes[8*i+:8],expected_q[W*wbeat+i]);
                end
                for(i=0;i<W/32;i=i+1) if(4*wbeat+i < groups && e[10*i+:10]!==expected_e[4*wbeat+i])
                    $fatal(1,"row %0d beat %0d group %0d: exponent",cs,wbeat,i);
                writes=writes+1;
            end
            #1 clk=1;#1 clk=0;cycles=cycles+1;
        end
    endtask
    reg [31:0] value;
    reg [8191:0] packed_q;
    reg [383:0] packed_e;
    reg [7:0] bar;
    task word; if($fscanf(file,"%h",value)!=1) $fatal(1,"bad vector"); endtask
    task sep; if($fscanf(file," %c",bar)!=1 || bar!="|") $fatal(1,"separator"); endtask
    task check_row;
        begin
            writes=0;row_start=cycles;start=1;tick;start=0;
            while(!done) begin tick; if(cycles-row_start>20000) $fatal(1,"row %0d timeout",cs); end
            if(writes!=(groups+3)/4 || {mn,ln,alpha}!=={expected_m,expected_l,expected_alpha})
                $fatal(1,"R7 row %0d, writes %0d, state %h %h %h expected %h %h %h",cs,writes,mn,ln,alpha,expected_m,expected_l,expected_alpha);
        end
    endtask
    task time_row(input integer n);
        begin
            groups=n/32;
            limit=1023;
            m=0;l=32'h3f800000;
            // Zero valid scores with unit scales give P=1, m=0, alpha=1 and l=n+1.
            // Supply stale positive scores for inactive lanes of the final beat.
            for(i=0;i<1024;i=i+1) begin
                s[i]=i<n ? 32'd0 : 32'h3f800000;
                ks[i]=32'h3f800000;vs[i]=32'h3f800000;expected_q[i]=8'd64;
            end
            for(i=0;i<32;i=i+1) expected_e[i]=-10'sd6;
            expected_m=0;expected_alpha=32'h3f800000;
            case(n)
                64: expected_l=32'h42820000;
                256: expected_l=32'h43808000;
                1024: expected_l=32'h44802000;
                default: $fatal(1,"no expected sum for this width");
            endcase
            cs=-1;
            check_row;
            $display("PASS R7 softmax tile, %0d clocks for %0d positions without stalls",cycles-row_start,n);
            $fflush();
        end
    endtask
    initial begin
        tick;reset=0;
        time_row(64);
        time_row(256);
        time_row(1024);
        stall=1;
        file=$fopen("../../golden/vectors/smtile_row.hex","r");
        for(cs=0;cs<64;cs=cs+1) begin
            for(i=0;i<1024;i=i+1) begin word;s[i]=value;end
            for(i=0;i<1024;i=i+1) begin word;ks[i]=value;end
            for(i=0;i<1024;i=i+1) begin word;vs[i]=value;end
            word;m=value;word;l=value;sep;
            rc=$fscanf(file,"%d",limit);sep;
            rc=$fscanf(file,"%h %h",packed_q,packed_e);if(rc!=2) $fatal;
            for(i=0;i<1024;i=i+1) expected_q[i]=packed_q[8*(1023-i)+:8];
            for(i=0;i<32;i=i+1) expected_e[i]=packed_e[12*(31-i)+:10];
            sep;word;expected_m=value;word;expected_l=value;word;expected_alpha=value;
            check_row;
            total=total+cycles-row_start;
        end
        $display("PASS 64 R7 softmax tiles, %0d clocks per 1,024-position row with stalls", total/64);
        $finish;
    end
endmodule
