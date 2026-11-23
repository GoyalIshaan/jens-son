module tb_reduce_pipe;
    localparam N=512;
    reg clk=0,reset=1,en=0,first=0,last=0,square;
    reg [N*32-1:0] x;
    wire ready,valid;
    wire [31:0] y;
    vec_reduce_pipe dut(clk,reset,en,first,last,square,x,ready,valid,y,,,,);
    task tick; begin #1 clk=1; #1 clk=0; end endtask
    reg [31:0] words[0:40000],want;
    integer pos=0,cs,beat,count,lane;
    initial begin
        $readmemh("vec/reduce.hex",words);
        tick; reset=0;
        for (cs=0;cs<10;cs=cs+1) begin
            count=words[pos]; square=words[pos+1][0]; want=words[pos+2]; pos=pos+3;
            for(beat=0;beat<count;beat=beat+1) begin
                while(!ready) tick;
                for(lane=0;lane<N;lane=lane+1) x[32*lane+:32]=words[pos+lane];
                pos=pos+N; en=1; first=beat==0; last=beat==count-1; tick; en=0;
            end
            while(!valid) tick;
            if(y!==want) $fatal(1,"case %0d got %h want %h",cs,y,want);
        end
        $display("PASS shared 512-lane reduction (10 rows)");
    end
endmodule
