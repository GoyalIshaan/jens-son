// SRAM reads and writes are each stalled independently. Use in-place RoPE.
module tb_box_command;
    localparam L=64;
    reg clk=0,reset=1,start=0;
    reg [4:0] op;
    reg [15:0] n,m,p,rot=64;
    reg [22:0] base[0:3];
    reg signed [19:0] sn[0:3],sm[0:3],sp[0:3];
    reg [3:0] grouped=0;
    reg [31:0] position=0,limit=0,step=1;
    reg window=0;
    reg [22:0] win=16384;
    wire done,busy,fault,rv,wv;
    integer cycles=0;
    wire rr=cycles%5!=1, wr=cycles%7!=2;
    wire [22:0] aa[0:L-1],ba[0:L-1],ca[0:L-1],ya[0:L-1];
    wire [31:0] av[0:L-1],bv[0:L-1],cv[0:L-1],yv[0:L-1];
    wire [L-1:0] mask, read_mask;
    reg [31:0] mem[0:8191];
    reg [31:0] expected[0:1023];
    vec_box_command #(L) dut(clk,reset,start,op,n,m,p,base,sn,sm,sp,grouped,1'b0,
        rot,position,limit,step,window,win,done,busy,fault,rv,rr,aa,ba,ca,av,bv,cv,wv,wr,ya,yv,mask,,,,,,read_mask);
    genvar j;
    for(j=0;j<L;j=j+1) begin
        assign av[j]=read_mask[j] ? mem[aa[j]>>2] : 32'hxxxxxxxx;
        assign bv[j]=read_mask[j] ? mem[ba[j]>>2] : 32'hxxxxxxxx;
        assign cv[j]=read_mask[j] ? mem[ca[j]>>2] : 32'hxxxxxxxx;
        always @(posedge clk) if(wv && wr && mask[j]) mem[ya[j]>>2]<=yv[j];
    end
    task tick; begin #1 clk=1; #1 clk=0; cycles=cycles+1; end endtask
    task run; begin
        start=1; tick; start=0;
        while(!done) begin tick; if(cycles>200000) $fatal(1,"timeout"); end
    end endtask
    integer f,cs,i,h,r,got;
    reg [31:0] value;
    reg [7:0] bar;
    task word(output reg [31:0] v);
        if($fscanf(f,"%h",v)!=1) $fatal(1,"bad hex");
    endtask
    task sep;
        if($fscanf(f," %c",bar)!=1 || bar!="|") $fatal(1,"missing separator");
    endtask
    initial begin
        tick;reset=0;
        for(i=0;i<4;i=i+1) begin base[i]=2048*i;sn[i]=4;sm[i]=0;sp[i]=256; end
        f=$fopen("../../golden/vectors/conv4.hex","r");
        op=21;n=64;m=1;p=8;sn[2]=16;window=1;
        for(cs=0;cs<16;cs=cs+1) begin
            for(i=0;i<192;i=i+1) begin word(value);mem[(base[3]>>2)+i]=value; end
            sep;
            for(i=0;i<512;i=i+1) begin word(value);mem[(base[1]>>2)+i]=value; end
            sep;
            for(i=0;i<256;i=i+1) begin word(value);mem[(base[2]>>2)+i]=value; end
            sep; for(i=0;i<512;i=i+1) begin word(value); expected[i]=value; end
            run;
            for(i=0;i<512;i=i+1) if(mem[i]!==expected[i]) $fatal(1,"conv case %0d word %0d",cs,i);
            for(i=0;i<704;i=i+1)
                if(mem[(win>>2)+i] !== (i<192 ? mem[(base[3]>>2)+i] : mem[(base[1]>>2)+i-192]))
                    $fatal(1,"conv window");
            sep; for(i=0;i<192;i=i+1) word(value);
        end
        $fclose(f);
        // The same 16 cases without the saved window take the streaming path (prefill's CONV4).
        f=$fopen("../../golden/vectors/conv4.hex","r");
        window=0;
        for(cs=0;cs<16;cs=cs+1) begin
            for(i=0;i<192;i=i+1) begin word(value);mem[(base[3]>>2)+i]=value; end
            sep;
            for(i=0;i<512;i=i+1) begin word(value);mem[(base[1]>>2)+i]=value; end
            sep;
            for(i=0;i<256;i=i+1) begin word(value);mem[(base[2]>>2)+i]=value; end
            sep; for(i=0;i<512;i=i+1) begin word(value); expected[i]=value; end
            run;
            for(i=0;i<512;i=i+1) if(mem[i]!==expected[i]) $fatal(1,"streamed conv case %0d word %0d",cs,i);
            sep; for(i=0;i<192;i=i+1) word(value);
        end
        $fclose(f);
        // Streamed ADD and FMA over 3 planes of 600 (many waves, a partial last beat), exact small integers.
        for(i=0;i<4;i=i+1) begin base[i]=8192*i;sn[i]=4;sm[i]=0;sp[i]=2400; end
        for(i=0;i<1800;i=i+1) begin
            mem[(base[1]>>2)+i]=$shortrealtobits(shortreal'(i%97+1));
            mem[(base[2]>>2)+i]=$shortrealtobits(shortreal'(2.0));
            mem[(base[3]>>2)+i]=$shortrealtobits(shortreal'(i%13));
        end
        op=1;n=600;m=1;p=3;got=cycles;run;
        for(i=0;i<1800;i=i+1) if(mem[i]!==$shortrealtobits(shortreal'(i%97+3))) $fatal(1,"streamed add %0d: got %h want %h, a %h b %h",i,mem[i],$shortrealtobits(shortreal'(i%97+3)),mem[(base[1]>>2)+i],mem[(base[2]>>2)+i]);
        $display("  streamed ADD: 1,800 values in %0d clocks with read and write stalls", cycles-got);
        op=5;got=cycles;run;
        for(i=0;i<1800;i=i+1) if(mem[i]!==$shortrealtobits(shortreal'(2*(i%97+1)+i%13))) $fatal(1,"streamed fma %0d",i);
        $display("  streamed FMA: 1,800 values in %0d clocks", cycles-got);
        for(i=0;i<4;i=i+1) begin base[i]=2048*i;sn[i]=4;sm[i]=0;sp[i]=256; end
        f=$fopen("../../golden/vectors/rope.hex","r");
        op=19;n=256;m=2;p=1;sn[2]=4;sm[0]=1024;sm[1]=1024;base[1]=base[0];window=0;
        for(cs=0;cs<64;cs=cs+1) begin
            word(value);sep;
            for(i=0;i<512;i=i+1) begin word(value);mem[i]=value; end
            sep; for(i=0;i<64;i=i+1) begin word(value);mem[(base[2]>>2)+i]=value; end
            sep; for(i=0;i<512;i=i+1) begin word(value);expected[i]=value; end
            run;
            for(i=0;i<512;i=i+1) if(mem[i]!==expected[i]) $fatal(1,"rope case %0d word %0d",cs,i);
        end
        $fclose(f);
        // Broadcast a different scalar to each plane, with a partial last beat.
        op=20;n=75;m=1;p=3;sp[0]=300;sp[1]=-4;sn[1]=0;base[1]=3208;
        position=17;limit=32;step=2;
        mem[802]=1;mem[801]=2;mem[800]=3;run;
        for(r=0;r<3;r=r+1) for(i=0;i<75;i=i+1)
            if(mem[r*75+i] !== (17+i>32+2*r ? 32'hff800000 : r+1)) $fatal(1,"mask");
        $display("PASS CONV4 windows, in-place RoPE, causal strides and stalls");
    end
endmodule
