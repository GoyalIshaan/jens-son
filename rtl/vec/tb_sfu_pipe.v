// Stream independent inputs on consecutive clocks to exercise sideband alignment.
module tb_sfu_pipe;
    parameter FUNC = 0, L = 8;                         // L = 9: input-registered FMAs (1 GHz), as the row path uses
    localparam G = 1;                                  // the glue stages added for 1 GHz (rsqrt 2 + 1, sigmoid 1)
    localparam LAT = FUNC==0 ? 8*L : FUNC==1 ? 3*L : FUNC==2 ? 5*L+3*G :
                     FUNC==3 ? 13*L+G : FUNC==4 ? 17*L+G : 14*L+G;
    localparam COUNT = FUNC==1 ? 4106 : FUNC==2 ? 4099 : FUNC==5 ? 4007 : 4108;
    reg [31:0] words [0:2*COUNT-1];
    reg clk=0;
    reg [31:0] x;
    wire [31:0] y;
    if (FUNC==0) sfu_exp_core #(L) dut(clk,x,y);
    if (FUNC==1) sfu_recip_core #(L) dut(clk,x,y);
    if (FUNC==2) sfu_rsqrt_core #(L) dut(clk,x,y);
    if (FUNC==3) sfu_sigmoid_core #(L) dut(clk,x,y);
    if (FUNC==4) sfu_softplus_core #(L) dut(clk,x,y);
    if (FUNC==5) sfu_silu_core #(L) dut(clk,x,y);
    integer i, pos;
    initial begin
        case (FUNC)
            0: $readmemh("vec/sfu_exp.hex",words);
            1: $readmemh("vec/sfu_recip.hex",words);
            2: $readmemh("vec/sfu_rsqrt.hex",words);
            3: $readmemh("vec/sfu_sigmoid.hex",words);
            4: $readmemh("vec/sfu_softplus.hex",words);
            5: $readmemh("vec/silu.hex",words);
        endcase
        for (i=0; i<COUNT+LAT-1; i=i+1) begin
            x = i<COUNT ? words[2*i] : 32'h3f800000;
            #1 clk=1; #1 clk=0;
            pos = i-LAT+1;
            if (pos>=0 && !(FUNC==2 && (words[2*pos][31] || words[2*pos][30:23]==0)))
                if (y !== words[2*pos+1])
                    $fatal(1,"SFU %0d sample %0d got %h expected %h",FUNC,pos,y,words[2*pos+1]);
        end
        $display("PASS pipelined SFU %0d, %0d clocks", FUNC,LAT);
    end
endmodule
