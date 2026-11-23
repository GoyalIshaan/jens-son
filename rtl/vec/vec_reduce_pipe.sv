// R1. The same bank of 512 pipelined FMAs handles accumulation and all nine
// tree levels. ready/valid allows a banked SRAM frontend to stall any beat.
module vec_reduce_pipe #(parameter LANES=512, parameter LOCAL_BANK=1)(
    input wire clk, reset, valid, first, last, square,
    input wire [LANES*32-1:0] x,
    output wire ready, output reg result_valid, output reg [31:0] result,
    output wire [LANES*32-1:0] fa_bus,fb_bus,fc_bus,
    input wire [LANES*32-1:0] bank_result
);
    localparam INPUT_BEAT=0, WAIT_BEAT=1, TREE=2, WAIT_TREE=3;
    reg [1:0] phase;
    reg [2:0] wait_count;
    reg final_beat;
    reg [$clog2(LANES):0] live;
    wire [LANES*32-1:0] fy;
    reg [LANES*32-1:0] acc;
    assign ready = phase == INPUT_BEAT;
    genvar j;
    for (j=0; j<LANES; j=j+1) begin : lane
        wire [31:0] left, right;
        if (2*j+1 < LANES) begin
            assign left = acc[64*j+:32];
            assign right = acc[64*j+32+:32];
        end else begin assign left=0; assign right=0; end
        wire [31:0] sample = x[j*32+:32];
        assign fa_bus[32*j+:32] = phase==TREE ? left : sample;
        assign fb_bus[32*j+:32] = phase==TREE || !square ? 32'h3f800000 : sample;
        assign fc_bus[32*j+:32] = phase==TREE ? right : first ? 32'd0 : acc[32*j+:32];
        if (LOCAL_BANK) fp32_fma_pipe8 madd(clk,fa_bus[32*j+:32],fb_bus[32*j+:32],fc_bus[32*j+:32],fy[32*j+:32]);
        else assign fy[32*j+:32]=bank_result[32*j+:32];
    end
    always @(posedge clk) begin
        result_valid <= 0;
        if (reset) phase <= INPUT_BEAT;
        else case (phase)
            INPUT_BEAT: if (valid) begin
                final_beat <= last;
                phase <= WAIT_BEAT;
                wait_count <= 7;
            end
            WAIT_BEAT, WAIT_TREE: if (wait_count != 0) wait_count <= wait_count-1'b1;
            else begin
                acc <= fy;
                if (phase == WAIT_BEAT && !final_beat) phase <= INPUT_BEAT;
                else if (phase == WAIT_TREE && live == 2 || LANES == 1) begin
                    result <= fy[31:0]; result_valid <= 1; phase <= INPUT_BEAT;
                end else begin
                    live <= phase == WAIT_BEAT ? LANES : live >> 1;
                    phase <= TREE;
                end
            end
            TREE: begin wait_count <= 7; phase <= WAIT_TREE; end
        endcase
    end
endmodule
