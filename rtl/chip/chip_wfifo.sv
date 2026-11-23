// The weight FIFO (192 KB, D-023): written only by DMA W, read only by the matrix unit, in program order.
//
// Storage: SIZE / 64 lines of 64 bytes, a ring. Line l lives in bank l mod 32 (32 two-port banks, one read and
// one write port each, 6 KB per bank), so every access below touches distinct banks:
//   read   the 1,024-byte front window spans at most 17 consecutive lines
//   write  a PB-byte (512) piece at any byte offset spans at most 9 consecutive lines
// The front window is a register: in each clock the FIFO reads the lines at the read pointer after this clock's
// pop, so the window for the next clock is ready at the edge. A line written in the same clock is bypassed from
// the write data, so the window always shows the ring's current bytes (bytes past `level` are don't-care).
// A pop removes 32, 576 or 1,024 bytes; a write adds up to PB.
module chip_wfifo #(parameter SIZE = 196608, PB = 512)(
    input wire clk, reset,
    input wire wr_valid, input wire [8*PB-1:0] wr_data, input wire [9:0] wr_len,
    input wire pop, input wire [10:0] pop_len,
    output reg [31:0] level, output wire [31:0] free, output reg [8191:0] head
);
    localparam NL = SIZE / 64;                         // lines
    localparam RL = 17;                                // lines a 1,024-byte window can span
    localparam WL = PB / 64 + 1;                       // lines a PB-byte piece can span
    reg [511:0] mem [0:NL-1];
    reg [31:0] rd, wr;                                 // byte positions in the ring
    assign free = SIZE - level;
    function [31:0] lmod(input [31:0] l); lmod = l % NL; endfunction

    // ---- this clock's write, as masked lines: line wl0 + i gets bytes where its mask is set
    wire [31:0] wl0 = wr >> 6;
    reg [511:0] w_line [0:WL-1]; reg [63:0] w_mask [0:WL-1];
    integer i, b, p;
    always @* begin
        for (i = 0; i < WL; i = i + 1) begin w_line[i] = 0; w_mask[i] = 0; end
        p = 0;
        if (wr_valid)
            for (b = 0; b < PB; b = b + 1)
                if (b < wr_len) begin
                    p = (wr % 64) + b;                     // byte position from the start of line wl0
                    w_line[p / 64][8*(p % 64)+:8] = wr_data[8*b+:8];
                    w_mask[p / 64][p % 64] = 1'b1;
                end
    end
    // ---- the window for the next clock: lines at the read pointer after this clock's pop, with bypass
    wire [31:0] rd_next = (rd + (pop ? pop_len : 0)) % SIZE;
    wire [31:0] rl0 = rd_next >> 6;
    reg [512*RL-1:0] win;
    reg [511:0] ln;
    integer k, wi, bb;
    always @* begin
        for (k = 0; k < RL; k = k + 1) begin
            ln = mem[lmod(rl0 + k)];
            for (wi = 0; wi < WL; wi = wi + 1)
                if (lmod(wl0 + wi) == lmod(rl0 + k))
                    for (bb = 0; bb < 64; bb = bb + 1) if (w_mask[wi][bb]) ln[8*bb+:8] = w_line[wi][8*bb+:8];
            win[512*k+:512] = ln;
        end
    end
    integer wq, bq;
    always @(posedge clk) begin
        if (reset) begin rd <= 0; wr <= 0; level <= 0; end
        else begin
            if (wr_valid) begin
`ifndef SYNTHESIS
                if (level + wr_len - (pop ? pop_len : 0) > SIZE) $fatal(1, "weight FIFO overflow");
`endif
                for (wq = 0; wq < WL; wq = wq + 1)
                    for (bq = 0; bq < 64; bq = bq + 1)
                        if (w_mask[wq][bq]) mem[lmod(wl0 + wq)][8*bq+:8] <= w_line[wq][8*bq+:8];
                wr <= (wr + wr_len) % SIZE;
            end
            if (pop) begin
`ifndef SYNTHESIS
                if (level < pop_len) $fatal(1, "weight FIFO underflow");
`endif
                rd <= rd_next;
            end
            level <= level + (wr_valid ? wr_len : 0) - (pop ? pop_len : 0);
        end
        head <= win >> (8 * (rd_next % 64));           // the 1,024 bytes from rd_next
    end
endmodule
