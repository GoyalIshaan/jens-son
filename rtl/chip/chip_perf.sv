// Performance counters (note 09, part 8): 48-bit counts of where the clocks go, read on the host bus
// (chip_ctrl, 0x100 + 2i). All count while the core runs and is not idle, except the DRAM byte counts.
//   0       clocks running
//   1-6     clocks each queue had a command in flight: W, R, X, L, M, V
//   7-12    commands finished: W, R, X, L, M, V
//   13      issue stalled on WAITS (a semaphore at 0)
//   14      issue stalled on a full D, M or V queue
//   15      an LW or SW waiting for the SRAM
//   16, 17  the vector unit's read, write waiting for the SRAM
//   18      the matrix unit waiting for the SRAM (any of its ports)
//   19, 20  DMA X, L waiting for the SRAM
//   21      the matrix unit busy with less than one lut4 beat (576 bytes) in the weight FIFO
//   22      DRAM bytes requested by W and R
//   23      DRAM bytes written by X
module chip_perf #(parameter NPERF = 24)(
    input wire clk, reset, clear, run, idle,
    input wire [5:0] q_active, input wire [5:0] u_done,
    input wire st_wait, st_full, st_sram,
    input wire v_rd_wait, v_wr_wait, m_wait, x_wait, l_wait,
    input wire m_busy, input wire [31:0] fifo_level,
    input wire w_req, input wire [9:0] w_len, input wire r_req, input wire [9:0] r_len,
    input wire x_req, input wire [9:0] x_len,
    output wire [48*NPERF-1:0] count
);
    reg [47:0] c [0:NPERF-1];
    genvar g;
    for (g = 0; g < NPERF; g = g + 1) begin : out assign count[48*g+:48] = c[g]; end
    wire on = run && !idle;
    integer u;
    always @(posedge clk) begin
        if (reset || clear) begin
            for (u = 0; u < NPERF; u = u + 1) c[u] <= 0;
        end else begin
            if (on) begin
                c[0] <= c[0] + 1;
                for (u = 0; u < 6; u = u + 1) begin
                    if (q_active[u]) c[1 + u] <= c[1 + u] + 1;
                    if (u_done[u] && q_active[u]) c[7 + u] <= c[7 + u] + 1;
                end
                if (st_wait) c[13] <= c[13] + 1;
                if (st_full) c[14] <= c[14] + 1;
                if (st_sram) c[15] <= c[15] + 1;
                if (v_rd_wait) c[16] <= c[16] + 1;
                if (v_wr_wait) c[17] <= c[17] + 1;
                if (m_wait) c[18] <= c[18] + 1;
                if (x_wait) c[19] <= c[19] + 1;
                if (l_wait) c[20] <= c[20] + 1;
                if (m_busy && fifo_level < 576) c[21] <= c[21] + 1;
            end
            c[22] <= c[22] + (w_req ? w_len : 0) + (r_req ? r_len : 0);
            if (x_req) c[23] <= c[23] + x_len;
        end
    end
endmodule
