// Control: the host's 32-bit register bus, the reset sequencer, program loading and the host input FIFO
// (note 09: the register map, reset and boot, talking to the host).
//
// The bus: one access per clock, always ready. A read returns its word the next clock (rvalid).
// Register map (word addresses):
//   0x000  ID           RO  0x43484950 ("CHIP")
//   0x001  VERSION      RO  1
//   0x002  CONTROL      RW  bit 0 run; writing bit 1 = 1 starts a soft reset (the sequencer, below)
//   0x003  STATUS       RO  bit 0 halted, 1 idle, 2 fault, 3 in reset; bits 8-11 which units faulted
//                           (sticky until reset: core, matrix, vector, DMA L)
//   0x004  PC           RO
//   0x005  CYCLES       RO  low word, 0x006 high word (clocks run while not idle)
//   0x007  BUNDLES      RO  low word, 0x008 high word
//   0x010  LD_ADDR      RW  word index into the instruction or descriptor memory
//   0x011  LD_DATA      WO  the next 32 bits of the staged word (low first; the piece index advances)
//   0x012  LD_COMMIT    WO  1: instruction word (256 bits) at LD_ADDR; 2: descriptor word (512 bits).
//                           Then LD_ADDR advances and the piece index returns to 0.
//   0x013  REG_IDX      RW  scalar register index
//   0x014  REG_LO       RW  bits 0-31 of the value
//   0x015  REG_HI       WO  bits 32-47; writing it sets the register
//   0x020  HIN_DATA     WO  push 4 bytes into the host input FIFO (H2S reads it)
//   0x021  HIN_LEVEL    RO  bytes in the FIFO
//   0x022  HIN_FREE     RO  bytes of room
//   0x0FF  PERF_CLEAR   WO  any write clears the performance counters
//   0x100  PERF         RO  counter i at 0x100 + 2i (low word) and 0x101 + 2i (high), chip_perf's order
// Loading, register writes and HIN pushes are meant for while the core is stopped (the core ignores loads
// while running).
//
// Reset sequencer: the external reset, or a soft reset, holds the rest of the chip in reset for RST_CLOCKS
// clocks; CONTROL.run clears. Boot: the host loads the program and registers, then sets CONTROL.run.
module chip_ctrl #(parameter RST_CLOCKS = 16, HIN = 4096, PB = 512, NPERF = 24)(
    input wire clk, ext_reset,
    // the host bus
    input wire bus_valid, bus_write, input wire [15:0] bus_addr, input wire [31:0] bus_wdata,
    output reg [31:0] bus_rdata, output reg bus_rvalid,
    // to the chip
    output wire chip_reset, output wire run,
    input wire halted, idle, fault, input wire [3:0] fault_src, input wire [31:0] pc, input wire [63:0] cycles, bundles,
    output reg ld_imem_we, ld_dmem_we, output reg [15:0] ld_addr, output reg [511:0] ld_data,
    output reg ld_reg_we, output reg [4:0] ld_reg, output reg [47:0] ld_reg_val,
    // the host input FIFO (H2S)
    output reg [31:0] hi_level, output wire [8*PB-1:0] hi_data, input wire hi_pop, input wire [9:0] hi_pop_len,
    // the performance counters
    input wire [48*NPERF-1:0] perf, output reg perf_clear
);
    reg [4:0] seq;                                     // reset clocks left
    reg ctl_run;
    assign chip_reset = ext_reset || seq != 0;
    assign run = ctl_run;
    reg [3:0] piece; reg [511:0] stage; reg [31:0] reg_lo; reg [3:0] faulted;
    // host input FIFO: a ring of HIN bytes; the front PB bytes are visible to DMA L
    reg [7:0] hin [0:HIN-1];
    reg [31:0] h_rd, h_wr;
    genvar g;
    for (g = 0; g < PB; g = g + 1) begin : hfront assign hi_data[8*g+:8] = hin[(h_rd + g) % HIN]; end
    wire push = bus_valid && bus_write && bus_addr == 16'h020;
    integer k;
    always @(posedge clk) begin
        ld_imem_we <= 0; ld_dmem_we <= 0; ld_reg_we <= 0; perf_clear <= 0; bus_rvalid <= 0;
        if (ext_reset) begin
            seq <= RST_CLOCKS; ctl_run <= 0; piece <= 0; ld_addr <= 0; faulted <= 0;
            h_rd <= 0; h_wr <= 0; hi_level <= 0;
        end else begin
            if (seq != 0) seq <= seq - 1'b1;
            if (chip_reset) faulted <= 0; else faulted <= faulted | fault_src;
            // host input FIFO
            if (push) for (k = 0; k < 4; k = k + 1) hin[(h_wr + k) % HIN] <= bus_wdata[8*k+:8];
            if (push) h_wr <= (h_wr + 4) % HIN;
            if (hi_pop) h_rd <= (h_rd + hi_pop_len) % HIN;
            hi_level <= hi_level + (push ? 4 : 0) - (hi_pop ? hi_pop_len : 0);
            // register writes
            if (bus_valid && bus_write) case (bus_addr)
                16'h002: begin
                    ctl_run <= bus_wdata[0];
                    if (bus_wdata[1]) begin seq <= RST_CLOCKS; ctl_run <= 0; end
                end
                16'h010: begin ld_addr <= bus_wdata[15:0]; piece <= 0; end
                16'h011: begin stage[32*piece+:32] <= bus_wdata; piece <= piece + 1'b1; end
                16'h012: begin
                    ld_data <= stage;
                    if (bus_wdata[1:0] == 2'd1) ld_imem_we <= 1;
                    if (bus_wdata[1:0] == 2'd2) ld_dmem_we <= 1;
                    piece <= 0;
                end
                16'h013: ld_reg <= bus_wdata[4:0];
                16'h014: reg_lo <= bus_wdata;
                16'h015: begin ld_reg_val <= {bus_wdata[15:0], reg_lo}; ld_reg_we <= 1; end
                16'h0ff: perf_clear <= 1;
                default: ;
            endcase
            // the load address advances after a commit (the write itself used the old address)
            if (ld_imem_we || ld_dmem_we) ld_addr <= ld_addr + 1'b1;
            // reads
            if (bus_valid && !bus_write) begin
                bus_rvalid <= 1;
                if (bus_addr >= 16'h100 && bus_addr < 16'h100 + 2 * NPERF)
                    bus_rdata <= bus_addr[0] ? {16'd0, perf[48*((bus_addr - 16'h100) >> 1) + 32 +: 16]}
                                             : perf[48*((bus_addr - 16'h100) >> 1) +: 32];
                else case (bus_addr)
                    16'h000: bus_rdata <= 32'h43484950;
                    16'h001: bus_rdata <= 32'd1;
                    16'h002: bus_rdata <= {31'd0, ctl_run};
                    16'h003: bus_rdata <= {20'd0, faulted, 4'd0, chip_reset, fault, idle, halted};
                    16'h004: bus_rdata <= pc;
                    16'h005: bus_rdata <= cycles[31:0];
                    16'h006: bus_rdata <= cycles[63:32];
                    16'h007: bus_rdata <= bundles[31:0];
                    16'h008: bus_rdata <= bundles[63:32];
                    16'h010: bus_rdata <= {16'd0, ld_addr};
                    16'h013: bus_rdata <= {27'd0, ld_reg};
                    16'h014: bus_rdata <= reg_lo;
                    16'h021: bus_rdata <= hi_level;
                    16'h022: bus_rdata <= HIN - hi_level;
                    default: bus_rdata <= 32'd0;
                endcase
            end
        end
    end
endmodule
