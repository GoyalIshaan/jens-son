// Whole-chip testbench: loads a program and its SRAM and DRAM images, runs until the core has halted and every
// queue and unit is idle, then dumps SRAM and DRAM for run_chip.py to compare with the functional simulator.
// DRAM model, as isa/sim_v1.py times it: one shared server of BW bytes per clock (512 GB/s x 0.92 at 1 GHz), reads
// before weight reads before writes, writes costing 1.15x their bytes; reads arrive LAT clocks after they are served.
// Loading: by default the program, descriptors and registers are written straight into the core's memories.
// +boot loads them the way a host would instead, over the register bus (chip_ctrl), and starts the chip with
// CONTROL.run. The performance counters are printed at the end, read over the bus.
module tb_chip #(parameter SRAM_BYTES = 6815744, DRAM_BYTES = 1 << 24, LAT = 150, RP = 4, MAXR = 128, ACCW = 8192, ER = 16,
                 PB = 512, BW = 471, BANKS = 64, ARB = 1, SKEW = 0, WBUF = 0);
    reg clk = 0, reset = 1, run = 0;
    wire halted, idle, fault, chip_reset;
    reg bus_valid = 0, bus_write = 0; reg [15:0] bus_addr = 0; reg [31:0] bus_wdata = 0;
    wire [31:0] bus_rdata; wire bus_rvalid;
    wire w_dr_valid, r_dr_valid, x_dw_valid, h_valid;
    wire [35:0] w_dr_addr, r_dr_addr, x_dw_addr; wire [9:0] w_dr_len, r_dr_len, h_len;
    wire [8*PB-1:0] x_dw_data, h_data; wire [PB-1:0] x_dw_mask;
    reg w_resp_valid = 0, r_resp_valid = 0; reg [8*PB-1:0] w_resp_data, r_resp_data;
    // the shared DRAM server: a credit of bytes, refilled by BW per clock
    integer credit = 0;
    wire [31:0] x_len = 0;
    reg [9:0] xl; integer xb;
    always @* begin xl = 0; for (xb = 0; xb < PB; xb = xb + 1) if (x_dw_mask[xb]) xl = xb + 1; end
    wire r_ok = r_dr_valid && credit >= r_dr_len;
    wire w_ok = w_dr_valid && credit - (r_ok ? r_dr_len : 0) >= w_dr_len;
    wire x_ok = x_dw_valid && credit - (r_ok ? r_dr_len : 0) - (w_ok ? w_dr_len : 0) >= (xl * 115 + 99) / 100;
    chip_top #(.SRAM_BYTES(SRAM_BYTES), .MXU_RP(RP), .MXU_MAXR(MAXR), .MXU_ACCW(ACCW), .MXU_ER(ER), .PB(PB), .SRAM_BANKS(BANKS), .SRAM_ARB(ARB), .SRAM_SKEW(SKEW), .SRAM_WB(WBUF)) dut(.clk(clk),
        .reset(reset), .run_pin(run), .halted(halted), .idle(idle), .fault(fault), .chip_reset(chip_reset),
        .bus_valid(bus_valid), .bus_write(bus_write), .bus_addr(bus_addr), .bus_wdata(bus_wdata),
        .bus_rdata(bus_rdata), .bus_rvalid(bus_rvalid),
        .w_dr_valid(w_dr_valid), .w_dr_ready(w_ok), .w_dr_addr(w_dr_addr), .w_dr_len(w_dr_len),
        .w_resp_valid(w_resp_valid), .w_resp_data(w_resp_data),
        .r_dr_valid(r_dr_valid), .r_dr_ready(r_ok), .r_dr_addr(r_dr_addr), .r_dr_len(r_dr_len),
        .r_resp_valid(r_resp_valid), .r_resp_data(r_resp_data),
        .x_dw_valid(x_dw_valid), .x_dw_ready(x_ok), .x_dw_addr(x_dw_addr), .x_dw_data(x_dw_data), .x_dw_mask(x_dw_mask),
        .h_valid(h_valid), .h_data(h_data), .h_len(h_len));
    reg [7:0] dram [0:DRAM_BYTES-1];
    // read pipelines: data is taken when the request is served, delivered LAT clocks later
    reg [8*PB-1:0] wq [0:LAT-1], rq [0:LAT-1]; reg [LAT-1:0] wv = 0, rv = 0;
    integer k, i, n, f, hf, limit, fails;
    reg [8*256-1:0] dir;
    function [8*PB-1:0] dram_line(input [35:0] a, input [9:0] len);
        integer b;
        begin
            dram_line = 0;
            for (b = 0; b < PB; b = b + 1)
                if (b < len) begin
                    if (a + b >= DRAM_BYTES) $fatal(1, "DRAM read out of range %0d", a + b);
                    dram_line[8*b+:8] = dram[a + b];
                end
        end
    endfunction
    always @(posedge clk) begin
        if (reset) credit <= 0;
        else credit <= (credit > 4 * PB ? 4 * PB : credit) + BW - (r_ok ? r_dr_len : 0) - (w_ok ? w_dr_len : 0)
                       - (x_ok ? (xl * 115 + 99) / 100 : 0);
        w_resp_valid <= wv[LAT-1]; w_resp_data <= wq[LAT-1];
        r_resp_valid <= rv[LAT-1]; r_resp_data <= rq[LAT-1];
        for (k = LAT-1; k > 0; k = k - 1) begin wq[k] <= wq[k-1]; rq[k] <= rq[k-1]; end
        wv <= {wv[LAT-2:0], w_ok && !reset}; rv <= {rv[LAT-2:0], r_ok && !reset};
        if (w_ok) wq[0] <= dram_line(w_dr_addr, w_dr_len);
        if (r_ok) rq[0] <= dram_line(r_dr_addr, r_dr_len);
        if (x_ok) for (k = 0; k < PB; k = k + 1) if (x_dw_mask[k]) begin
            if (x_dw_addr + k >= DRAM_BYTES) $fatal(1, "DRAM write out of range");
            dram[x_dw_addr + k] <= x_dw_data[8*k+:8];
        end
        if (h_valid) begin
            for (k = 0; k < PB; k = k + 1) if (k < h_len) $fwrite(hf, "%02x", h_data[8*k+:8]);
            $fwrite(hf, "\n");
        end
    end
    // vector-unit SRAM waits by opcode
    longint v_wait_r [0:31], v_wait_w [0:31];
    initial for (int i = 0; i < 32; i++) begin v_wait_r[i] = 0; v_wait_w[i] = 0; end
    always @(posedge clk) if (run) begin
        if (dut.v_rd_valid && !dut.v_rd_ready) v_wait_r[dut.vec.o] += 1;
        if (dut.v_wr_valid && !dut.v_wr_ready) v_wait_w[dut.vec.o] += 1;
    end
    // per-command trace: start and done clocks of every unit (compare_timing.py)
    integer tf = 0, clk_n = 0, tu;
    always @(posedge clk) if (run) begin
        clk_n <= clk_n + 1;
        if (tf != 0) for (tu = 0; tu < 6; tu = tu + 1) begin
            if (dut.core.u_start[tu]) $fwrite(tf, "%0d S %0d %0d\n", clk_n, tu, dut.core.u_op[tu]);
            if (dut.core.u_done[tu] && dut.core.active[tu]) $fwrite(tf, "%0d D %0d\n", clk_n, tu);
        end
    end
    always #1 clk = !clk;
    // the host bus: one write or read per clock
    task bus_wr(input [15:0] a, input [31:0] d);
        begin bus_valid = 1; bus_write = 1; bus_addr = a; bus_wdata = d; @(posedge clk); #0.5 bus_valid = 0; end
    endtask
    reg [31:0] rd_word;
    task bus_rd(input [15:0] a);
        begin bus_valid = 1; bus_write = 0; bus_addr = a; @(posedge clk); #0.5 bus_valid = 0; @(posedge clk); #0.5 rd_word = bus_rdata; end
    endtask
    reg [255:0] l_imem [0:8191]; reg [511:0] l_dmem [0:1023]; integer boot, w;
    integer bk;                                        // the main sequence's own loop index (the DRAM block clobbers k every clock)
    reg [47:0] rval; integer ridx;
    initial begin
        if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= needed");
        if (!$value$plusargs("limit=%d", limit)) limit = 2000000;
        boot = $test$plusargs("boot");
        if (boot) begin
            for (w = 0; w < 8192; w = w + 1) l_imem[w] = 'x;
            for (w = 0; w < 1024; w = w + 1) l_dmem[w] = 'x;
            $readmemh({dir, "/imem.hex"}, l_imem);
            $readmemh({dir, "/dmem.hex"}, l_dmem);
        end else begin
            $readmemh({dir, "/imem.hex"}, dut.core.imem);
            $readmemh({dir, "/dmem.hex"}, dut.core.dmem);
        end
        $readmemh({dir, "/sram.hex"}, dut.sram.mem);
        $readmemh({dir, "/dram.hex"}, dram);
        hf = $fopen({dir, "/host_out.txt"}, "w");
        tf = $fopen({dir, "/trace.txt"}, "w");
        repeat (3) @(posedge clk);
        #0.5 reset = 0;
        while (chip_reset) @(posedge clk);                 // the reset sequencer
        #0.5;
        for (k = 0; k < 32; k = k + 1) dut.core.x[k] = 0;
        if (boot) begin
            // the host's boot: program words, descriptor words, registers, then run
            bus_rd(16'h000); if (rd_word !== 32'h43484950) $fatal(1, "bad chip ID %h", rd_word);
            for (w = 0; w < 8192 && l_imem[w] !== 'x; w = w + 1) begin
                if (w == 0) bus_wr(16'h010, 0);
                for (bk = 0; bk < 8; bk = bk + 1) bus_wr(16'h011, l_imem[w][32*bk+:32]);
                bus_wr(16'h012, 1);
            end
            for (w = 0; w < 1024 && l_dmem[w] !== 'x; w = w + 1) begin
                if (w == 0) bus_wr(16'h010, 0);
                for (bk = 0; bk < 16; bk = bk + 1) bus_wr(16'h011, l_dmem[w][32*bk+:32]);
                bus_wr(16'h012, 2);
            end
            f = $fopen({dir, "/regs.txt"}, "r");
            while ($fscanf(f, "%d %d", ridx, rval) == 2) begin
                bus_wr(16'h013, ridx); bus_wr(16'h014, rval[31:0]); bus_wr(16'h015, {16'd0, rval[47:32]});
            end
            $fclose(f);
            $display("booted over the bus");
            bus_wr(16'h0ff, 1);
            $display("run start");
            bus_wr(16'h002, 1);
        end else begin
            f = $fopen({dir, "/regs.txt"}, "r");
            while ($fscanf(f, "%d %d", ridx, rval) == 2) dut.core.x[ridx] = rval;
            $fclose(f);
            $display("run start");
            @(posedge clk); #0.5 run = 1;
        end
        n = 0;
        while (!(idle && !wv && !rv) && n < limit && !fault) begin
            @(posedge clk); n = n + 1;
            if (n % 2000 == 0) begin $display("clock %0d pc %0d bundles %0d", n, dut.core.pc, dut.core.bundles); end
        end
        repeat (2) @(posedge clk);
        $display("SRAM banks: %0d clocks with SRAM traffic, %0d lines; %0d one-line banks need %0d clocks for them",
                 dut.sram.busy_clocks, dut.sram.lines_total, BANKS, dut.sram.bank_clocks);
        $display("SRAM arbiter waits, in line-clocks, by holder (V-read V-write M X L S R order):");
        for (int p = 0; p < 6; p++)
            $display("  %s: %0d %0d %0d %0d %0d %0d %0d %0d", p == 0 ? "V-read " : p == 1 ? "V-write" : p == 2 ? "M      " : p == 3 ? "X      " : p == 4 ? "L      " : "S      ",
                     dut.sram.lost[p*8], dut.sram.lost[p*8+1], dut.sram.lost[p*8+2], dut.sram.lost[p*8+3],
                     dut.sram.lost[p*8+4], dut.sram.lost[p*8+5], dut.sram.lost[p*8+6], dut.sram.lost[p*8+7]);
        $display("M wait clocks by active ports {weights loads drain-reads drain-writes}:");
        for (int i = 0; i < 16; i++) if (dut.sram.m_wait_kind[i] != 0) $display("  %b: %0d", i[3:0], dut.sram.m_wait_kind[i]);
        $display("V wait clocks by opcode (read, write):");
        for (int i = 0; i < 32; i++) if (v_wait_r[i] != 0 || v_wait_w[i] != 0) $display("  op %0d: %0d %0d", i, v_wait_r[i], v_wait_w[i]);
        $display("SRAM arbiter (%0d banks, skew=%0d, on=%0d): %0d clocks with a waiting requester; waits V-read %0d V-write %0d M %0d X %0d L %0d S %0d",
                 BANKS, SKEW, ARB, dut.sram.any_stall, dut.sram.stall_clocks[0], dut.sram.stall_clocks[1], dut.sram.stall_clocks[2],
                 dut.sram.stall_clocks[3], dut.sram.stall_clocks[4], dut.sram.stall_clocks[5]);
        $display("RESULT halted=%0d idle=%0d fault=%0d clocks=%0d bundles=%0d", halted, idle, fault, n, dut.core.bundles);
        // the performance counters, read the way the host would
        $write("PERF");
        for (bk = 0; bk < 24; bk = bk + 1) begin bus_rd(16'h100 + 2 * bk); $write(" %0d", rd_word); end
        $display("");
        if (n >= limit) $display("TIMEOUT at pc %0d", dut.core.pc);
        $writememh({dir, "/sram_out.hex"}, dut.sram.mem);
        $writememh({dir, "/dram_out.hex"}, dram);
        $fclose(hf); $fclose(tf);
        $finish;
    end
endmodule
