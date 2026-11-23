// The scalar core: issue, the S slot, the six command queues and the semaphores (ISA v1, D-022).
//
// One 256-bit bundle issues per clock, in order: S (bits 0-63), D (64-127), M (128-191), V (192-255).
// Issue stalls when WAITS finds its semaphore at 0 or a D/M/V slot's queue is full. The D/M/V slots read
// registers before the S slot writes. A D/M/V slot is present when its word is nonzero (an M or V NOP that
// only waits or signals is a fence; an all-zero word is an empty slot).
//
// Queues (16 deep): 0 W, 1 R, 2 X, 3 L (DMA), 4 M, 5 V. A queue's head starts when its unit is idle and every
// semaphore it waits on is > 0 (2 if it waits twice on one), then decrements them. When the unit signals done,
// the command's signal semaphores increment. Several starts in one clock are granted in unit order against
// the counts left by earlier grants; the scalar WAITS is granted after them.
//
// Scalar registers are 48 bits, unsigned (x0 = 0). The loop stack behaves as isa/sim_v1.py: at the end of a
// body the innermost loop branches back while iterations remain; several loops may end on one bundle.
module chip_core #(parameter IMEM = 8192, DMEM = 1024, QD = 16)(
    input wire clk, reset, run,
    output reg halted, output wire idle, output reg fault,
    // to the units: the head command of each queue, and a start pulse
    output wire [5:0] u_start, input wire [5:0] u_busy, input wire [5:0] u_done,
    output wire [4:0] u_op [0:5], output wire [511:0] u_desc [0:5], output wire [47:0] u_rv [0:5][0:4],
    // scalar SRAM port (LW / SW)
    output wire s_rd_valid, output wire [22:0] s_rd_addr, input wire [31:0] s_rd_data, input wire s_ready,
    output wire s_wr_valid, output wire [22:0] s_wr_addr, output wire [31:0] s_wr_data,
    output reg [63:0] cycles, output reg [63:0] bundles,
    // loading (chip_ctrl, while not running): instruction and descriptor memory words, scalar registers
    input wire ld_imem_we, ld_dmem_we, input wire [15:0] ld_addr, input wire [511:0] ld_data,
    input wire ld_reg_we, input wire [4:0] ld_reg, input wire [47:0] ld_reg_val,
    // for the performance counters: queues with a command in flight, and why issue stalled this clock
    output wire [5:0] q_active, output wire st_wait, output wire st_full, output wire st_sram,
    output wire [31:0] pc_o
);
    reg [255:0] imem [0:IMEM-1];
    reg [511:0] dmem [0:DMEM-1];
    reg [47:0] x [0:31];
    reg [15:0] sem [0:63];
    reg [31:0] pc;
    // ---------------------------------------------------------------- queues
    localparam EW = 5 + 10 + 240 + 24;                 // op, desc, 5 registers, 2 waits + 2 signals
    reg [EW-1:0] q [0:5][0:QD-1];
    reg [4:0] qn [0:5]; reg [3:0] qh [0:5], qt [0:5];
    // Each unit's commands in flight and their signal sets, oldest first. The DRAM-reading DMA units (W, R) take
    // their next command once the previous one has issued its last request (as isa/sim_v1.py), so up to SQ
    // commands can be waiting for their last bytes; done pulses come back in order. Other units: one at a time.
    localparam SQ = 4;
    reg [11:0] sq [0:6*SQ-1]; reg [1:0] sq_h [0:5]; reg [2:0] sq_n [0:5];
    wire [5:0] active;
    genvar ga;
    for (ga = 0; ga < 6; ga = ga + 1) begin : act
        assign active[ga] = sq_n[ga] != 0;
    end
    wire [11:0] cur_sig [0:5];
    for (ga = 0; ga < 6; ga = ga + 1) begin : cs
        assign cur_sig[ga] = sq[ga*SQ + sq_h[ga]];
    end
    genvar u, i;
    wire [EW-1:0] head [0:5];
    for (u = 0; u < 6; u = u + 1) begin : heads
        assign head[u] = q[u][qh[u]];
        assign u_op[u] = head[u][4:0];
        assign u_desc[u] = dmem[head[u][14:5]];
        for (i = 0; i < 5; i = i + 1) begin : rv
            assign u_rv[u][i] = head[u][15 + 48*i +: 48];
        end
    end
    // ---------------------------------------------------------------- the bundle
    wire [255:0] bd = imem[pc];
    wire [63:0] S = bd[63:0], D = bd[127:64], M = bd[191:128], V = bd[255:192];
    wire [4:0] s_op = S[4:0], rd = S[9:5], rs1 = S[14:10], rs2 = S[19:15];
    wire [31:0] imm = S[51:20];
    wire [47:0] simm = {{16{imm[31]}}, imm};
    wire [47:0] a = x[rs1], b = x[rs2];
    localparam NOP=0, ADD=1, SUB=2, MUL=3, AND=4, OR=5, XOR=6, MIN=7, MAX=8, ADDI=9, ANDI=10, SLLI=11, SRLI=12,
               LI=13, LW=14, SW=15, BEQ=16, BNE=17, BLT=18, J=19, LOOP=20, WAITS=21, SIGS=22, HALT=23;
    // D opcodes: 1 W 2 R 3 X 4 L 5 H2S 6 S2H 7 NOP_W 8 NOP_R 9 NOP_X 10 NOP_L
    wire [4:0] d_op = D[4:0];
    wire d_on = d_op != 0, m_on = M != 0, v_on = V != 0;
    wire [2:0] d_unit = (d_op == 1 || d_op == 7) ? 3'd0 : (d_op == 2 || d_op == 8) ? 3'd1 :
                        (d_op == 3 || d_op == 9) ? 3'd2 : 3'd3;
    // the queue entry of a slot: registers read now
    function [EW-1:0] entry(input [63:0] w, input [4:0] uop);
        integer k;
        begin
            entry = 0;
            entry[4:0] = uop; entry[14:5] = w[14:5];
            for (k = 0; k < 5; k = k + 1) entry[15 + 48*k +: 48] = x[w[15 + 5*k +: 5]];
            entry[255 +: 24] = w[63:40];
        end
    endfunction
    // D commands enter their queue with the unit-local opcode: W/R/X 1 = copy, 0 = NOP;
    // L: 1 = L, 2 = S2H, 3 = H2S (unsupported), 0 = NOP
    wire [4:0] d_uop = d_op >= 7 ? 5'd0 : d_op == 6 ? 5'd2 : d_op == 5 ? 5'd3 : 5'd1;
    // ---------------------------------------------------------------- semaphore grants
    reg [5:0] grant; reg [15:0] left [0:63]; reg waits_ok; integer s_, k_;
    always @* begin
        for (s_ = 0; s_ < 64; s_ = s_ + 1) left[s_] = sem[s_];
        grant = 0;
        for (k_ = 0; k_ < 6; k_ = k_ + 1) begin : g
            reg [5:0] w1, w2;
            w1 = head[k_][255 +: 6]; w2 = head[k_][261 +: 6];
            // W, R and M (the matrix unit overlaps commands: its busy says when it can take one) may have
            // several commands in flight; their dones come in order
            if (qn[k_] != 0 && (k_ < 2 || k_ == 4 ? sq_n[k_] < SQ : !active[k_]) && !u_busy[k_] &&
                (w1 == 0 || left[w1] > (w2 == w1 ? 1 : 0)) && (w2 == 0 || left[w2] > 0)) begin
                grant[k_] = 1;
                if (w1 != 0) left[w1] = left[w1] - 1;
                if (w2 != 0) left[w2] = left[w2] - 1;
            end
        end
        waits_ok = left[imm[5:0]] > 0;
    end
    assign u_start = grant;
    // ---------------------------------------------------------------- issue
    wire s_waits = S != 0 && s_op == WAITS;
    wire stall = (s_waits && !waits_ok) || (d_on && qn[d_unit] >= QD) || (m_on && qn[4] >= QD) || (v_on && qn[5] >= QD);
    // an LW or SW waits for the SRAM arbiter; the request does not depend on its answer
    wire pre_go = run && !halted && !stall && !reset;
    wire s_mem = S != 0 && (s_op == LW || s_op == SW);
    wire go = pre_go && (!s_mem || s_ready);
    assign q_active = active; assign pc_o = pc;
    wire issuing_clock = run && !halted && !reset;
    assign st_wait = issuing_clock && s_waits && !waits_ok;            // a WAITS on a semaphore at 0
    assign st_full = issuing_clock && stall && !(s_waits && !waits_ok); // a D, M or V queue full
    assign st_sram = pre_go && s_mem && !s_ready;                     // an LW or SW waiting for the SRAM
    // loading writes land only while the core is not running
    always @(posedge clk) begin
        if (ld_imem_we && !run) imem[ld_addr] <= ld_data[255:0];
        if (ld_dmem_we && !run) dmem[ld_addr] <= ld_data;
    end
    assign s_rd_valid = pre_go && S != 0 && s_op == LW; assign s_rd_addr = a[22:0] + imm[22:0];
    assign s_wr_valid = pre_go && S != 0 && s_op == SW; assign s_wr_addr = a[22:0] + imm[22:0]; assign s_wr_data = b[31:0];
    // next pc, before the loop stack
    reg [31:0] lst [0:3], lend [0:3]; reg [47:0] lcnt [0:3]; reg [2:0] lsp;
    reg [31:0] npc; reg [47:0] v; reg vw;
    always @* begin
        npc = pc + 1; v = 0; vw = 0;
        if (S != 0) case (s_op)
            ADD: begin v = a + b; vw = 1; end
            SUB: begin v = a - b; vw = 1; end
            MUL: begin v = a * b; vw = 1; end
            AND: begin v = a & b; vw = 1; end
            OR:  begin v = a | b; vw = 1; end
            XOR: begin v = a ^ b; vw = 1; end
            MIN: begin v = a < b ? a : b; vw = 1; end
            MAX: begin v = a > b ? a : b; vw = 1; end
            ADDI: begin v = a + simm; vw = 1; end
            ANDI: begin v = a & simm; vw = 1; end
            SLLI: begin v = a << imm[5:0]; vw = 1; end
            SRLI: begin v = a >> imm[5:0]; vw = 1; end
            LI: begin v = simm; vw = 1; end
            LW: begin v = {16'd0, s_rd_data}; vw = 1; end
            BEQ: if (a == b) npc = pc + 1 + imm;
            BNE: if (a != b) npc = pc + 1 + imm;
            BLT: if (a < b) npc = pc + 1 + imm;
            J: npc = pc + 1 + imm;
            LOOP: if (a == 0) npc = pc + 1 + imm;
            default: ;
        endcase
    end
    // the loop stack after this bundle: push, then the end-of-body check from the innermost loop outward
    reg [31:0] fpc; reg [2:0] nsp; reg [47:0] ncnt [0:3]; reg push, stop; integer l_;
    reg [31:0] pst [0:3], pend_ [0:3];
    always @* begin
        for (l_ = 0; l_ < 4; l_ = l_ + 1) begin pst[l_] = lst[l_]; pend_[l_] = lend[l_]; ncnt[l_] = lcnt[l_]; end
        nsp = lsp; push = S != 0 && s_op == LOOP && a != 0;
        if (push) begin pst[lsp[1:0]] = pc + 1; pend_[lsp[1:0]] = pc + 1 + imm; ncnt[lsp[1:0]] = a; nsp = lsp + 1; end
        fpc = npc; stop = 0;
        for (l_ = 0; l_ < 4; l_ = l_ + 1)
            if (!stop && nsp != 0 && fpc == pend_[nsp[1:0] - 1'b1]) begin
                if (ncnt[nsp[1:0] - 1'b1] > 1) begin
                    ncnt[nsp[1:0] - 1'b1] = ncnt[nsp[1:0] - 1'b1] - 1; fpc = pst[nsp[1:0] - 1'b1]; stop = 1;
                end else nsp = nsp - 1;
            end
    end
    // ---------------------------------------------------------------- state update
    assign idle = halted && active == 0 && qn[0] == 0 && qn[1] == 0 && qn[2] == 0 && qn[3] == 0 &&
                  qn[4] == 0 && qn[5] == 0;
    reg [15:0] inc [0:63], dec [0:63]; integer z, w_;
    always @(posedge clk) begin
        if (reset) begin
            pc <= 0; halted <= 0; fault <= 0; lsp <= 0;
            for (z = 0; z < 6; z = z + 1) begin sq_h[z] <= 0; sq_n[z] <= 0; end cycles <= 0; bundles <= 0;
            for (z = 0; z < 64; z = z + 1) sem[z] <= 0;
            for (z = 0; z < 6; z = z + 1) begin qn[z] <= 0; qh[z] <= 0; qt[z] <= 0; end
        end else begin
            if (run && !idle) cycles <= cycles + 1;
            if (ld_reg_we && !run && ld_reg != 0) x[ld_reg] <= ld_reg_val;
            for (z = 0; z < 64; z = z + 1) begin inc[z] = 0; dec[z] = 0; end
            // unit starts and completions
            for (z = 0; z < 6; z = z + 1) begin
                if (grant[z]) begin
                    if (head[z][255 +: 6] != 0) dec[head[z][255 +: 6]] = dec[head[z][255 +: 6]] + 1;
                    if (head[z][261 +: 6] != 0) dec[head[z][261 +: 6]] = dec[head[z][261 +: 6]] + 1;
                    sq[z*SQ + ((sq_h[z] + sq_n[z]) % SQ)] <= head[z][267 +: 12];
                    qh[z] <= qh[z] + 1'b1;
                end
                sq_n[z] <= sq_n[z] + grant[z] - (u_done[z] && active[z]);
                if (u_done[z] && active[z]) begin
                    sq_h[z] <= sq_h[z] + 1'b1;
                    if (cur_sig[z][5:0] != 0) inc[cur_sig[z][5:0]] = inc[cur_sig[z][5:0]] + 1;
                    if (cur_sig[z][11:6] != 0) inc[cur_sig[z][11:6]] = inc[cur_sig[z][11:6]] + 1;
                end
            end
            if (go) begin
                bundles <= bundles + 1;
                if (d_on) begin q[d_unit][qt[d_unit]] <= entry(D, d_uop); qt[d_unit] <= qt[d_unit] + 1'b1; end
                if (m_on) begin q[4][qt[4]] <= entry(M, M[4:0]); qt[4] <= qt[4] + 1'b1; end
                if (v_on) begin q[5][qt[5]] <= entry(V, V[4:0]); qt[5] <= qt[5] + 1'b1; end
                if (S != 0 && s_op == WAITS) dec[imm[5:0]] = dec[imm[5:0]] + 1;
                if (S != 0 && s_op == SIGS) inc[imm[5:0]] = inc[imm[5:0]] + 1;
                if (S != 0 && s_op == HALT) halted <= 1;
                if (vw && rd != 0) x[rd] <= v;
                if (S != 0 && s_op == LOOP && lsp == 4 && a != 0) begin fault <= 1; halted <= 1; end
                pc <= fpc; lsp <= nsp;
                for (z = 0; z < 4; z = z + 1) begin lst[z] <= pst[z]; lend[z] <= pend_[z]; lcnt[z] <= ncnt[z]; end
            end
            for (z = 0; z < 6; z = z + 1)
                qn[z] <= qn[z] + ((go && ((z < 4 && d_on && d_unit == z) || (z == 4 && m_on) || (z == 5 && v_on))) ? 1 : 0)
                               - (grant[z] ? 1 : 0);
            for (z = 1; z < 64; z = z + 1) sem[z] <= sem[z] + inc[z] - dec[z];
        end
    end
endmodule
