// On-chip SRAM (6.5 MB by default; byte addresses, 23 bits), with a cycle model of the banked arbiter.
//
// Function. Every client has its own port. A port's access happens, all at once, in the clock its ready is
// high: a read sees the memory as it was before that clock's writes (reads are sampled at the falling
// edge), and a write lands at the rising edge. Writes from different ports in one clock land in a fixed
// order: V, M, R, L, scalar. A correct program never has two ports writing one byte in the same clock
// (semaphores order them). This is the functional contract the banked SRAM keeps; the arbiter below only
// decides when each ready goes high, so it can change timing but never results.
//
//   V  the vector unit's gather and scatter ports: 3 x LANES reads and LANES writes of 1, 2 or 4 bytes
//   M  the matrix unit (D-029): W, A 1,024-byte reads (MMS weights, activations), E a 64-byte read (exponents),
//      D ER 64-byte reads (the acc epilogue's dst) and O ER 64-byte masked writes (the epilogue's lines)
//   R  DMA R (DRAM -> SRAM): a PB-byte masked write (PB = 512)
//   X  DMA X (SRAM -> DRAM): a PB-byte read
//   L  DMA L (SRAM -> SRAM): a PB-byte read and a PB-byte masked write
//   S  the scalar core's LW and SW: 32-bit words
//
// The arbiter (ARB = 1, Verilator builds; D-028). NB single-ported banks of 64-byte lines, low-order
// interleaved (bank = line mod NB). Each bank serves one line per clock; reads of one line by several
// ports in the same clock share that access. Six requesters: V read (all three operands together), V write,
// M (all its ports together), X, L (read and write together) and S. R is served first every clock: DRAM
// responses cannot wait, and its at most 9 contiguous lines always fall in distinct banks. The other six
// take turns at first pick (round robin, rotated every clock); each takes every line it still needs whose
// bank is free. A request can be served over several clocks: the arbiter remembers which of its lines are
// done, and its ready goes high in the clock the last one is. V read and V write go ready together (the vector unit
// moves only when both are served), so one half never finishes alone and starts over (that livelocked). Within a requester, writes are served only
// once all its reads are (M's acc epilogue and L read lines and then write them). A request that changes
// before it is ready (a client may move on) starts over. ARB = 0, or an Icarus build: every ready is 1.
//
// SKEW = 1 rotates each block of NB lines by its block number (plus the block-of-blocks number):
// bank = (line + line / NB + line / NB^2) mod NB. Rows at a power-of-two stride (4 KB rows with 64 banks)
// then spread over the banks instead of all landing in one; any run of up to NB - 1 contiguous lines still
// falls in distinct banks. Software sees the same flat addresses.
//
// Attribution: every clock, every line a requester still needs and cannot get counts once against whoever
// holds its bank (R, a requester, or the requester itself) or against the reads-before-writes order.
module chip_sram #(parameter SIZE = 6815744, LANES = 512, PB = 512, ER = 16, NB = 64, ARB = 1, SKEW = 0, WB = 0)(
    input wire clk,
    // V
    input wire v_rd_valid, output wire v_rd_ready,
    input wire [22:0] v_rd_addr[0:2][0:LANES-1], input wire [2:0] v_rd_type[0:2][0:LANES-1],
    input wire [LANES-1:0] v_rd_mask[0:2], output reg [31:0] v_rd_data[0:2][0:LANES-1],
    input wire v_wr_valid, output wire v_wr_ready, input wire [LANES-1:0] v_wr_mask, input wire [22:0] v_wr_addr[0:LANES-1],
    input wire [31:0] v_wr_data[0:LANES-1], input wire [2:0] v_wr_type[0:LANES-1],
    // M (one ready for all its ports)
    output wire m_ready,
    input wire mw_rd_valid, input wire [22:0] mw_rd_addr, output reg [8191:0] mw_rd_data,
    input wire ma_rd_valid, input wire [22:0] ma_rd_addr, output reg [8191:0] ma_rd_data,
    input wire me_rd_valid, input wire [22:0] me_rd_addr, output reg [511:0] me_rd_data,
    input wire [ER-1:0] md_rd_valid, input wire [23*ER-1:0] md_rd_addr, output reg [512*ER-1:0] md_rd_data,
    input wire [ER-1:0] mo_wr_valid, input wire [23*ER-1:0] mo_wr_addr, input wire [512*ER-1:0] mo_wr_data,
    input wire [64*ER-1:0] mo_wr_mask,
    // R (never waits)
    input wire r_wr_valid, input wire [22:0] r_wr_addr, input wire [8*PB-1:0] r_wr_data, input wire [PB-1:0] r_wr_mask,
    // X
    input wire x_rd_valid, output wire x_ready, input wire [22:0] x_rd_addr, output reg [8*PB-1:0] x_rd_data,
    // L (one ready for its read and write)
    output wire l_ready,
    input wire l_rd_valid, input wire [22:0] l_rd_addr, output reg [8*PB-1:0] l_rd_data,
    input wire l_wr_valid, input wire [22:0] l_wr_addr, input wire [8*PB-1:0] l_wr_data, input wire [PB-1:0] l_wr_mask,
    // S
    output wire s_ready,
    input wire s_rd_valid, input wire [22:0] s_rd_addr, output reg [31:0] s_rd_data,
    input wire s_wr_valid, input wire [22:0] s_wr_addr, input wire [31:0] s_wr_data
);
    reg [7:0] mem [0:SIZE-1];
    integer s, j, k, a, n;
    function integer size_of(input [2:0] t);
        size_of = t == 1 ? 2 : (t == 2 || t == 4) ? 1 : 4;
    endfunction
    task check(input integer addr, input integer bytes, input [8*8-1:0] who);
        if (addr < 0 || addr + bytes > SIZE) $fatal(1, "SRAM %0s access out of range: %0d + %0d", who, addr, bytes);
    endtask

    // ---- readiness: requester i = 0 V read, 1 V write, 2 M, 3 X, 4 L, 5 S
    localparam NP = 6;
    reg [NP-1:0] rdy = {NP{1'b1}};
    reg [NP-1:0] pv_arb = 0;           // the requests the arbiter decided on (checked again at the edge)
    assign v_rd_ready = rdy[0]; assign v_wr_ready = rdy[1]; assign m_ready = rdy[2];
    assign x_ready = rdy[3]; assign l_ready = rdy[4]; assign s_ready = rdy[5];
    wire m_any = mw_rd_valid || ma_rd_valid || me_rd_valid || md_rd_valid != 0 || mo_wr_valid != 0;
    wire [NP-1:0] pv = {s_rd_valid || s_wr_valid, l_rd_valid || l_wr_valid, x_rd_valid, m_any, v_wr_valid, v_rd_valid};

    // ---- bank demand statistics (unchanged from note 33): each clock, the distinct lines every port
    // touches, grouped by bank, as if every request were served in the clock it is made.
    localparam CAP = 64, LB = $clog2(NB);
    function integer bank_of(input integer line);
        bank_of = SKEW ? (line + (line >> LB) + (line >> (2 * LB))) % NB : line % NB;
    endfunction
    // lost[p * 8 + h]: line-clocks requester p waited on holder h (0-5 requesters, 6 R, 7 its own write-after-read order)
    longint lost [0:NP*8-1];
    // M's waits by which of its ports were active: {weights, loads, drain reads, drain writes}
    longint m_wait_kind [0:15];
    initial begin
        for (integer i = 0; i < NP*8; i = i + 1) lost[i] = 0;
        for (integer i = 0; i < 16; i = i + 1) m_wait_kind[i] = 0;
    end
    reg [16:0] bl [0:NB-1][0:CAP-1]; integer bn [0:NB-1];
    longint busy_clocks = 0, bank_clocks = 0, lines_total = 0;
    longint stall_clocks [0:NP-1];
    longint any_stall = 0, arb_clocks = 0;
    initial for (integer i = 0; i < NP; i = i + 1) stall_clocks[i] = 0;
    integer b_, mx, q_;
    task add_line(input integer addr);
        integer line, bank, t, hit;
        begin
            line = addr >> 6; bank = bank_of(line); hit = 0;
            for (t = 0; t < bn[bank] && t < CAP; t = t + 1) if (bl[bank][t] == line / NB) hit = 1;
            if (!hit) begin
                if (bn[bank] < CAP) bl[bank][bn[bank]] = line / NB;
                bn[bank] = bn[bank] + 1; lines_total = lines_total + 1;
            end
        end
    endtask
    task add_range(input integer addr, input integer bytes);
        integer x;
        for (x = addr & ~63; x < addr + bytes; x = x + 64) add_line(x);
    endtask

`ifdef VERILATOR
    // ---- the arbiter's state. Keys are line * 2 + (1 for a write).
    bit req [NP][int];                 // this clock's request of each requester
    bit srv [NP][int];                 // lines of the current request already served
    longint sig [NP], psig [NP];       // a hash of each request (every lane's address and size), to see it change
    bit pvalid [NP];
    int rr = 0;
    int wbq [$];                       // the write buffer: V write lines waiting for a free bank (WB lines)
    int bank_st [NB];                  // this clock: -1 free, -2 taken by a write, else the line being read
    int bank_own [NB];                 // who took it: 0-5 a requester, 6 R
    task automatic rq(input int p, input int addr, input int bytes, input int w);
        int x;
        sig[p] = sig[p] * 1000003 + addr * 16 + bytes * 2 + w + 1;
        for (x = addr & ~63; x < addr + bytes; x += 64) if (x < SIZE) req[p][(x >> 6) * 2 + w] = 1;
    endtask
    // a masked write of up to PB bytes: only lines with a byte in the mask
    task automatic rq_mask(input int p, input int addr, input logic [PB-1:0] mask, input int bytes);
        int x, b, any;
        sig[p] = sig[p] * 1000003 + addr * 16 + 1;
        for (x = addr & ~63; x < addr + bytes; x += 64) begin
            any = 0;
            for (b = (x < addr ? addr : x); b < x + 64 && b < addr + bytes; b++) if (mask[b - addr]) any = 1;
            if (any && x < SIZE) req[p][(x >> 6) * 2 + 1] = 1;
        end
    endtask
    task automatic arbitrate();
        int p, i, line, b, w, allr, all, key;
        for (p = 0; p < NP; p++) begin req[p].delete(); sig[p] = 0; end
        if (v_rd_valid) for (s = 0; s < 3; s++) for (j = 0; j < LANES; j++)
            if (v_rd_mask[s][j]) rq(0, v_rd_addr[s][j], size_of(v_rd_type[s][j]), 0);
        if (v_wr_valid) for (j = 0; j < LANES; j++) if (v_wr_mask[j]) rq(1, v_wr_addr[j], size_of(v_wr_type[j]), 1);
        if (mw_rd_valid) rq(2, mw_rd_addr, 1024, 0);
        if (ma_rd_valid) rq(2, ma_rd_addr, 1024, 0);
        if (me_rd_valid) rq(2, me_rd_addr, 64, 0);
        for (q_ = 0; q_ < ER; q_++) begin
            if (md_rd_valid[q_]) rq(2, md_rd_addr[23*q_+:23], 64, 0);
            if (mo_wr_valid[q_]) rq_mask(2, mo_wr_addr[23*q_+:23], {{(PB-64){1'b0}}, mo_wr_mask[64*q_+:64]}, 64);
        end
        if (x_rd_valid) rq(3, x_rd_addr, PB, 0);
        if (l_rd_valid) rq(4, l_rd_addr, PB, 0);
        if (l_wr_valid) rq_mask(4, l_wr_addr, l_wr_mask, PB);
        if (s_rd_valid) rq(5, s_rd_addr, 4, 0);
        if (s_wr_valid) rq(5, s_wr_addr, 4, 1);
        // a new request (none last clock, the last one finished, or it changed) forgets what was served
        for (p = 0; p < NP; p++) begin
            if (!pvalid[p] || rdy[p] || sig[p] != psig[p]) srv[p].delete();
            pvalid[p] = pv[p]; psig[p] = sig[p];
        end
        for (b = 0; b < NB; b++) begin bank_st[b] = -1; bank_own[b] = -1; end
        // R first: its lines always fall in distinct banks
        if (r_wr_valid)
            for (int x = r_wr_addr & ~63; x < r_wr_addr + PB; x += 64) begin
                int any = 0;
                for (int y = (x < r_wr_addr ? r_wr_addr : x); y < x + 64 && y < r_wr_addr + PB; y++)
                    if (r_wr_mask[y - r_wr_addr]) any = 1;
                if (any && x < SIZE) begin bank_st[bank_of(x >> 6)] = -2; bank_own[bank_of(x >> 6)] = 6; end
            end
        // then the six requesters, first pick rotating
        for (i = 0; i < NP; i++) begin
            p = (rr + i) % NP;
            if (!pv[p]) begin rdy[p] = 1; continue; end
            // with the write buffer, V write needs only room for its lines; they take banks later
            if (p == 1 && WB != 0) begin
                int need = 0;
                foreach (req[1][key]) need++;
                rdy[1] = wbq.size() + need <= WB;
                if (!rdy[1]) stall_clocks[1] = stall_clocks[1] + 1;
                continue;
            end
            allr = 1;
            foreach (req[p][key]) if (!(key & 1) && !srv[p].exists(key)) begin
                line = key >> 1; b = bank_of(line);
                if (bank_st[b] == -1) begin bank_st[b] = line; bank_own[b] = p; srv[p][key] = 1; end
                else if (bank_st[b] == line) srv[p][key] = 1;                   // the same line, read: shared
                else begin allr = 0; lost[p * 8 + bank_own[b]] += 1; end
            end
            all = allr;
            foreach (req[p][key]) if ((key & 1) && !srv[p].exists(key)) begin
                if (!allr) begin all = 0; lost[p * 8 + 7] += 1; continue; end
                line = key >> 1; b = bank_of(line);
                if (bank_st[b] == -1) begin bank_st[b] = -2; bank_own[b] = p; srv[p][key] = 1; end
                else begin all = 0; lost[p * 8 + bank_own[b]] += 1; end
            end
            rdy[p] = all;
            if (!all) begin
                stall_clocks[p] = stall_clocks[p] + 1;
                if (p == 2) m_wait_kind[{mw_rd_valid, ma_rd_valid || me_rd_valid, md_rd_valid != 0, mo_wr_valid != 0}] += 1;
            end
        end
        // the vector unit moves only when its read and its write both complete in one clock (its pipeline
        // freezes otherwise), so they become ready together; a half already served keeps its lines meanwhile
        // (it is forgotten only once ready, at the top) instead of finishing alone and starting over
        if (pv[0] && pv[1] && !(rdy[0] && rdy[1])) begin
            if (rdy[0]) stall_clocks[0] = stall_clocks[0] + 1;
            if (rdy[1]) stall_clocks[1] = stall_clocks[1] + 1;
            rdy[0] = 0; rdy[1] = 0;
        end
        // the write buffer: a V write that went ready leaves its lines, and buffered lines drain into banks
        // nothing else took this clock (reads first; oldest lines first). The functional write lands at ready,
        // so a read of a buffered line sees the new bytes (in silicon: the buffer forwards them).
        if (WB != 0) begin
            int k2;
            if (pv[1] && rdy[1]) foreach (req[1][key]) wbq.push_back(key >> 1);
            k2 = 0;
            while (k2 < wbq.size()) begin
                b = bank_of(wbq[k2]);
                if (bank_st[b] == -1) begin bank_st[b] = -2; bank_own[b] = 1; wbq.delete(k2); end
                else k2 = k2 + 1;
            end
        end
        rr = (rr + 1) % NP;
        if ((pv & ~rdy) != 0) any_stall = any_stall + 1;
    endtask
`endif

    always @(negedge clk) begin
        // demand statistics
        for (b_ = 0; b_ < NB; b_ = b_ + 1) bn[b_] = 0;
        if (v_rd_valid) for (s = 0; s < 3; s = s + 1) for (j = 0; j < LANES; j = j + 1)
            if (v_rd_mask[s][j]) add_line(v_rd_addr[s][j]);
        if (v_wr_valid) for (j = 0; j < LANES; j = j + 1) if (v_wr_mask[j]) add_line(v_wr_addr[j]);
        if (mw_rd_valid) add_range(mw_rd_addr, 1024);
        if (ma_rd_valid) add_range(ma_rd_addr, 1024);
        if (me_rd_valid) add_range(me_rd_addr, 64);
        for (q_ = 0; q_ < ER; q_ = q_ + 1) begin
            if (md_rd_valid[q_]) add_range(md_rd_addr[23*q_+:23], 64);
            if (mo_wr_valid[q_]) add_range(mo_wr_addr[23*q_+:23], 64);
        end
        if (r_wr_valid) add_range(r_wr_addr, PB);
        if (x_rd_valid) add_range(x_rd_addr, PB);
        if (l_rd_valid) add_range(l_rd_addr, PB);
        if (l_wr_valid) add_range(l_wr_addr, PB);
        if (s_rd_valid) add_line(s_rd_addr);
        if (s_wr_valid) add_line(s_wr_addr);
        mx = 0; for (b_ = 0; b_ < NB; b_ = b_ + 1) if (bn[b_] > mx) mx = bn[b_];
        if (mx > 0) begin busy_clocks = busy_clocks + 1; bank_clocks = bank_clocks + mx; end
`ifdef VERILATOR
        if (ARB) arbitrate();
        pv_arb = pv;
`endif
        // reads, in the clock each requester is ready
        if (v_rd_valid && rdy[0])
            for (s = 0; s < 3; s = s + 1)
                for (j = 0; j < LANES; j = j + 1)
                    if (v_rd_mask[s][j]) begin
                        a = v_rd_addr[s][j]; n = size_of(v_rd_type[s][j]); check(a, n, "V read");
                        v_rd_data[s][j] = 0;
                        for (k = 0; k < n; k = k + 1) v_rd_data[s][j][8*k+:8] = mem[a+k];
                    end
        if (rdy[2]) begin
            if (mw_rd_valid) begin a = mw_rd_addr; for (k = 0; k < 1024; k = k + 1) mw_rd_data[8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0; end
            if (ma_rd_valid) begin a = ma_rd_addr; for (k = 0; k < 1024; k = k + 1) ma_rd_data[8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0; end
            if (me_rd_valid) begin a = me_rd_addr; for (k = 0; k < 64; k = k + 1) me_rd_data[8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0; end
            for (j = 0; j < ER; j = j + 1)
                if (md_rd_valid[j]) begin
                    a = md_rd_addr[23*j+:23];
                    for (k = 0; k < 64; k = k + 1) md_rd_data[512*j+8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0;
                end
        end
        if (x_rd_valid && rdy[3]) begin a = x_rd_addr; for (k = 0; k < PB; k = k + 1) x_rd_data[8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0; end
        if (l_rd_valid && rdy[4]) begin a = l_rd_addr; for (k = 0; k < PB; k = k + 1) l_rd_data[8*k+:8] = a + k < SIZE ? mem[a+k] : 8'd0; end
        if (s_rd_valid && rdy[5]) begin
            a = s_rd_addr; check(a, 4, "LW");
            for (k = 0; k < 4; k = k + 1) s_rd_data[8*k+:8] = mem[a+k];
        end
    end

    task line_write(input [22:0] addr, input [8*PB-1:0] data, input [PB-1:0] mask, input [8*8-1:0] who);
        for (k = 0; k < PB; k = k + 1)
            if (mask[k]) begin
                if (addr + k >= SIZE) $fatal(1, "SRAM %0s write out of range: %0d", who, addr + k);
                mem[addr+k] = data[8*k+:8];
            end
    endtask
    always @(posedge clk) begin
        // a request must not depend on its own ready: the arbiter decided from the requests it saw at the
        // falling edge, and a requester that wasn't asking was ready by default
        if (ARB && pv != pv_arb)
            $fatal(1, "SRAM: a request changed after arbitration (requesters %b then %b): it depends on its ready", pv_arb, pv);
        if (v_wr_valid && rdy[1])
            for (j = 0; j < LANES; j = j + 1)
                if (v_wr_mask[j]) begin
                    a = v_wr_addr[j]; n = size_of(v_wr_type[j]); check(a, n, "V write");
                    for (k = 0; k < n; k = k + 1) mem[a+k] = v_wr_data[j][8*k+:8];
                end
        if (rdy[2])
            for (j = 0; j < ER; j = j + 1)
                if (mo_wr_valid[j]) line_write(mo_wr_addr[23*j+:23], {{(8*PB-512){1'b0}}, mo_wr_data[512*j+:512]},
                                               {{(PB-64){1'b0}}, mo_wr_mask[64*j+:64]}, "M");
        if (r_wr_valid) line_write(r_wr_addr, r_wr_data, r_wr_mask, "R");
        if (l_wr_valid && rdy[4]) line_write(l_wr_addr, l_wr_data, l_wr_mask, "L");
        if (s_wr_valid && rdy[5]) begin
            a = s_wr_addr; check(a, 4, "SW");
            for (k = 0; k < 4; k = k + 1) mem[a+k] = s_wr_data[8*k+:8];
        end
    end
endmodule
