// The whole chip, as far as ISA v1 sees it: scalar core and queues, DMA W/R/X/L, the weight FIFO, the
// matrix unit, the vector unit and the SRAM. DRAM is outside: three ports (W reads, R reads, X writes)
// that a DRAM controller (or the testbench's model) serves in order with any latency.
// The host talks to the chip on a 32-bit register bus (chip_ctrl): reset, boot (program and register loading),
// run, status, the host input FIFO (H2S) and the performance counters (chip_perf). The run pin is ORed with
// the control register's run bit, so a testbench can also load memories directly and raise the pin.
`include "desc_fields.vh"
module chip_top #(parameter SRAM_BYTES = 6815744, LANES = 512, MXU_RP = 4, MXU_MAXR = 128, MXU_ACCW = 8192, MXU_ER = 16, PB = 512,
                  SRAM_BANKS = 64, SRAM_ARB = 1, SRAM_SKEW = 0, SRAM_WB = 0)(
    input wire clk, reset, run_pin,
    output wire halted, output wire idle, output wire fault, output wire chip_reset,
    // the host register bus (chip_ctrl)
    input wire bus_valid, bus_write, input wire [15:0] bus_addr, input wire [31:0] bus_wdata,
    output wire [31:0] bus_rdata, output wire bus_rvalid,
    // DRAM: W and R reads (request, then in-order 64-byte responses), X writes
    output wire w_dr_valid, input wire w_dr_ready, output wire [35:0] w_dr_addr, output wire [9:0] w_dr_len,
    input wire w_resp_valid, input wire [8*PB-1:0] w_resp_data,
    output wire r_dr_valid, input wire r_dr_ready, output wire [35:0] r_dr_addr, output wire [9:0] r_dr_len,
    input wire r_resp_valid, input wire [8*PB-1:0] r_resp_data,
    output wire x_dw_valid, input wire x_dw_ready, output wire [35:0] x_dw_addr, output wire [8*PB-1:0] x_dw_data,
    output wire [PB-1:0] x_dw_mask,
    // host output (S2H)
    output wire h_valid, output wire [8*PB-1:0] h_data, output wire [9:0] h_len
);
    localparam W = 0, R = 1, X = 2, L = 3, M = 4, V = 5;
    wire [5:0] u_start, u_busy, u_done; wire [4:0] u_op [0:5]; wire [511:0] u_desc [0:5]; wire [47:0] u_rv [0:5][0:4];
    wire s_rd_valid, s_wr_valid, s_ready, v_rd_ready, v_wr_ready, m_ready, x_ready, l_ready; wire [22:0] s_rd_addr, s_wr_addr; wire [31:0] s_rd_data, s_wr_data;
    wire core_fault, m_fault, v_fault;
    wire [63:0] cycles, bundles;
    // ---------------------------------------------------------------- control, reset, boot, counters
    wire rst, ctl_run, ld_imem_we, ld_dmem_we, ld_reg_we, perf_clear, hi_pop; wire [15:0] ld_addr; wire [511:0] ld_data;
    wire [4:0] ld_reg; wire [47:0] ld_reg_val; wire [31:0] hi_level, pc; wire [9:0] hi_pop_len; wire [8*PB-1:0] hi_data;
    wire [5:0] q_active; wire st_wait, st_full, st_sram;
    localparam NPERF = 24; wire [48*NPERF-1:0] perf;
    wire run = run_pin || ctl_run;
    assign chip_reset = rst;
    chip_ctrl #(.PB(PB), .NPERF(NPERF)) ctrl(.clk(clk), .ext_reset(reset),
        .bus_valid(bus_valid), .bus_write(bus_write), .bus_addr(bus_addr), .bus_wdata(bus_wdata),
        .bus_rdata(bus_rdata), .bus_rvalid(bus_rvalid), .chip_reset(rst), .run(ctl_run),
        .halted(halted), .idle(idle), .fault(fault), .fault_src({1'b0, v_fault, m_fault, core_fault}), .pc(pc),
        .cycles(cycles), .bundles(bundles), .ld_imem_we(ld_imem_we), .ld_dmem_we(ld_dmem_we), .ld_addr(ld_addr),
        .ld_data(ld_data), .ld_reg_we(ld_reg_we), .ld_reg(ld_reg), .ld_reg_val(ld_reg_val),
        .hi_level(hi_level), .hi_data(hi_data), .hi_pop(hi_pop), .hi_pop_len(hi_pop_len), .perf(perf), .perf_clear(perf_clear));
    chip_core core(.clk(clk), .reset(rst), .run(run), .halted(halted), .idle(idle), .fault(core_fault),
        .u_start(u_start), .u_busy(u_busy), .u_done(u_done), .u_op(u_op), .u_desc(u_desc), .u_rv(u_rv),
        .s_rd_valid(s_rd_valid), .s_rd_addr(s_rd_addr), .s_rd_data(s_rd_data), .s_ready(s_ready),
        .s_wr_valid(s_wr_valid), .s_wr_addr(s_wr_addr), .s_wr_data(s_wr_data), .cycles(cycles), .bundles(bundles),
        .ld_imem_we(ld_imem_we), .ld_dmem_we(ld_dmem_we), .ld_addr(ld_addr), .ld_data(ld_data),
        .ld_reg_we(ld_reg_we), .ld_reg(ld_reg), .ld_reg_val(ld_reg_val),
        .q_active(q_active), .st_wait(st_wait), .st_full(st_full), .st_sram(st_sram), .pc_o(pc));
    reg sticky_fault;
    assign fault = sticky_fault;
    always @(posedge clk)
        if (rst) sticky_fault <= 0;
        else if (core_fault || m_fault || v_fault) sticky_fault <= 1;

    // ---------------------------------------------------------------- DMA: W, R, X, L
    wire [31:0] fifo_level, fifo_free; wire [8191:0] fifo_head; wire fifo_pop; wire [10:0] fifo_pop_len;
    wire fw_valid; wire [8*PB-1:0] fw_data; wire [9:0] fw_len;
    wire r_sw_valid, l_sw_valid, x_sr_valid, l_sr_valid;
    wire [22:0] r_sw_addr, l_sw_addr, x_sr_addr, l_sr_addr; wire [8*PB-1:0] r_sw_data, l_sw_data, x_sr_data, l_sr_data;
    wire [PB-1:0] r_sw_mask, l_sw_mask;
    reg [3:0] nop_done; wire [3:0] dma_done, dma_busy;
    genvar u;
    for (u = 0; u < 4; u = u + 1) begin : dma
        wire [511:0] d = u_desc[u];
        wire copy = u_start[u] && u_op[u] != 0;
        wire [23:0] n1 = u_rv[u][2] != 0 ? u_rv[u][2][23:0] : d[`DMA_N1];
        wire [23:0] n2 = u_rv[u][3] != 0 ? u_rv[u][3][23:0] : d[`DMA_N2];
        wire [35:0] src = d[`DMA_SRC] + u_rv[u][0][35:0], dst = d[`DMA_DST] + u_rv[u][1][35:0];
        // unused ports of each kind
        wire dr_v, dw_v, sr_v, sw_v, f_v, h_v; wire [35:0] dr_a, dw_a; wire [9:0] dr_l, f_l, h_l;
        wire [22:0] sr_a, sw_a; wire [8*PB-1:0] dw_d, sw_d, f_d, h_d; wire [PB-1:0] dw_m, sw_m;
        wire hp; wire [9:0] hpl;
        chip_dma #(.KIND(u), .PB(PB)) e(.clk(clk), .reset(rst), .start(copy), .to_host(u_op[u] == 2), .from_host(u == L && u_op[u] == 3),
            .src(src), .dst(dst), .n0(d[`DMA_N0]), .n1(n1), .n2(n2),
            .s1_src(d[`DMA_S1_SRC]), .s1_dst(d[`DMA_S1_DST]), .s2_src(d[`DMA_S2_SRC]), .s2_dst(d[`DMA_S2_DST]),
            .busy(dma_busy[u]), .done(dma_done[u]), .sram_ready(u == X ? x_ready : u == L ? l_ready : 1'b1),
            .dr_valid(dr_v), .dr_ready(u == W ? w_dr_ready : r_dr_ready), .dr_addr(dr_a), .dr_len(dr_l),
            .dr_resp_valid(u == W ? w_resp_valid : u == R ? r_resp_valid : 1'b0),
            .dr_resp_data(u == W ? w_resp_data : r_resp_data),
            .dw_valid(dw_v), .dw_ready(x_dw_ready), .dw_addr(dw_a), .dw_data(dw_d), .dw_mask(dw_m),
            .sr_valid(sr_v), .sr_addr(sr_a), .sr_data(u == X ? x_sr_data : l_sr_data),
            .sw_valid(sw_v), .sw_addr(sw_a), .sw_data(sw_d), .sw_mask(sw_m),
            .fw_valid(f_v), .fw_data(f_d), .fw_len(f_l), .fifo_free(fifo_free),
            .h_valid(h_v), .h_data(h_d), .h_len(h_l),
            .hi_level(u == L ? hi_level : 32'd0), .hi_data(hi_data), .hi_pop(hp), .hi_pop_len(hpl));
        if (u == W) begin : wp
            assign w_dr_valid = dr_v; assign w_dr_addr = dr_a; assign w_dr_len = dr_l;
            assign fw_valid = f_v; assign fw_data = f_d; assign fw_len = f_l;
        end
        if (u == R) begin : rp
            assign r_dr_valid = dr_v; assign r_dr_addr = dr_a; assign r_dr_len = dr_l;
            assign r_sw_valid = sw_v; assign r_sw_addr = sw_a; assign r_sw_data = sw_d; assign r_sw_mask = sw_m;
        end
        if (u == X) begin : xp
            assign x_dw_valid = dw_v; assign x_dw_addr = dw_a; assign x_dw_data = dw_d; assign x_dw_mask = dw_m;
            assign x_sr_valid = sr_v; assign x_sr_addr = sr_a;
        end
        if (u == L) begin : lp
            assign l_sr_valid = sr_v; assign l_sr_addr = sr_a;
            assign l_sw_valid = sw_v; assign l_sw_addr = sw_a; assign l_sw_data = sw_d; assign l_sw_mask = sw_m;
            assign h_valid = h_v; assign h_data = h_d; assign h_len = h_l;
            assign hi_pop = hp; assign hi_pop_len = hpl;
        end
        assign u_busy[u] = dma_busy[u];
        assign u_done[u] = dma_done[u] || nop_done[u];
    end
    always @(posedge clk)
        for (integer k = 0; k < 4; k = k + 1) nop_done[k] <= !rst && u_start[k] && u_op[k] == 0;
    chip_wfifo #(.PB(PB)) fifo(.clk(clk), .reset(rst), .wr_valid(fw_valid), .wr_data(fw_data), .wr_len(fw_len),
        .pop(fifo_pop), .pop_len(fifo_pop_len), .level(fifo_level), .free(fifo_free), .head(fifo_head));

    // ---------------------------------------------------------------- matrix unit
    wire [511:0] md = u_desc[M];
    wire [1:0] movr = md[`MM_OVR]; wire [47:0] mr4 = u_rv[M][4];
    localparam ER = MXU_ER;
    wire m_busy, m_done, mw_valid, ma_valid, me_valid; wire [22:0] mw_addr, ma_addr, me_addr;
    wire [8191:0] mw_data, ma_data; wire [511:0] me_data;
    wire [ER-1:0] md_valid, mo_valid; wire [23*ER-1:0] md_addr, mo_addr; wire [512*ER-1:0] md_data, mo_data;
    wire [64*ER-1:0] mo_mask;
    chip_mxu #(.RP(MXU_RP), .MAXR(MXU_MAXR), .ACCW(MXU_ACCW), .ER(ER)) mxu(.clk(clk), .reset(rst), .ce(m_ready), .start(u_start[M]), .op(u_op[M][1:0]),
        .n_in(movr == 1 && mr4 != 0 ? mr4[19:0] : md[`MM_N_IN]), .n_out(movr == 2 && mr4 != 0 ? mr4[19:0] : md[`MM_N_OUT]),
        .rows(md[`MM_ROWS]), .act(md[`MM_ACT] + u_rv[M][0][22:0]), .exps(md[`MM_EXPS] + u_rv[M][1][22:0]),
        .dst(md[`MM_DST] + u_rv[M][2][22:0]), .wsrc(md[`MM_WSRC] + u_rv[M][3][22:0]),
        .act_stride(md[`MM_ACT_STRIDE]), .exps_stride(md[`MM_EXPS_STRIDE]), .dst_stride(md[`MM_DST_STRIDE]),
        .wfmt_int8(md[`MM_WFMT]), .seg4(md[`MM_SEG4]), .epi(md[`MM_EPI]), .id_base(md[`MM_ID_BASE]),
        .tscale(md[`MM_TSCALE]), .busy(m_busy), .done(m_done), .fault(m_fault),
        .fifo_level(fifo_level), .fifo_head(fifo_head), .fifo_pop(fifo_pop), .fifo_pop_len(fifo_pop_len),
        .w_rd_valid(mw_valid), .w_rd_addr(mw_addr), .w_rd_data(mw_data),
        .a_rd_valid(ma_valid), .a_rd_addr(ma_addr), .a_rd_data(ma_data),
        .e_rd_valid(me_valid), .e_rd_addr(me_addr), .e_rd_data(me_data),
        .d_rd_valid(md_valid), .d_rd_addr(md_addr), .d_rd_data(md_data),
        .o_wr_valid(mo_valid), .o_wr_addr(mo_addr), .o_wr_data(mo_data), .o_wr_mask(mo_mask));
    assign u_busy[M] = m_busy; assign u_done[M] = m_done || m_fault;

    // ---------------------------------------------------------------- vector unit
    wire v_busy, v_done, v_rd_valid, v_wr_valid;
    wire [22:0] v_rd_addr [0:2][0:LANES-1]; wire [2:0] v_rd_type [0:2][0:LANES-1]; wire [LANES-1:0] v_rd_mask [0:2];
    wire [31:0] v_rd_data [0:2][0:LANES-1];
    wire [LANES-1:0] v_wr_mask, v_wr_raw; wire [22:0] v_wr_addr [0:LANES-1]; wire [31:0] v_wr_data [0:LANES-1];
    wire [2:0] v_wr_type [0:LANES-1];
    chip_vunit #(.LANES(LANES)) vec(.clk(clk), .reset(rst), .start(u_start[V]), .op(u_op[V]), .desc(u_desc[V]),
        .rv0(u_rv[V][0]), .rv1(u_rv[V][1]), .rv2(u_rv[V][2]), .rv3(u_rv[V][3]), .rv4(u_rv[V][4]),
        .busy(v_busy), .done(v_done), .fault(v_fault),
        .rd_valid(v_rd_valid), .rd_ready(v_rd_ready), .rd_addr(v_rd_addr), .rd_type(v_rd_type), .rd_mask(v_rd_mask),
        .rd_data(v_rd_data), .wr_valid(v_wr_valid), .wr_ready(v_wr_ready), .wr_mask(v_wr_mask), .wr_addr(v_wr_addr),
        .wr_data(v_wr_data), .wr_type(v_wr_type), .wr_raw(v_wr_raw));
    assign u_busy[V] = v_busy; assign u_done[V] = v_done;

    // ---------------------------------------------------------------- SRAM
    chip_sram #(.SIZE(SRAM_BYTES), .LANES(LANES), .PB(PB), .ER(ER), .NB(SRAM_BANKS), .ARB(SRAM_ARB), .SKEW(SRAM_SKEW), .WB(SRAM_WB)) sram(.clk(clk),
        .v_rd_ready(v_rd_ready), .v_wr_ready(v_wr_ready), .m_ready(m_ready), .x_ready(x_ready), .l_ready(l_ready), .s_ready(s_ready),
        .v_rd_valid(v_rd_valid), .v_rd_addr(v_rd_addr), .v_rd_type(v_rd_type), .v_rd_mask(v_rd_mask), .v_rd_data(v_rd_data),
        .v_wr_valid(v_wr_valid), .v_wr_mask(v_wr_mask), .v_wr_addr(v_wr_addr), .v_wr_data(v_wr_data), .v_wr_type(v_wr_type),
        .mw_rd_valid(mw_valid), .mw_rd_addr(mw_addr), .mw_rd_data(mw_data),
        .ma_rd_valid(ma_valid), .ma_rd_addr(ma_addr), .ma_rd_data(ma_data),
        .me_rd_valid(me_valid), .me_rd_addr(me_addr), .me_rd_data(me_data),
        .md_rd_valid(md_valid), .md_rd_addr(md_addr), .md_rd_data(md_data),
        .mo_wr_valid(mo_valid), .mo_wr_addr(mo_addr), .mo_wr_data(mo_data), .mo_wr_mask(mo_mask),
        .r_wr_valid(r_sw_valid), .r_wr_addr(r_sw_addr), .r_wr_data(r_sw_data), .r_wr_mask(r_sw_mask),
        .x_rd_valid(x_sr_valid), .x_rd_addr(x_sr_addr), .x_rd_data(x_sr_data),
        .l_rd_valid(l_sr_valid), .l_rd_addr(l_sr_addr), .l_rd_data(l_sr_data),
        .l_wr_valid(l_sw_valid), .l_wr_addr(l_sw_addr), .l_wr_data(l_sw_data), .l_wr_mask(l_sw_mask),
        .s_rd_valid(s_rd_valid), .s_rd_addr(s_rd_addr), .s_rd_data(s_rd_data),
        .s_wr_valid(s_wr_valid), .s_wr_addr(s_wr_addr), .s_wr_data(s_wr_data));

    // ---------------------------------------------------------------- performance counters
    wire m_any = mw_valid || ma_valid || me_valid || md_valid != 0 || mo_valid != 0;
    chip_perf #(.NPERF(NPERF)) counters(.clk(clk), .reset(rst), .clear(perf_clear), .run(run), .idle(idle),
        .q_active(q_active), .u_done(u_done), .st_wait(st_wait), .st_full(st_full), .st_sram(st_sram),
        .v_rd_wait(v_rd_valid && !v_rd_ready), .v_wr_wait(v_wr_valid && !v_wr_ready), .m_wait(m_any && !m_ready),
        .x_wait(x_sr_valid && !x_ready), .l_wait((l_sr_valid || l_sw_valid) && !l_ready),
        .m_busy(m_busy), .fifo_level(fifo_level),
        .w_req(w_dr_valid && w_dr_ready), .w_len(w_dr_len), .r_req(r_dr_valid && r_dr_ready), .r_len(r_dr_len),
        .x_req(x_dw_valid && x_dw_ready), .x_len(10'($countones(x_dw_mask))), .count(perf));
endmodule
