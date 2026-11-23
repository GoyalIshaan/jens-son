// The V queue's unit: decodes a 64-byte Vec descriptor plus the command's five registers into the decoded
// vector unit's ports (rtl/vec/vec_unit.sv) and holds them stable until done.
//
// Register meanings (isa/fsim_v1.py): r0-r3 offset the dst, a, b, c bases; r4 overrides n, m or p.
//   CAUSAL: r2 = position base, r3 = limit (not offsets).   DN: r1 = first value head, r4 = replay count
//   (a and b are absolute). SMTILE: r0 also offsets the exponent output (dst + e_off).
`include "desc_fields.vh"
module chip_vunit #(parameter LANES = 512)(
    input wire clk, reset,
    input wire start, input wire [4:0] op, input wire [511:0] desc, input wire [47:0] rv0, rv1, rv2, rv3, rv4,
    output wire busy, output reg done, output reg fault,
    output wire rd_valid, input wire rd_ready,
    output wire [22:0] rd_addr[0:2][0:LANES-1], output wire [2:0] rd_type[0:2][0:LANES-1],
    output wire [LANES-1:0] rd_mask[0:2], input wire [31:0] rd_data[0:2][0:LANES-1],
    output wire wr_valid, input wire wr_ready,
    output wire [LANES-1:0] wr_mask, output wire [22:0] wr_addr[0:LANES-1],
    output wire [31:0] wr_data[0:LANES-1], output wire [2:0] wr_type[0:LANES-1], output wire [LANES-1:0] wr_raw
);
    localparam CAUSAL = 20, DN = 22, SMTILE = 24, QUANT = 16, ROPE = 19, CONV4 = 21;
    reg [511:0] d; reg [4:0] o; reg [47:0] r0, r1, r2, r3, r4;
    reg [1:0] st;                                      // 0 idle, 1 launch, 2 run, 3 finish
    assign busy = st != 0;
    wire [95:0] par = d[`VEC_PAR];
    wire [1:0] ovr = d[`VEC_OVR];
    wire ov = r4 != 0;
    wire [15:0] n = ovr == 1 && ov ? r4[15:0] : d[`VEC_N];
    wire [15:0] m = ovr == 2 && ov ? r4[15:0] : d[`VEC_M];
    wire [15:0] p = ovr == 3 && ov ? r4[15:0] : d[`VEC_P];
    wire [22:0] base [0:3]; wire signed [19:0] sn [0:3], sm [0:3], sp [0:3]; wire [2:0] dtype [0:3];
    wire [22:0] off [0:3];
    assign off[0] = r0[22:0];
    assign off[1] = o == DN ? 23'd0 : r1[22:0];
    assign off[2] = o == DN || o == CAUSAL ? 23'd0 : r2[22:0];
    assign off[3] = o == CAUSAL ? 23'd0 : r3[22:0];
    assign base[0] = d[`VEC_D_BASE] + off[0]; assign base[1] = d[`VEC_A_BASE] + off[1];
    assign base[2] = d[`VEC_B_BASE] + off[2]; assign base[3] = d[`VEC_C_BASE] + off[3];
    assign sn[0] = d[`VEC_D_SN]; assign sm[0] = d[`VEC_D_SM]; assign sp[0] = d[`VEC_D_SP]; assign dtype[0] = d[`VEC_D_DT];
    assign sn[1] = d[`VEC_A_SN]; assign sm[1] = d[`VEC_A_SM]; assign sp[1] = d[`VEC_A_SP]; assign dtype[1] = d[`VEC_A_DT];
    assign sn[2] = d[`VEC_B_SN]; assign sm[2] = d[`VEC_B_SM]; assign sp[2] = d[`VEC_B_SP]; assign dtype[2] = d[`VEC_B_DT];
    assign sn[3] = d[`VEC_C_SN]; assign sm[3] = d[`VEC_C_SM]; assign sp[3] = d[`VEC_C_SP]; assign dtype[3] = d[`VEC_C_DT];
    wire [15:0] lv = par[`PAR_DN_LV], lqk = par[`PAR_DN_LQK];
    wire [31:0] step = par[`PAR_CAUSAL_STEP_ON] ? par[`PAR_CAUSAL_STEP] : 32'd1;
    wire v_ready, v_done, v_fault;
    reg v_start;
    vec_unit #(.LANES(LANES)) vu(.clk(clk), .reset(reset), .start(v_start), .ready(v_ready), .done(v_done), .fault(v_fault),
        .op(o), .n(n), .m(m), .p(p), .base(base), .sn(sn), .sm(sm), .sp(sp), .dtype(dtype),
        .has_b(d[`VEC_B_ON]), .has_c(d[`VEC_C_ON]), .negate(d[`VEC_NEG]),
        .exp_bias(o == QUANT ? par[`PAR_QUANT_EXP_BIAS] : 10'sd0), .rot(o == ROPE ? par[`PAR_ROPE_ROT] : 16'd0),
        .position(r2[31:0]), .limit(r3[31:0]), .causal_step(o == CAUSAL ? step : 32'd1),
        .save_window(o == CONV4 && par[`PAR_CONV4_WIN_ON]), .window_base(par[`PAR_CONV4_WINDOW]),
        .head(r1[15:0]), .key_head(r1[15:0] >> par[`PAR_DN_KSHIFT]), .key_heads(lqk), .value_heads(lv),
        .replay_count(r4[15:0]), .token_count(par[`PAR_DN_T]),
        .dn_extra_heads(o == DN ? par[`PAR_DN_HEADS_M1] : 2'd0), .commit_spec(par[`PAR_DN_COMMIT]),
        .work_base(par[`PAR_DN_WORK]),
        .vs_offset(o == SMTILE ? par[`PAR_SMTILE_VS] : 23'd0), .e_offset(o == SMTILE ? par[`PAR_SMTILE_E_OFF] : 23'd0),
        .e_stride(o == SMTILE ? par[`PAR_SMTILE_E_STRIDE] : 23'd0), .mlb(o == SMTILE ? par[`PAR_SMTILE_MLB] : 23'd0),
        .rd_valid(rd_valid), .rd_ready(rd_ready), .rd_addr(rd_addr), .rd_type(rd_type), .rd_mask(rd_mask),
        .rd_data(rd_data), .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_mask(wr_mask), .wr_addr(wr_addr),
        .wr_data(wr_data), .wr_type(wr_type), .wr_raw(wr_raw), .wr_visible(1'b1));
    always @(posedge clk) begin
        done <= 0; fault <= 0; v_start <= 0;
        if (reset) st <= 0;
        else case (st)
            0: if (start) begin
                d <= desc; o <= op; r0 <= rv0; r1 <= rv1; r2 <= rv2; r3 <= rv3; r4 <= rv4;
                st <= op == 0 ? 3 : 1;
            end
            1: if (v_ready) begin v_start <= 1; st <= 2; end
            2: if (v_done) begin done <= 1; st <= 0; end
               else if (v_fault) begin fault <= 1; done <= 1; st <= 0; end
            3: begin done <= 1; st <= 0; end
        endcase
    end
endmodule
