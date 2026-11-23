// R4 (golden/bitexact_vec.py dn_token), scheduled by dependencies on a shared 512-lane FMA bank.
//
// Same operations and rounding as the rule, in the same order within every chain. What changed from the
// first version (one op in flight per head, 8-cycle wait after each) is only WHEN ops issue:
//   pass 1, per row group g (4 rows x 128 columns = 512 lanes):
//     SK_g  = fma(S_g, k_row, sk)        the S.k chain (32 dependent steps; reads S_g from the source bank)
//     SQ_g  = fma(S_g, q_row, sq)        the S.q chain, from the saved copy of S_g, in the gaps of SK
//     DS_g  = d * S_g                    written back to the destination bank in place of S_g
//   kq  = R1 tree of k*q (7 levels, independent of pass 1)
//   K chain: pair, sum, d*sk, v - that, * beta = delta;  Q chain: pair, sum, d*sq;  OUT = fma(kq, delta, d*sq)
//   pass 2, per group: UP_g = fma(k_row, delta, DS_g) (reads DS_g back from the destination bank) -> store
// UP ops are independent, so they issue back to back. d*S is computed while S_g is being read anyway, which
// removes the second state read of the old schedule and the need to hold a scaled copy.
//
// Interfaces: one bank issue per clock per context (a group arbitrates among four). An op that reads state
// issues only in a cycle the read is accepted. Results return LAT clocks later into this context's registers
// or its write queue (two entries). DS and UP issue only with a free queue credit, so SRAM write stalls never
// lose a result. Write order within a head follows issue order.
module vec_dn_context #(parameter DIM=128, parameter LOCAL_BANK=1)(
    input wire clk,reset,start,commit_spec,
    input wire [15:0] replay_count,token_count,
    output reg replay, output reg [15:0] token_index,
    input wire [32*DIM-1:0] k,q,v,
    input wire [31:0] beta,decay,
    // issue / state read
    output wire req, output wire req_read, output wire req_read_dst,
    output wire [$clog2(DIM/4)-1:0] req_group,
    input wire grant, input wire read_ready,
    input wire [4*DIM*32-1:0] state_in,
    output wire [(4*DIM)*32-1:0] fa_bus,fb_bus,fc_bus,
    input wire [(4*DIM)*32-1:0] bank_result,
    // state write queue
    output wire wq_valid, output wire [$clog2(DIM/4)-1:0] wq_group, output wire [4*DIM*32-1:0] wq_data,
    input wire wq_pop,
    // legacy bank selection (standalone streaming of several tokens)
    output wire read_work, output wire write_work,
    output wire [DIM*32-1:0] head_out,
    output reg out_valid, output reg [15:0] out_index, output reg done, output wire busy
);
    localparam N=4*DIM, GROUPS=DIM/4, GB=$clog2(DIM/4), LAT=8, KQL=$clog2(DIM), KQ_DONE=KQL+1;
    // op codes
    localparam O_KQM=1,O_KQS=2,O_SK=3,O_SQ=4,O_DS=5,O_PK=6,O_SUMK=7,O_DK=8,O_DIFF=9,O_DELTA=10,
               O_PQ=11,O_SUMQ=12,O_DQ=13,O_OUT=14,O_UP=15;
    reg running;
    // pass 1
    reg [GB:0] sk_next, ds_next, sq_next;          // next group to issue (GROUPS = finished issuing)
    reg sk_busy, sq_busy;
    reg [GB:0] sk_ret;                              // SK results returned so far
    reg [GB:0] sq_ret;
    reg saved_valid; reg [GB-1:0] saved_g; reg saved_sq, saved_ds; // which of SQ/DS still need the copy
    reg [N*32-1:0] saved;
    reg [N*32-1:0] sk, sq;
    // kq tree
    reg [3:0] kq_stage;                             // 0: MUL due, 1..KQL: tree level due, KQ_DONE: done
    reg kq_busy; reg [DIM*32-1:0] dot_tree; reg [31:0] kq;
    // post chains
    reg [2:0] kc; reg kc_busy;                      // 0 PK,1 SUMK,2 DK,3 DIFF,4 DELTA,5 done
    reg [1:0] qc; reg qc_busy;                      // 0 PQ,1 SUMQ,2 DQ,3 done
    reg out_due, out_busy, out_done;
    reg [DIM*32-1:0] delta, output_row;
    // pass 2
    reg [GB:0] up_next, stored;
    reg [GB:0] ds_stored;
    // write queue (2 entries) and credits
    reg [N*32-1:0] wq_d[0:1]; reg [GB-1:0] wq_g[0:1];
    reg [1:0] wq_count; reg wq_head;
    reg [3:0] inflight_w;
    wire credit = (wq_count + inflight_w) < 2;
    // return delay line
    reg [LAT-1:0] rv; reg [3:0] rop[0:LAT-1]; reg [GB-1:0] rgrp[0:LAT-1];
    wire ret = rv[LAT-1]; wire [3:0] ret_op = rop[LAT-1]; wire [GB-1:0] ret_g = rgrp[LAT-1];

    wire tentative = !replay && !commit_spec;
    assign read_work  = req_read_dst ? tentative : (tentative && token_index != 0);
    assign write_work = tentative;
    assign busy = running;

    // ------------------------------------------------------------------ choose the op to request
    wire delta_ready = kc == 5;
    wire q_ready = replay || qc == 3;
    reg [3:0] want; reg [GB-1:0] want_g;
    always @* begin
        want = 0; want_g = 0;
        if (running) begin
            if (sk_next < GROUPS && !sk_busy && !saved_valid) begin want = O_SK; want_g = sk_next[GB-1:0]; end
            else if (sk_ret == GROUPS && kc < 5 && !kc_busy) want = kc == 0 ? O_PK : kc == 1 ? O_SUMK : kc == 2 ? O_DK : kc == 3 ? O_DIFF : O_DELTA;
            else if (saved_valid && saved_sq && !sq_busy) begin want = O_SQ; want_g = saved_g; end
            else if (saved_valid && saved_ds && credit) begin want = O_DS; want_g = saved_g; end
            else if (!replay && kq_stage < KQ_DONE && !kq_busy) want = kq_stage == 0 ? O_KQM : O_KQS;
            else if (!replay && sq_ret == GROUPS && qc < 3 && !qc_busy) want = qc == 0 ? O_PQ : qc == 1 ? O_SUMQ : O_DQ;
            else if (!replay && out_due && !out_busy && delta_ready && qc == 3 && kq_stage == KQ_DONE) want = O_OUT;
            else if (delta_ready && ds_stored == GROUPS && up_next < GROUPS && credit) begin want = O_UP; want_g = up_next[GB-1:0]; end
        end
    end
    // a stalled read keeps its op until accepted
    reg pend; reg [3:0] pend_op; reg [GB-1:0] pend_g;
    wire [3:0] op = pend ? pend_op : want;
    wire [GB-1:0] og = pend ? pend_g : want_g;
    assign req = op != 0;
    assign req_read = op == O_SK || op == O_UP;
    assign req_read_dst = op == O_UP;
    assign req_group = og;
    wire fire = grant && req && (!req_read || read_ready);

    // ------------------------------------------------------------------ operands (lane i: row R, column J)
    wire [N*32-1:0] fy;
    genvar i;
    for (i = 0; i < N; i = i + 1) begin : lane
        localparam R = i / DIM, J = i % DIM;
        localparam A = (2 * R) % 4, B = (2 * R + 1) % 4;
        wire [31:0] krow = k[32 * (4 * og + R) +: 32];
        wire [31:0] qrow = q[32 * (4 * og + R) +: 32];
        wire [31:0] left = dot_tree[32 * ((2 * J) % DIM) +: 32];
        wire [31:0] right = dot_tree[32 * ((2 * J + 1) % DIM) +: 32];
        // Narrow each source before the operand mux. Pass the lane words to
        // a combinational function so Icarus need not infer a wide sensitivity
        // list for each of the 512 lanes in each context.
        wire [31:0] kcol = k[32 * J +: 32], qcol = q[32 * J +: 32], vcol = v[32 * J +: 32];
        wire [31:0] read_lane = state_in[32 * i +: 32], saved_lane = saved[32 * i +: 32];
        wire [31:0] sk_lane = sk[32 * i +: 32], sq_lane = sq[32 * i +: 32];
        wire [31:0] sk_even = sk[32 * (A * DIM + J) +: 32], sk_odd = sk[32 * (B * DIM + J) +: 32];
        wire [31:0] sq_even = sq[32 * (A * DIM + J) +: 32], sq_odd = sq[32 * (B * DIM + J) +: 32];
        wire [31:0] sk0 = sk[32 * J +: 32], sk1 = sk[32 * (DIM + J) +: 32];
        wire [31:0] sq0 = sq[32 * J +: 32], sq1 = sq[32 * (DIM + J) +: 32];
        wire [31:0] delta_col = delta[32 * J +: 32];
        function automatic [95:0] operands(
            input [3:0] issue_op, input first_group,
            input [31:0] kcol, qcol, vcol, left, right, krow, qrow, read_lane, saved_lane,
            input [31:0] sk_lane, sq_lane, sk_even, sk_odd, sq_even, sq_odd, sk0, sk1, sq0, sq1,
            input [31:0] delta_col, beta, decay, kq
        );
            reg [31:0] a, b, c;
            a = 0; b = 32'h3f800000; c = 0;
            case (issue_op)
                O_KQM:  begin a = kcol; b = qcol; end
                O_KQS:  begin a = left; c = right; end
                O_SK:   begin a = read_lane; b = krow; c = first_group ? 0 : sk_lane; end
                O_SQ:   begin a = saved_lane; b = qrow; c = first_group ? 0 : sq_lane; end
                O_DS:   begin a = decay; b = saved_lane; end
                O_PK:   begin a = sk_even; c = sk_odd; end
                O_SUMK: begin a = sk0; c = sk1; end
                O_DK:   begin a = decay; b = sk0; end
                O_DIFF: begin a = vcol; c = sk0 ^ 32'h80000000; end
                O_DELTA:begin a = sk0; b = beta; end
                O_PQ:   begin a = sq_even; c = sq_odd; end
                O_SUMQ: begin a = sq0; c = sq1; end
                O_DQ:   begin a = decay; b = sq0; end
                O_OUT:  begin a = kq; b = delta_col; c = sq0; end
                O_UP:   begin a = krow; b = delta_col; c = read_lane; end
                default: ;
            endcase
            operands = {a, b, c};
        endfunction
        wire [31:0] a, b, c;
        assign {a, b, c} = operands(op, og == 0, kcol, qcol, vcol, left, right, krow, qrow,
                                    read_lane, saved_lane, sk_lane, sq_lane, sk_even, sk_odd,
                                    sq_even, sq_odd, sk0, sk1, sq0, sq1, delta_col, beta, decay, kq);
        assign fa_bus[32 * i +: 32] = a;
        assign fb_bus[32 * i +: 32] = b;
        assign fc_bus[32 * i +: 32] = c;
        if (LOCAL_BANK) fp32_fma_pipe8 madd(clk, a, b, c, fy[32 * i +: 32]);
        else assign fy[32 * i +: 32] = bank_result[32 * i +: 32];
    end

    assign wq_valid = wq_count != 0;
    assign wq_group = wq_g[wq_head];
    assign wq_data = wq_d[wq_head];
    assign head_out = output_row;

    integer s;
    wire token_end = running && stored == GROUPS && (replay || out_done) && wq_count == 0 && inflight_w == 0;
    task begin_token(input rep);
        begin
            sk_next <= 0; ds_next <= 0; sq_next <= 0; sk_busy <= 0; sq_busy <= 0; sk_ret <= 0; sq_ret <= 0;
            saved_valid <= 0; kq_stage <= rep ? KQ_DONE : 0; kq_busy <= 0; kc <= 0; kc_busy <= 0;
            qc <= rep ? 3 : 0; qc_busy <= 0; out_due <= !rep; out_busy <= 0; out_done <= rep;
            up_next <= 0; stored <= 0; ds_stored <= 0;
        end
    endtask
    always @(posedge clk) begin
        done <= 0; out_valid <= 0;
        if (reset) begin
            running <= 0; pend <= 0; rv <= 0; wq_count <= 0; wq_head <= 0; inflight_w <= 0;
        end else begin
            // ---- return pipeline
            rv <= {rv[LAT-2:0], fire};
            for (s = LAT - 1; s > 0; s = s - 1) begin rop[s] <= rop[s - 1]; rgrp[s] <= rgrp[s - 1]; end
            rop[0] <= op; rgrp[0] <= og;
            // ---- stalled read: hold the op
            if (grant && req && req_read && !read_ready) begin pend <= 1; pend_op <= op; pend_g <= og; end
            else if (fire) pend <= 0;
            // ---- issue bookkeeping
            if (fire) case (op)
                O_SK: begin sk_busy <= 1; sk_next <= sk_next + 1'b1; saved <= state_in; saved_valid <= 1;
                            saved_g <= og; saved_sq <= !replay; saved_ds <= 1; end
                O_SQ: begin sq_busy <= 1; saved_sq <= 0; if (!saved_ds) saved_valid <= 0; end
                O_DS: begin saved_ds <= 0; ds_next <= ds_next + 1'b1; if (!saved_sq) saved_valid <= 0; end
                O_KQM, O_KQS: kq_busy <= 1;
                O_PK, O_SUMK, O_DK, O_DIFF, O_DELTA: kc_busy <= 1;
                O_PQ, O_SUMQ, O_DQ: qc_busy <= 1;
                O_OUT: out_busy <= 1;
                O_UP: up_next <= up_next + 1'b1;
                default: ;
            endcase
            // ---- results
            if (ret) case (ret_op)
                O_KQM: begin dot_tree <= fy[32 * DIM - 1:0]; kq_busy <= 0; kq_stage <= 1; end
                O_KQS: begin dot_tree <= fy[32 * DIM - 1:0]; kq_busy <= 0; kq_stage <= kq_stage + 1'b1;
                             if (kq_stage == KQL) kq <= fy[31:0]; end
                O_SK: begin sk <= fy; sk_busy <= 0; sk_ret <= sk_ret + 1'b1; end
                O_SQ: begin sq <= fy; sq_busy <= 0; sq_ret <= sq_ret + 1'b1; end
                O_PK, O_SUMK, O_DK, O_DIFF: begin sk <= fy; kc_busy <= 0; kc <= kc + 1'b1; end
                O_DELTA: begin delta <= fy[32 * DIM - 1:0]; kc_busy <= 0; kc <= 5; end
                O_PQ, O_SUMQ, O_DQ: begin sq <= fy; qc_busy <= 0; qc <= qc + 1'b1; end
                O_OUT: begin output_row <= fy[32 * DIM - 1:0]; out_busy <= 0; out_done <= 1; out_due <= 0; end
                default: ;
            endcase
            // ---- write queue: push results of DS / UP, pop accepted writes
            begin : wq
                reg push, popq; reg [1:0] cnt;
                push = ret && (ret_op == O_DS || ret_op == O_UP);
                popq = wq_pop && wq_count != 0;
                if (push) begin                          // tail = head + count (mod 2); credits keep count + push <= 2
                    wq_d[wq_head ^ wq_count[0]] <= fy;
                    wq_g[wq_head ^ wq_count[0]] <= ret_g;
                end
                if (popq) wq_head <= wq_head ^ 1'b1;
                wq_count <= wq_count + push - popq;
                inflight_w <= inflight_w + (fire && (op == O_DS || op == O_UP)) - push;
                if (popq) begin
                    // DS writes come first for every group (pass 1 ends before pass 2 starts)
                    if (ds_stored < GROUPS) ds_stored <= ds_stored + 1'b1;
                    else stored <= stored + 1'b1;
                end
            end
            // ---- token sequencing
            if (!running) begin
                if (start) begin
                    replay <= replay_count != 0; token_index <= 0;
                    if (replay_count != 0) begin running <= 1; begin_token(1); end
                    else if (token_count != 0) begin running <= 1; begin_token(0); end
                    else done <= 1;
                end
            end else if (token_end) begin
                out_valid <= !replay; out_index <= token_index;
                if (token_index + 1 < (replay ? replay_count : token_count)) begin
                    token_index <= token_index + 1'b1; begin_token(replay);
                end else if (replay && token_count != 0) begin
                    token_index <= 0; replay <= 0; begin_token(0);
                end else begin done <= 1; running <= 0; end
            end
        end
    end
endmodule

// Standalone single head with its own 512 FMAs (the R4 reference bench). Reads use rd_group and the bank
// read_work selects; writes use wr_group and write_work.
module vec_dn_head #(parameter DIM=128, parameter LOCAL_BANK=1)(
    input wire clk,reset,start,commit_spec,
    input wire [15:0] replay_count,token_count,
    output wire replay, output wire [15:0] token_index,
    input wire [32*DIM-1:0] k,q,v,
    input wire [31:0] beta,decay,
    output wire [$clog2(DIM/4)-1:0] rd_group, output wire [$clog2(DIM/4)-1:0] wr_group,
    output wire read_work,write_work,
    output wire read_valid,input wire read_ready,
    input wire [4*DIM*32-1:0] state_in,
    output wire write_valid,input wire write_ready,
    output wire [4*DIM*32-1:0] state_out,
    output wire [DIM*32-1:0] head_out,
    output wire out_valid, output wire [15:0] out_index, output wire done, output wire busy
);
    wire req, rd;
    wire [(4*DIM)*32-1:0] fa, fb, fc;
    vec_dn_context #(DIM, LOCAL_BANK) ctx(.clk(clk), .reset(reset), .start(start), .commit_spec(commit_spec),
        .replay_count(replay_count), .token_count(token_count), .replay(replay), .token_index(token_index),
        .k(k), .q(q), .v(v), .beta(beta), .decay(decay),
        .req(req), .req_read(rd), .req_read_dst(), .req_group(rd_group), .grant(1'b1), .read_ready(read_ready),
        .state_in(state_in), .fa_bus(fa), .fb_bus(fb), .fc_bus(fc), .bank_result({(4*DIM*32){1'b0}}),
        .wq_valid(write_valid), .wq_group(wr_group), .wq_data(state_out), .wq_pop(write_valid && write_ready),
        .read_work(read_work), .write_work(write_work),
        .head_out(head_out), .out_valid(out_valid), .out_index(out_index), .done(done), .busy(busy));
    assign read_valid = req && rd;
endmodule
