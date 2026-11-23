// Four R4 contexts (vec_dn_head.sv) on the execution cluster's one 512-lane FMA bank.
// Issue: one context per clock, round robin over those with a ready op; an op that reads state issues only
// when the read is accepted, and a stalled read keeps the same context and request until it is.
// Writes: a separate round-robin arbiter drains the contexts' write queues, one 512-word row group per clock;
// a stalled write keeps its context and request. Results return to the context that issued (fixed latency).
module vec_dn_group #(parameter DIM=128)(
    input wire clk,reset,start,commit_spec,
    input wire [1:0] extra_heads,
    input wire [15:0] replay_count,token_count,
    output wire replay,output wire [15:0] token_index,
    input wire [4*DIM*32-1:0] k,q,v,input wire [127:0] beta,decay,
    // state reads (with the issue)
    output wire [1:0] slot,output wire [$clog2(DIM/4)-1:0] row_group,output wire read_dst,
    output wire read_work,write_work,state_read,input wire read_ready,
    input wire [4*DIM*32-1:0] state_in,
    // state writes
    output wire [1:0] wslot,output wire [$clog2(DIM/4)-1:0] wrow_group,
    output wire state_write,input wire write_ready,output wire [4*DIM*32-1:0] state_out,
    output wire [4*DIM*32-1:0] head_out,output wire head_valid,output wire [15:0] head_index,
    output wire done,output wire [4*DIM*32-1:0] fa,fb,fc,input wire [4*DIM*32-1:0] fy
);
    localparam W=4*DIM*32, GB=$clog2(DIM/4);
    wire [3:0] req, rd, rdst, finished, out_valid, wqv, working_src, working_dst, log_src;
    wire [15:0] token[0:3], out_idx[0:3];
    wire [GB-1:0] rg[0:3], wg[0:3];
    wire [W-1:0] a[0:3], b[0:3], c[0:3], wd[0:3];
    reg [3:0] live;
    // ---- issue arbiter
    reg [1:0] first, held, chosen; reg locked, found; integer i; reg [1:0] cand;
    always @* begin
        chosen = first; found = 0;
        for (i = 0; i < 4; i = i + 1) begin cand = first + i; if (!found && req[cand]) begin chosen = cand; found = 1; end end
    end
    assign slot = locked ? held : chosen;
    wire issue_ok = req[slot] && (!rd[slot] || read_ready);
    // ---- write arbiter
    reg [1:0] wfirst, wheld, wchosen; reg wlocked, wfound; reg [1:0] wc;
    always @* begin
        wchosen = wfirst; wfound = 0;
        for (i = 0; i < 4; i = i + 1) begin wc = wfirst + i; if (!wfound && wqv[wc]) begin wchosen = wc; wfound = 1; end end
    end
    assign wslot = wlocked ? wheld : wchosen;
    assign state_write = wqv[wslot];
    wire wpop = state_write && write_ready;
    always @(posedge clk) begin
        if (reset) begin live <= 0; first <= 0; locked <= 0; held <= 0; wfirst <= 0; wlocked <= 0; wheld <= 0; end
        else begin
            if (start) begin live <= 4'b1111 >> (3 - extra_heads); first <= 0; wfirst <= 0; end
            else live <= live & ~finished;
            if (issue_ok) begin first <= slot + 1'b1; locked <= 0; end
            else if (req[slot]) begin held <= slot; locked <= 1; end
            if (wpop) begin wfirst <= wslot + 1'b1; wlocked <= 0; end
            else if (state_write) begin wheld <= wslot; wlocked <= 1; end
        end
    end
    assign done = (|live) && ((live & ~finished) == 0);
    assign state_read = req[slot] && rd[slot];
    assign read_dst = rdst[slot];
    assign row_group = rg[slot];
    assign {fa, fb, fc} = {a[slot], b[slot], c[slot]};
    assign wrow_group = wg[wslot];
    assign state_out = wd[wslot];
    assign read_work = working_src[slot];     // the context already selects the bank for its current read
    assign write_work = working_dst[wslot];
    assign replay = log_src[slot], token_index = token[slot];
    assign head_valid = extra_heads == 0 ? out_valid[0] : done && token_count != 0;
    assign head_index = out_idx[0];
    for (genvar h = 0; h < 4; h = h + 1) begin : contexts
        localparam integer INDEX = h;
        vec_dn_context #(DIM, 0) ctx(.clk(clk), .reset(reset), .start(start && INDEX <= extra_heads),
            .commit_spec(commit_spec), .replay_count(replay_count), .token_count(token_count),
            .replay(log_src[INDEX]), .token_index(token[INDEX]),
            .k(k[INDEX*DIM*32 +: DIM*32]), .q(q[INDEX*DIM*32 +: DIM*32]), .v(v[INDEX*DIM*32 +: DIM*32]),
            .beta(beta[32*INDEX +: 32]), .decay(decay[32*INDEX +: 32]),
            .req(req[INDEX]), .req_read(rd[INDEX]), .req_read_dst(rdst[INDEX]), .req_group(rg[INDEX]),
            .grant(slot == INDEX), .read_ready(read_ready), .state_in(state_in),
            .fa_bus(a[INDEX]), .fb_bus(b[INDEX]), .fc_bus(c[INDEX]), .bank_result(fy),
            .wq_valid(wqv[INDEX]), .wq_group(wg[INDEX]), .wq_data(wd[INDEX]), .wq_pop(wpop && wslot == INDEX),
            .read_work(working_src[INDEX]), .write_work(working_dst[INDEX]),
            .head_out(head_out[INDEX*DIM*32 +: DIM*32]), .out_valid(out_valid[INDEX]), .out_index(out_idx[INDEX]),
            .done(finished[INDEX]), .busy());
    end
endmodule
