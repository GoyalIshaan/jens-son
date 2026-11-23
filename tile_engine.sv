typedef enum logic [1:0] {
    IDLE,
    RUN,
    DONE
} fsm_state;

module tile_engine #(
    parameter int INNER_K = 2,
    parameter int ROWS = 2,
    parameter int COLS = 2,
    parameter int DATA_W=8,
    parameter int ACC_W=32
    )(
        input logic reset,
        input logic clock,
        input logic start,
        input logic init_acc,

        input logic [$clog2(ROWS+1)-1:0]    valid_rows,
        input logic [$clog2(COLS+1)-1:0]    valid_cols,
        input logic [$clog2(INNER_K+1)-1:0] valid_k,

        input  logic signed [DATA_W-1:0] A [ROWS-1:0][INNER_K-1:0],
        input  logic signed [DATA_W-1:0] B [INNER_K-1:0][COLS-1:0],

        output logic signed [ACC_W-1:0] ACC [ROWS-1:0][COLS-1:0],
        output logic done,
        output logic ready
    );

    fsm_state state, next_state;

    localparam int TOTAL_CYCLES= ROWS + COLS + INNER_K - 2; // K + ROWS + COLS - 2
    localparam int COUNT_W = $clog2(TOTAL_CYCLES + 1); // clog2(N) gives the number of bits required to represent N bytes
    logic [COUNT_W-1:0] clock_count;

    // latches for the tile state
    logic signed [DATA_W-1:0]A_reg[ROWS-1:0][INNER_K-1:0];
    logic signed [DATA_W-1:0]B_reg[INNER_K-1:0][COLS-1:0];
    logic [$clog2(ROWS+1)-1:0]    valid_rows_reg;
    logic [$clog2(COLS+1)-1:0]    valid_cols_reg;
    logic [$clog2(INNER_K+1)-1:0] valid_k_reg;
    logic [COUNT_W-1:0] actual_max_clock_count;
    logic init_acc_reg;

    logic signed [DATA_W-1:0]a_broadcast[ROWS-1:0];
    logic signed [DATA_W-1:0]b_broadcast[COLS-1:0];

    logic clear_acc;
    logic enable;

    assign enable = (state == RUN);
    assign clear_acc = init_acc_reg && (state == RUN && clock_count == 0);
    assign ready = (state == IDLE);
    assign done = (state == DONE);

    always_comb begin
        next_state = state;
        case(state)
            IDLE: begin
                if (start) begin
                    next_state=RUN;
                end
            end
            RUN: begin
                if (clock_count == actual_max_clock_count - 1) begin
                    next_state=DONE;
                end 
            end
            DONE: begin
                next_state=IDLE;
            end
            default: begin
                next_state=IDLE;
            end
        endcase
    end

    // handshake
    always_ff @(posedge clock) begin
        if (ready && start) begin
            A_reg <= A;
            B_reg <= B;
            valid_rows_reg <= valid_rows;
            valid_cols_reg <= valid_cols;
            valid_k_reg <= valid_k;
            init_acc_reg <= init_acc;
            actual_max_clock_count <= valid_rows + valid_cols + valid_k - 2;
        end
    end

    // there are 3 states
    always_ff @(posedge clock) begin
        if (reset) begin
            state <= IDLE;
            clock_count <= 0;
        end else begin
            state <= next_state;
            case(state)
                IDLE: begin
                    clock_count <= 0;
                end
                RUN: begin                
                    clock_count <= clock_count + 1;
                end
                DONE: begin
                    clock_count <= 0;
                end
                default: begin
                    clock_count <= 0;
                end
            endcase
        end
    end

    // skew in data
    always_comb begin
        for (int row = 0; row < ROWS; row++) begin
            a_broadcast[row] = '0;
        end

        for (int col = 0; col < COLS; col++) begin
            b_broadcast[col] = '0;
        end
        if (state==RUN) begin
            for (int row=0; row < ROWS; row++) begin
                int k_idx = 0;
                k_idx = int'(clock_count) - row;
                if (
                    row < int'(valid_rows_reg) && 
                    k_idx >= 0 && k_idx < INNER_K && 
                    k_idx < int'(valid_k_reg)
                ) begin
                    a_broadcast[row] = A_reg[row][k_idx];
                end
            end
            for (int col=0; col < COLS; col++) begin
                int k_idx=0;
                k_idx = int'(clock_count) - col;
                if (
                    col < int'(valid_cols_reg) &&
                    k_idx >= 0 &&
                    k_idx < INNER_K &&
                    k_idx < int'(valid_k_reg)
                ) begin
                    b_broadcast[col] = B_reg[k_idx][col];
                end
            end
        end
    end
    
    sys_array #(
        .ROWS(ROWS),
        .COLS(COLS),
        .DATA_W(DATA_W), 
        .ACC_W(ACC_W)
    ) tile_matmul_fabric(
        .clock(clock),
        .reset(reset),
        .clear_acc(clear_acc),
        .enable(enable),
        .A(a_broadcast),
        .B(b_broadcast),
        .ACC(ACC)
    );

endmodule;
