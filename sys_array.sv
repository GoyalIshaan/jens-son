module sys_array #(
        parameter int ROWS = 2,
        parameter int COLS = 2,
        parameter int DATA_W = 8,
        parameter int ACC_W = 32
    ) (
        input logic clock,
        input logic reset,
        input logic clear_acc,
        input logic enable,

        input logic signed [DATA_W - 1:0]A[ROWS-1:0],
        input logic signed [DATA_W - 1:0]B[COLS-1:0],

        output logic signed [ACC_W-1:0]ACC[ROWS-1:0][COLS-1:0]
    );

    localparam int PRODUCT_W = 2 * DATA_W;

    initial begin
        assert (ACC_W >= PRODUCT_W)
            else $fatal("ACC_W must be >= PRODUCT_W");
    end
    
    logic signed [DATA_W - 1:0]link_a[ROWS-1:0][COLS:0]; // so that the last col can write 1 col ahead of it
    logic signed [DATA_W - 1:0]link_b[ROWS:0][COLS-1:0]; // so that the last row can write 1 row ahead of it

    genvar row, col;
    generate
        for(row = 0; row < ROWS; row++) begin: LINK_ROWS
            assign link_a[row][0] = A[row];
        end

        for(col = 0; col < COLS; col++) begin: LINK_COLS
            assign link_b[0][col] = B[col];
        end
    endgenerate

    generate
        for(row = 0; row < ROWS; row++) begin: GENERATE_ROWS
            for(col = 0; col < COLS; col++) begin: GENERATE_COLS
                pe #(
                    .DATA_W(DATA_W),
                    .ACC_W(ACC_W)
                ) pe_cell (
                    .clock(clock),
                    .reset(reset),
                    .clear_acc(clear_acc),
                    .enable(enable),

                    .a(link_a[row][col]),
                    .b(link_b[row][col]),

                    .a_out(link_a[row][col + 1]),
                    .b_out(link_b[row + 1][col]),
                    .acc_out(ACC[row][col])
                );
            end
        end
    endgenerate

endmodule
