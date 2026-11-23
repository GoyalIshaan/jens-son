module pe #(
        parameter int DATA_W = 8, 
        parameter int ACC_W = 32
    ) (
        input logic clock,
        input logic reset,
        input logic clear_acc,
        input logic enable,

        input logic signed [DATA_W-1:0] a,
        input logic signed [DATA_W-1:0] b,

        output logic signed [DATA_W-1:0] a_out,
        output logic signed [DATA_W-1:0] b_out,
        output logic signed [ACC_W-1:0] acc_out
    );

    localparam int PRODUCT_W = 2 * DATA_W;

    logic signed [ACC_W-1:0] acc;
    logic signed [PRODUCT_W-1:0] product_raw;
    logic signed [ACC_W-1:0] product_to_add;

    initial begin
        assert (ACC_W >= PRODUCT_W)
            else $fatal("ACC_W must be >= PRODUCT_W");
    end

    assign product_raw = a*b;
    // grabs the MSB of the raw product and concatenates it to product raw ACC_W - PRODUCT_W times
    assign product_to_add = {{(ACC_W - PRODUCT_W){product_raw[PRODUCT_W-1]}}, product_raw};

    always_ff @(posedge clock) begin
        if (reset) begin
            a_out <= '0;
            b_out <= '0;
            acc <= '0;
        end else if (enable) begin
            if (clear_acc) begin
                acc <= product_to_add;
            end else begin
                acc <= acc + product_to_add;
            end
            a_out <= a;
            b_out <= b;
        end
    end

    assign acc_out = acc;
endmodule
