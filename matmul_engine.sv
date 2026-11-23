module matmul_engine #(
        parameter int MAX_M=16,
        parameter int MAX_N=16,
        parameter int MAX_K=16,

        parameter int ROWS = 2,
        parameter int COLS = 2,
        parameter int INNER_K = 2,
        parameter int DATA_W=8,
        parameter int ACC_W=32
    ) (
        input logic reset,
        input logic clock,
        input logic start,

        input logic [$clog2((MAX_M*MAX_K)+1)-1:0]ADDR_A,
        input logic [$clog2((MAX_K*MAX_N)+1)-1:0]ADDR_B,
        
        output logic [$clog2((MAX_M*MAX_N)+1)-1:0]ADDR_C,
        output logic done,
        output logic ready
    );

    

endmodule