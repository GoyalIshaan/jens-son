`timescale 1ns/1ps

module sys_array_dynamic_case #(
    parameter int ROWS       = 2,
    parameter int COLS       = 2,
    parameter int DATA_W     = 8,
    parameter int ACC_W      = 32,
    parameter int K_MAX      = 4,
    parameter int RANDOM_RUNS = 8,
    parameter int SEED       = 32'h1234
);
    logic clock;
    logic reset;
    logic clear_acc;
    logic enable;

    logic signed [DATA_W-1:0] A[ROWS-1:0];
    logic signed [DATA_W-1:0] B[COLS-1:0];
    logic signed [ACC_W-1:0] ACC[ROWS-1:0][COLS-1:0];

    logic signed [DATA_W-1:0] mat_a[ROWS-1:0][K_MAX-1:0];
    logic signed [DATA_W-1:0] mat_b[K_MAX-1:0][COLS-1:0];
    logic signed [ACC_W-1:0] expected[ROWS-1:0][COLS-1:0];
    logic signed [ACC_W-1:0] snapshot[ROWS-1:0][COLS-1:0];

    int test_errors;
    int unsigned lcg_state;

    sys_array #(
        .ROWS(ROWS),
        .COLS(COLS),
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) dut (
        .clock(clock),
        .reset(reset),
        .clear_acc(clear_acc),
        .enable(enable),
        .A(A),
        .B(B),
        .ACC(ACC)
    );

    initial begin
        clock = 1'b0;
    end

    always #5 clock = ~clock;

    task automatic drive_zero;
        int r;
        int c;
        begin
            for (r = 0; r < ROWS; r++) begin
                A[r] = '0;
            end

            for (c = 0; c < COLS; c++) begin
                B[c] = '0;
            end
        end
    endtask

    task automatic tick;
        begin
            @(posedge clock);
            #1;
        end
    endtask

    task automatic check_acc_all(input string test_name);
        int r;
        int c;
        int mismatches;
        begin
            mismatches = 0;

            for (r = 0; r < ROWS; r++) begin
                for (c = 0; c < COLS; c++) begin
                    if (ACC[r][c] !== expected[r][c]) begin
                        mismatches++;
                        test_errors++;
                        $error(
                            "%m %s ACC[%0d][%0d] mismatch: got %0d, expected %0d",
                            test_name, r, c, ACC[r][c], expected[r][c]
                        );
                    end
                end
            end

            if (mismatches == 0) begin
                $display("PASS: %m %s", test_name);
            end
        end
    endtask

    task automatic check_acc_zero(input string test_name);
        int r;
        int c;
        int mismatches;
        begin
            mismatches = 0;

            for (r = 0; r < ROWS; r++) begin
                for (c = 0; c < COLS; c++) begin
                    if (ACC[r][c] !== '0) begin
                        mismatches++;
                        test_errors++;
                        $error(
                            "%m %s ACC[%0d][%0d] should be zero, got %0d",
                            test_name, r, c, ACC[r][c]
                        );
                    end
                end
            end

            if (mismatches == 0) begin
                $display("PASS: %m %s", test_name);
            end
        end
    endtask

    task automatic take_snapshot;
        int r;
        int c;
        begin
            for (r = 0; r < ROWS; r++) begin
                for (c = 0; c < COLS; c++) begin
                    snapshot[r][c] = ACC[r][c];
                end
            end
        end
    endtask

    task automatic check_snapshot(input string test_name);
        int r;
        int c;
        int mismatches;
        begin
            mismatches = 0;

            for (r = 0; r < ROWS; r++) begin
                for (c = 0; c < COLS; c++) begin
                    if (ACC[r][c] !== snapshot[r][c]) begin
                        mismatches++;
                        test_errors++;
                        $error(
                            "%m %s ACC[%0d][%0d] changed during stall: got %0d, expected %0d",
                            test_name, r, c, ACC[r][c], snapshot[r][c]
                        );
                    end
                end
            end

            if (mismatches == 0) begin
                $display("PASS: %m %s", test_name);
            end
        end
    endtask

    task automatic reset_dut(input string test_name);
        begin
            reset     = 1'b1;
            clear_acc = 1'b0;
            enable    = 1'b0;
            drive_zero();

            tick();
            tick();

            reset = 1'b0;
            enable = 1'b1;
            drive_zero();
            tick();

            check_acc_zero(test_name);
        end
    endtask

    task automatic clear_with_zero(input string test_name);
        begin
            clear_acc = 1'b1;
            enable    = 1'b1;
            drive_zero();
            tick();
            clear_acc = 1'b0;

            check_acc_zero(test_name);
        end
    endtask

    task automatic compute_expected(input int k_len);
        int r;
        int c;
        int k;
        int signed product;
        begin
            for (r = 0; r < ROWS; r++) begin
                for (c = 0; c < COLS; c++) begin
                    expected[r][c] = '0;
                    for (k = 0; k < k_len; k++) begin
                        product = mat_a[r][k] * mat_b[k][c];
                        expected[r][c] = expected[r][c] + product;
                    end
                end
            end
        end
    endtask

    task automatic drive_stream_cycle(input int cycle, input int k_len);
        int r;
        int c;
        int k_idx;
        begin
            for (r = 0; r < ROWS; r++) begin
                k_idx = cycle - r;
                if ((k_idx >= 0) && (k_idx < k_len)) begin
                    A[r] = mat_a[r][k_idx];
                end else begin
                    A[r] = '0;
                end
            end

            for (c = 0; c < COLS; c++) begin
                k_idx = cycle - c;
                if ((k_idx >= 0) && (k_idx < k_len)) begin
                    B[c] = mat_b[k_idx][c];
                end else begin
                    B[c] = '0;
                end
            end
        end
    endtask

    task automatic run_stream_matmul(
        input string test_name,
        input int k_len,
        input bit lazy_clear_first_cycle
    );
        int cycle;
        int cycles_to_run;
        begin
            $display("\nTEST: %m %s", test_name);
            compute_expected(k_len);

            if (!lazy_clear_first_cycle) begin
                clear_with_zero({test_name, " clear"});
            end

            cycles_to_run = k_len + ROWS + COLS;
            enable = 1'b1;

            for (cycle = 0; cycle < cycles_to_run; cycle++) begin
                clear_acc = lazy_clear_first_cycle && (cycle == 0);
                drive_stream_cycle(cycle, k_len);
                tick();
            end

            clear_acc = 1'b0;
            drive_zero();
            check_acc_all(test_name);
        end
    endtask

    task automatic run_stream_matmul_with_stall(
        input string test_name,
        input int k_len,
        input int stall_cycle
    );
        int cycle;
        int cycles_to_run;
        begin
            $display("\nTEST: %m %s", test_name);
            compute_expected(k_len);
            clear_with_zero({test_name, " clear"});

            cycles_to_run = k_len + ROWS + COLS;
            enable = 1'b1;

            for (cycle = 0; cycle < cycles_to_run; cycle++) begin
                clear_acc = 1'b0;
                drive_stream_cycle(cycle, k_len);

                if (cycle == stall_cycle) begin
                    take_snapshot();
                    enable = 1'b0;
                    tick();
                    check_snapshot({test_name, " holds while enable is low"});
                    enable = 1'b1;
                    drive_stream_cycle(cycle, k_len);
                end

                tick();
            end

            drive_zero();
            check_acc_all(test_name);
        end
    endtask

    task automatic run_reset_dominates_test;
        begin
            $display("\nTEST: %m reset dominates clear and enable");
            clear_with_zero("reset-dominates setup clear");

            A[0] = 8'sd7;
            B[0] = -8'sd5;
            tick();

            reset = 1'b1;
            clear_acc = 1'b1;
            enable = 1'b1;
            drive_bad_inputs();
            tick();
            check_acc_zero("reset dominates active clear/enable");

            reset = 1'b0;
            clear_acc = 1'b0;
            enable = 1'b1;
            drive_zero();
            tick();
            check_acc_zero("reset release leaves array idle");
        end
    endtask

    task automatic drive_bad_inputs;
        int r;
        int c;
        begin
            for (r = 0; r < ROWS; r++) begin
                A[r] = 8'sd17 + r;
            end

            for (c = 0; c < COLS; c++) begin
                B[c] = -8'sd11 - c;
            end
        end
    endtask

    task automatic clear_matrices;
        int r;
        int c;
        int k;
        begin
            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < K_MAX; k++) begin
                    mat_a[r][k] = '0;
                end
            end

            for (k = 0; k < K_MAX; k++) begin
                for (c = 0; c < COLS; c++) begin
                    mat_b[k][c] = '0;
                end
            end
        end
    endtask

    task automatic fill_positive_pattern(input int k_len);
        int r;
        int c;
        int k;
        begin
            clear_matrices();

            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < k_len; k++) begin
                    mat_a[r][k] = 1 + r + k;
                end
            end

            for (k = 0; k < k_len; k++) begin
                for (c = 0; c < COLS; c++) begin
                    mat_b[k][c] = 2 + (2 * k) + c;
                end
            end
        end
    endtask

    task automatic fill_signed_pattern(input int k_len);
        int r;
        int c;
        int k;
        int signed value;
        begin
            clear_matrices();

            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < k_len; k++) begin
                    value = 1 + r + k;
                    if (((r + k) % 2) == 1) begin
                        value = -value;
                    end
                    mat_a[r][k] = value;
                end
            end

            for (k = 0; k < k_len; k++) begin
                for (c = 0; c < COLS; c++) begin
                    value = 2 + k + c;
                    if (((k + c) % 2) == 0) begin
                        value = -value;
                    end
                    mat_b[k][c] = value;
                end
            end
        end
    endtask

    task automatic fill_edge_pattern(input int k_len);
        int r;
        int c;
        int k;
        int selector;
        begin
            clear_matrices();

            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < k_len; k++) begin
                    selector = (r + k) % 4;
                    case (selector)
                        0: mat_a[r][k] = 8'sd127;
                        1: mat_a[r][k] = -8'sd128;
                        2: mat_a[r][k] = 8'sd64;
                        default: mat_a[r][k] = -8'sd32;
                    endcase
                end
            end

            for (k = 0; k < k_len; k++) begin
                for (c = 0; c < COLS; c++) begin
                    selector = (k + c) % 4;
                    case (selector)
                        0: mat_b[k][c] = -8'sd3;
                        1: mat_b[k][c] = 8'sd5;
                        2: mat_b[k][c] = -8'sd7;
                        default: mat_b[k][c] = 8'sd9;
                    endcase
                end
            end
        end
    endtask

    task automatic fill_lcg_pattern(input int k_len, input int unsigned seed);
        int r;
        int c;
        int k;
        int signed value;
        begin
            clear_matrices();
            lcg_state = seed;

            for (r = 0; r < ROWS; r++) begin
                for (k = 0; k < k_len; k++) begin
                    lcg_state = (lcg_state * 32'd1664525) + 32'd1013904223;
                    value = (lcg_state % 21) - 10;
                    mat_a[r][k] = value;
                end
            end

            for (k = 0; k < k_len; k++) begin
                for (c = 0; c < COLS; c++) begin
                    lcg_state = (lcg_state * 32'd1664525) + 32'd1013904223;
                    value = (lcg_state % 21) - 10;
                    mat_b[k][c] = value;
                end
            end
        end
    endtask

    task automatic run_random_sweep;
        int run;
        int k_len;
        int unsigned run_seed;
        string test_name;
        begin
            for (run = 0; run < RANDOM_RUNS; run++) begin
                k_len = 1 + (run % K_MAX);
                run_seed = SEED + (run * 32'd97) + (ROWS * 32'd13) + (COLS * 32'd29);
                fill_lcg_pattern(k_len, run_seed);
                test_name = $sformatf("random sweep run=%0d k=%0d", run, k_len);
                run_stream_matmul(test_name, k_len, run[0]);
            end
        end
    endtask

    task automatic run_all(output int errors_out);
        begin
            test_errors = 0;

            $display("\n==============================");
            $display("CASE: %m ROWS=%0d COLS=%0d K_MAX=%0d", ROWS, COLS, K_MAX);
            $display("==============================");

            reset_dut("initial reset clears accumulators");

            fill_positive_pattern(K_MAX);
            run_stream_matmul("positive dynamic matmul", K_MAX, 1'b0);

            fill_signed_pattern(K_MAX);
            run_stream_matmul("signed dynamic matmul", K_MAX, 1'b0);

            fill_edge_pattern(K_MAX);
            run_stream_matmul("edge-value dynamic matmul", K_MAX, 1'b1);

            fill_signed_pattern(K_MAX);
            run_stream_matmul_with_stall("enable stall dynamic matmul", K_MAX, (K_MAX + ROWS + COLS) / 2);

            run_reset_dominates_test();
            run_random_sweep();

            errors_out = test_errors;
        end
    endtask
endmodule

module tb_sys_array;
    int total_errors;
    int case_errors;

    sys_array_dynamic_case #(
        .ROWS(2),
        .COLS(2),
        .K_MAX(4),
        .RANDOM_RUNS(8),
        .SEED(32'h1020)
    ) case_square_2x2();

    sys_array_dynamic_case #(
        .ROWS(3),
        .COLS(2),
        .K_MAX(5),
        .RANDOM_RUNS(8),
        .SEED(32'h3040)
    ) case_tall_3x2();

    sys_array_dynamic_case #(
        .ROWS(2),
        .COLS(3),
        .K_MAX(5),
        .RANDOM_RUNS(8),
        .SEED(32'h5060)
    ) case_wide_2x3();

    initial begin
        $dumpfile("sys_array.vcd");
        $dumpvars(0, tb_sys_array);
    end

    initial begin
        total_errors = 0;

        case_square_2x2.run_all(case_errors);
        total_errors += case_errors;

        case_tall_3x2.run_all(case_errors);
        total_errors += case_errors;

        case_wide_2x3.run_all(case_errors);
        total_errors += case_errors;

        if (total_errors == 0) begin
            $display("\nALL SYS ARRAY TESTS PASSED");
        end else begin
            $fatal(1, "\nSYS ARRAY TESTS FAILED: %0d mismatches", total_errors);
        end

        $finish;
    end
endmodule
