// The special-function unit against golden/vectors/sfu_*.hex (inputs -> golden chip_sfu outputs, bit for bit).
`timescale 1ns/1ps
module tb_sfu;
    reg  [31:0] v [0:2*4200-1];
    reg  [31:0] x;
    wire [31:0] ye, yr, yq, ys, yp;
    sfu_exp      d0 (.x(x), .y(ye));
    sfu_recip    d1 (.x(x), .y(yr));
    sfu_rsqrt    d2 (.x(x), .y(yq));
    sfu_sigmoid  d3 (.x(x), .y(ys));
    sfu_softplus d4 (.x(x), .y(yp));
    integer f, i, n, bad, skip, total;
    reg [31:0] got;
    function isnan; input [31:0] t; isnan = (t[30:23] == 8'hff) && (t[22:0] != 0); endfunction
    task run(input integer which, input integer count);
        begin
            bad = 0; skip = 0;
            for (i = 0; i < count; i = i + 1) begin
                x = v[2*i]; #1;
                got = which == 0 ? ye : which == 1 ? yr : which == 2 ? yq : which == 3 ? ys : yp;
                if (which == 2 && (x[31] || x[30:23] == 0)) skip = skip + 1;      // rsqrt: positive inputs only
                else if (!(got === v[2*i+1] || (isnan(got) && isnan(v[2*i+1])))) begin
                    bad = bad + 1;
                    if (bad <= 6) $display("  f%0d MISMATCH x=%h got %h want %h", which, x, got, v[2*i+1]);
                end
            end
            $display("%s: %0d inputs (%0d skipped), %0d mismatches -> %s",
                     which == 0 ? "exp     " : which == 1 ? "recip   " : which == 2 ? "rsqrt   " : which == 3 ? "sigmoid " : "softplus",
                     count, skip, bad, bad == 0 ? "PASS" : "FAIL");
        end
    endtask
    initial begin
        $readmemh("vec/sfu_exp.hex", v);      run(0, 4108);
        $readmemh("vec/sfu_recip.hex", v);    run(1, 4106);
        $readmemh("vec/sfu_rsqrt.hex", v);    run(2, 4099);
        $readmemh("vec/sfu_sigmoid.hex", v);  run(3, 4108);
        $readmemh("vec/sfu_softplus.hex", v); run(4, 4108);
        $finish;
    end
endmodule
