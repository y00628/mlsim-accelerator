// PE Testbench (Phase 3, July)
//
// Tests to write:
//   1. Single multiply-accumulate: a=3, b=4, expect result=12
//   2. Pipeline latency check: count cycles from valid_in to valid_out
//   3. Back-to-back transactions: no bubble cycles between inputs
//   4. Reset behavior: result clears to 0 after rst_n deasserted

`timescale 1ns/1ps

module pe_tb;
    localparam DATA_W = 16;

    logic clk = 0;
    logic rst_n;
    logic valid_in;
    logic [DATA_W-1:0] a, b;
    logic valid_out;
    logic [2*DATA_W-1:0] result;

    always #5 clk = ~clk;  // 100 MHz

    pe #(.DATA_W(DATA_W)) dut (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in), .a(a), .b(b),
        .valid_out(valid_out), .result(result)
    );

    initial begin
        $dumpfile("pe_tb.vcd");
        $dumpvars(0, pe_tb);

        rst_n = 0; valid_in = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        rst_n = 1;

        // TODO: drive stimulus, check outputs

        #200 $finish;
    end
endmodule
