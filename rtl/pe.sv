// Processing Element — RTL (Phase 3, July)
//
// Pipelined MAC: result = a * b + acc
// Valid/ready handshake so the systolic array can apply backpressure.
//
// Pipeline stages (fill in during July):
//   Stage 1: register inputs (a, b), assert valid_s1
//   Stage 2: multiply  → product
//   Stage 3: accumulate → result, assert valid_out
//
// Key things to understand before coding:
//   - Why pipeline registers go on the OUTPUT of each stage, not input
//   - How valid propagates through the pipe (shift register of valid bits)
//   - Why you need rst_n (active-low reset) on all state registers

module pe #(
    parameter DATA_W = 16  // bit width — start with 16-bit fixed point
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               valid_in,
    input  logic [DATA_W-1:0]  a,        // activation in
    input  logic [DATA_W-1:0]  b,        // weight (stationary — loaded at reset)
    output logic               valid_out,
    output logic [2*DATA_W-1:0] result   // partial sum out
);
    // TODO July: implement 3-stage pipeline
endmodule
