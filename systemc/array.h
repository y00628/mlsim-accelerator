#pragma once
// NxN Systolic Array — Phase 2 (June)
//
// Instantiates an N x N grid of PE modules.
// Weight-stationary: weights are loaded once, activations stream left-to-right,
// partial sums accumulate top-to-bottom.
//
// After Phase 2 this model should:
//   1. Accept a matrix A (activations) and B (weights)
//   2. Simulate the cycle-by-cycle data movement
//   3. Log total cycles, utilization %, stall cycles
//   4. Output matrix C that matches golden/gemm.cpp within 1e-4

#include <systemc.h>

SC_MODULE(Array) {
    // --- Ports ---
    sc_in<bool>      clk;          // clock — all behavior is synchronous to this
    sc_fifo_in<float>  act_in;     // activation arriving from the left neighbor
    sc_fifo_out<float> C_out;    // activation forwarded to the right neighbor

    // --- Internal state ---
    float weight;   // stationary — loaded once before simulation starts
    float acc;      // accumulator — builds up the partial sum over K steps

    // --- Process declaration ---
    void run();     // defined in pe.cpp; this is the SC_THREAD

    // --- Constructor ---
    SC_CTOR(Array) {
        SC_CTHREAD(run, clk.pos()); // resumes on rising edge automatically
    }
};
