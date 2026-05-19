#pragma once
// Processing Element (PE) — Phase 2 (June)
//
// Models a single MAC unit in SystemC TLM-2.0.
// Each PE:
//   - Holds a stationary weight (weight-stationary dataflow)
//   - Receives an input activation each cycle via sc_fifo
//   - Passes activation to the next PE (right neighbor)
//   - Accumulates partial sum and sends it downward
//
// Key SystemC concepts you'll use here:
//   SC_MODULE, SC_THREAD, sc_fifo<float>, wait(1, SC_NS)
//
// TODO June week 1: implement single PE
// TODO June week 2: wire N PEs into a row, then NxN array


#include <systemc.h>

SC_MODULE(PE) {
    // --- Ports ---
    sc_in<bool>      clk;          // clock — all behavior is synchronous to this
    sc_fifo_in<float>  act_in;     // activation arriving from the left neighbor
    sc_fifo_out<float> act_out;    // activation forwarded to the right neighbor

    // --- Internal state ---
    float weight;   // stationary — loaded once before simulation starts
    float acc;      // accumulator — builds up the partial sum over K steps

    // --- Process declaration ---
    void run();     // defined in pe.cpp; this is the SC_THREAD

    // --- Constructor ---
    SC_CTOR(PE) {
        SC_THREAD(run);         // register run() as a thread process
        sensitive << clk.pos(); // append to the list of events that wake up the thread
    }
};
