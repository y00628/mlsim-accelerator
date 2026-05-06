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
