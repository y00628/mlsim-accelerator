#include "pe.h"
#include <systemc.h>

// ---------------------------------------------------------------------------
// PE::run() — SC_CTHREAD, wakes on clk.pos()
//
// Each rising edge:
//   1. Read one activation from act_in  (blocks if fifo is empty)
//   2. MAC: acc += weight * activation
//   3. Forward activation to act_out    (blocks if fifo is full)
//
// After K steps the accumulator holds the completed partial sum for this PE.
// The testbench reads acc directly after simulation ends.
// ---------------------------------------------------------------------------
void PE::run() {
    // --- Reset / initialization ---
    acc = 0.0f;

    for (;;) {
        // Wait for the next rising clock edge
        wait();

        // Step 1: read incoming activation
        float act = act_in.read();

        // Step 2: multiply-accumulate
        acc += weight * act;

        // Step 3: forward activation to right neighbor
        act_out.write(act);
    }
}