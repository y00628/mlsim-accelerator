# Phase 2 — SystemC Systolic Array

## Goal

Model a weight-stationary systolic array in SystemC TLM-2.0 and validate its output against the Phase 1 golden GEMM.

The model is cycle-approximate: one clock cycle = one MAC per PE. It does not model pipeline latency, wire delays, or SRAM access times — those come in Phase 3 (RTL).

---

## Background — weight-stationary dataflow

In a weight-stationary systolic array:

- **Weights** are loaded once into each PE before simulation starts and stay fixed.
- **Activations** stream left-to-right across each row, one element per cycle.
- **Partial sums** accumulate top-to-bottom within each column.

For an M×K × K×N GEMM on an N×N array:
- Each PE at position (row, col) holds weight `B[row][col]`
- The activation stream for column `col` is the corresponding row of A
- After K cycles, `PE[row][col].acc` holds the partial sum contribution for `C[row][col]`

---

## Processing Element (`systemc/pe.h` + `pe.cpp`)

### Interface

```cpp
SC_MODULE(PE) {
    sc_in<bool>        clk;      // synchronous to rising edge
    sc_fifo_in<float>  act_in;   // activation from left neighbor
    sc_fifo_out<float> act_out;  // activation forwarded to right neighbor

    float weight;  // stationary — set before simulation
    float acc;     // accumulator — readable after simulation
};
```

### Behavior (`PE::run`)

Runs as `SC_CTHREAD` clocked on `clk.pos()`. Each rising edge:

1. Read one activation from `act_in` (blocks if fifo is empty)
2. `acc += weight * act`
3. Write activation to `act_out` (pass-through to right neighbor)

After K cycles, `acc` holds the completed dot product for this PE's row/column pair.

### Key design decisions

- `act_in` / `act_out` are `sc_fifo` channels — decoupled, blocking reads/writes give natural flow control without explicit handshake logic.
- `weight` and `acc` are plain `float` members, not ports — the testbench sets `weight` before `sc_start()` and reads `acc` after.
- The `for(;;)` loop with `wait()` at the top means the PE is always ready for the next activation; it never terminates on its own.

---

## Testbench (`systemc/tb_pe.cpp`)

Validates the PE in isolation before the full array is wired.

### Structure

All tests share one elaborated `TB` module (one clock, one PE, one fifo pair). SystemC does not allow re-elaboration after `sc_start`, so `run_test()` resets `dut.acc = 0.0f` and `dut.weight` directly between tests.

`run_all()` runs as an `SC_THREAD`. It calls `wait(K, SC_NS)` rather than `sc_start(K, SC_NS)` — the latter would restart the scheduler, which is illegal inside a process. `sc_stop()` is called at the end to terminate the PE's infinite loop.

### Test list

| # | Name | What it checks |
|---|------|---------------|
| 1 | Basic MAC correctness | `acc == sum(weight * act[i])` for K integer inputs |
| 2a | Zero weight | `acc == 0.0f` exactly, regardless of activations |
| 2b | Zero activations | `acc == 0.0f` exactly, regardless of weight |
| 2c | Negative values | Sign handling: negative weight and/or negative activations |
| 2d | Unit weight | `acc == sum(acts)` when `weight == 1.0f` |
| 3 | Pass-through integrity | Every `act_in` value appears on `act_out` unchanged and in order; guards against accidentally forwarding `acc` or `weight` |
| 4 | Cumulative accumulation | `acc` is not reset each cycle — after K cycles it reflects all K MACs |
| 5 | Numerical precision | K=8, mixed signs; reference computed in plain C++ in the same order; comparison uses `EPS = 1e-4f` |
| 6 | K boundary | Both fifos are empty after exactly K cycles — timing matches what the array will expect |

### Comparison policy

- Zero-case checks (2a, 2b) use `== 0.0f` — exact by IEEE 754 (`0 * x = 0`)
- All other checks use `std::fabs(got - ref) < 1e-4f`
- Pass-through uses `!=` (bit-identical, not approximate)

### Build and run

```bash
cd systemc/build
cmake ..
make tb_pe
./tb_pe
```

Expected output:
```
=== tb_pe: PE unit tests ===

[1] Basic MAC correctness
  PASS  acc == sum(weight * act[i])

[2a] Zero weight
  PASS  acc == 0 with zero weight

[2b] Zero activations
  PASS  acc == 0 with zero activations

[2c] Negative weight / activations
  PASS  acc correct with negative weight/activations

[2d] Unit weight
  PASS  acc == sum(acts) for weight=1

[3] Pass-through integrity
  PASS  act_out values match act_in values exactly
  PASS  act_out order matches act_in order

[4] Cumulative accumulation
  PASS  acc cumulative after K cycles (not reset per cycle)

[5] Numerical precision (K=8, mixed signs)
  PASS  acc within 1e-4f of C++ reference (mixed signs)

[6] K boundary — exactly K cycles
  PASS  act_in empty after K cycles
  PASS  act_out empty after draining K values

=== Results: 11 passed, 0 failed ===
```

---

## Array (`systemc/array.h`) — TODO

The `Array` module will instantiate an N×N grid of PEs and manage the inter-PE wiring and input/output scheduling. Current state: interface skeleton only.

Planned behavior:
- Accept weight matrix B and activation matrix A before simulation
- Wire activation fifos left-to-right across each row
- Wire partial-sum accumulation top-to-bottom across each column
- After K cycles, collect `C[i][j] = PE[i][j].acc` for all (i, j)
- Log total cycles, utilization %, and stall cycles
- Output C must match `gemm_naive` within 1e-4

---

## Array testbench (`systemc/tb_array.cpp`) — TODO

Planned flow:
1. Generate random A (M×K) and B (K×N)
2. Compute reference `C_ref = gemm_naive(A, B)`
3. Run systolic array simulation
4. Compare `C_sim` against `C_ref` element-wise within 1e-4
5. Print cycle count, utilization %, stall cycles
6. Hook into `analysis/cycle_stats.py` for log parsing and visualization

---

## Up next — Phase 3

- Implement the 3-stage pipelined MAC in SystemVerilog (`rtl/pe.sv`)
- Add roofline data points for `workloads/attention.cpp` and `workloads/depthwise_conv.cpp`
