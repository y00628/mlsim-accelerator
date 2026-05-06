# mlsim-accelerator

A cycle-approximate ML accelerator simulator built in three layers:
- **C++ golden model** — reference GEMM + roofline analysis tooling
- **SystemC TLM model** — NxN systolic array, weight-stationary dataflow
- **SystemVerilog RTL** — pipelined MAC processing element

## Build (golden model)
```bash
cd golden
mkdir build && cd build
cmake ..
make
./test_gemm      # runs unit tests
./roofline       # prints benchmark numbers
```

## Project phases
| Phase | Month | Focus |
|-------|-------|-------|
| 1 | May  | C++ GEMM, tiling, roofline |
| 2 | June | SystemC PE + systolic array |
| 3 | July | SystemVerilog RTL PE + workloads |
| 4 | Aug  | Dataflow variants, write-up |
