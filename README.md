# mlsim-accelerator

A cycle-approximate ML accelerator simulator built in three layers:
- **C++ golden model** — reference GEMM + roofline analysis tooling
- **SystemC TLM model** — NxN systolic array, weight-stationary dataflow
- **SystemVerilog RTL** — pipelined MAC processing element

## Build (SystemC model)

### 1 — Install SystemC (one-time)
```bash
# Download (macOS uses curl, not wget)
curl -L -O https://github.com/accellera-official/systemc/archive/refs/tags/3.0.0.tar.gz
tar xf 3.0.0.tar.gz
cd systemc-3.0.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/systemc -DCMAKE_CXX_STANDARD=17
make -j$(sysctl -n hw.logicalcpu) && make install
cd ../..   # back to wherever you started
```

Then add to `~/.zshrc` (run once, then open a new terminal):
```bash
echo 'export SYSTEMC_HOME=$HOME/systemc' >> ~/.zshrc
source ~/.zshrc
```

### 2 — Build the testbench (standalone)
```bash
cd /path/to/mlsim-accelerator/systemc
mkdir -p build && cd build
cmake ..
make
./tb_array
```

### 3 — Build everything from the repo root
```bash
cd /path/to/mlsim-accelerator
mkdir -p build && cd build
cmake ..
make
./systemc/tb_array
```

## Build (golden model)
```bash
cd golden
mkdir build && cd build
cmake ..
make
./test_gemm      # runs unit tests
./roofline       # prints benchmark numbers
```

## Benchmark results (host CPU baseline)

Measured on Apple M-series, `-O3 -march=native`. Reproduce with `./build/roofline`.

| Kernel | AI (FLOP/B) | GFLOP/s (256³) | vs naive |
|--------|------------:|---------------:|---------:|
| naive | 0.125 | 2.96 | — |
| tiled-16 | 2.0 | 16.65 | 5.6× |
| tiled-32 | 4.0 | 20.06 | 6.8× |
| tiled-64 | 8.0 | 30.89 | **10.4×** |

AI uses a no-cache worst-case model (conservative lower bounds). Ridge point ≈ 3.6 FLOP/B (80 GFLOP/s ÷ 22 GB/s empirical BW). tile=64 at 8.0 FLOP/B crosses into the compute-bound region.

See [docs/phase1-roofline.md](docs/phase1-roofline.md) for the full AI derivation, memory traffic model, all matrix configs, and roofline plot.

## Project phases
| Phase | Focus | Status |
|-------|-------|--------|
| 1 | C++ GEMM, tiling, roofline | Done |
| 2 | SystemC PE + systolic array | In progress |
| 3 | SystemVerilog RTL PE + workloads | Not started |
| 4 | Dataflow variants, write-up | Not started |

## Phase 2 goals — SystemC systolic array

Model a weight-stationary systolic array in SystemC TLM-2.0 and validate it against the Phase 1 golden GEMM.

**Processing Element (`systemc/pe.h`)**
- Single MAC unit modeled as an `SC_MODULE`
- Holds a stationary weight; receives activations via `sc_fifo<float>` each cycle
- Passes activation to right neighbor, accumulates partial sum downward

**Array (`systemc/array.h`)**
- N×N grid of PEs
- Weight-stationary dataflow: weights loaded once, activations stream left-to-right, partial sums accumulate top-to-bottom
- Outputs: total cycles, utilization %, stall cycles

**Testbench (`systemc/tb_array.sc`)**
- Generate random A (M×K) and B (K×N)
- Run systolic simulation; compare output C against `gemm_naive` within 1e-4
- Print cycle count and utilization stats

**Definition of done:** simulation output matches golden within tolerance, `analysis/cycle_stats.py` parses the log and plots utilization.
