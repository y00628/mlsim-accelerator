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

## Project phases
| Phase | Focus | Status |
|-------|-------|--------|
| 1 | C++ GEMM, tiling, roofline | Done |
| 2 | SystemC PE + systolic array | In progress |
| 3 | SystemVerilog RTL PE + workloads | Not started |
| 4 | Dataflow variants, write-up | Not started |


## Phase 1 - Benchmark results (host CPU baseline)

Measured on Apple M-series, `-O3 -march=native`. Reproduce with `./build/roofline`.

| Kernel | AI (FLOP/B) | GFLOP/s (256³) | vs naive |
|--------|------------:|---------------:|---------:|
| naive | 0.125 | 2.96 | — |
| tiled-16 | 2.0 | 16.65 | 5.6× |
| tiled-32 | 4.0 | 20.06 | 6.8× |
| tiled-64 | 8.0 | 30.89 | **10.4×** |

AI uses a no-cache worst-case model (conservative lower bounds). Ridge point ≈ 3.6 FLOP/B (80 GFLOP/s ÷ 22 GB/s empirical BW). tile=64 at 8.0 FLOP/B crosses into the compute-bound region.

See [docs/phase1-roofline.md](docs/phase1-roofline.md) for the full AI derivation, memory traffic model, all matrix configs, and roofline plot.



## Phase 2 — SystemC systolic array

PE is implemented and unit-tested. Array and full testbench are TODO.

| Component | File | Status |
|-----------|------|--------|
| Processing Element | `systemc/pe.h`, `pe.cpp` | Done |
| PE unit tests | `systemc/tb_pe.cpp` | Done — 11/11 pass |
| N×N Array | `systemc/array.h` | Skeleton |
| Array testbench | `systemc/tb_array.cpp` | Skeleton |

```bash
cd systemc/build && cmake .. && make tb_pe && ./tb_pe
```

See [docs/phase2-systemc.md](docs/phase2-systemc.md) for PE design, test coverage, and array goals.
