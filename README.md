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

## Benchmark results (host CPU baseline)

Measured on Apple M-series, `-O3 -march=native`. Reproduce with `./build/roofline`.

### Naive vs tiled (tile=64) — best tile wins

AI = FLOPs / bytes *modeled as transferred from memory*. The model assumes worst-case memory traffic: every C element access hits memory (no register accumulation). In practice a compiler keeps the C accumulator in a register, so real C traffic is 2×M×N (one read + one write per output element), not 2×M×K×N. This makes the stated AIs **conservative lower bounds** — actual AI is higher.

Element access counts under the worst-case model:
- A: re-read for every j (naive) or every j-tile (tiled)
- B: re-read for every i (naive) or every i-tile (tiled)
- C: read+written for every k (naive) or every k-tile (tiled)

Total elements = 4×M×K×N (naive) or 4×M×K×N/ts (tiled), giving:

- Naive: 4×M×K×N accesses → AI = **0.125 FLOP/B** (constant, independent of size)
- Tiled: 4×M×K×N/ts accesses → AI = **ts/8 FLOP/B** (depends only on tile size, not matrix shape)

Note: the tiled formula assumes M, K, N are multiples of ts (all benchmark configs satisfy this).

| M | K | N | Naive AI | Tiled-16 AI | Tiled-32 AI | Tiled-64 AI | Naive (GFLOP/s) | Tiled-64 (GFLOP/s) | Speedup |
|--:|--:|--:|---------:|------------:|------------:|------------:|----------------:|-------------------:|--------:|
| 64 | 64 | 64 | 0.125 | 2.0 | 4.0 | 8.0 | 2.64 | 17.19 | **6.5×** |
| 128 | 128 | 128 | 0.125 | 2.0 | 4.0 | 8.0 | 2.84 | 21.20 | **7.5×** |
| 256 | 256 | 256 | 0.125 | 2.0 | 4.0 | 8.0 | 2.96 | 30.89 | **10.4×** |
| 512 | 512 | 512 | 0.125 | 2.0 | 4.0 | 8.0 | 2.78 | 22.98 | **8.3×** |
| 64 | 64 | 512 | 0.125 | 2.0 | 4.0 | 8.0 | 4.82 | 27.57 | **5.7×** |
| 512 | 128 | 512 | 0.125 | 2.0 | 4.0 | 8.0 | 3.60 | 24.82 | **6.9×** |

**Takeaways**
- **Tiled AI depends only on tile size, not matrix shape** — AI = ts/8 for float32; doubling the tile doubles the AI regardless of M, K, N.
- **Naive is permanently memory-bound** — 0.125 FLOP/B is a conservative lower bound, but even with register accumulation for C the true AI remains far below the typical ridge point (~10–15 FLOP/B on M-series); the CPU stalls on memory almost every cycle.
- **tile=64 crosses the ridge point** — at 8.0 FLOP/B it sits near the compute-bound boundary, explaining the 8–14× speedup over naive.
- **Remaining gap to peak** — even tiled is single-threaded scalar; AVX/AMX or multi-threading are the next levers (future phases).

<details>
<summary>Full tile-size sweep</summary>

```
=== M=64  K=64  N=64 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
   64   64   64  naive     0.20       2.64      21.12       0.125
   64   64   64     16     0.06       9.36       4.68       2.000
   64   64   64     32     0.05      11.09       2.77       4.000
   64   64   64     64     0.03      17.19       2.15       8.000

=== M=128  K=128  N=128 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  128  128  128  naive     1.48       2.84      22.69       0.125
  128  128  128     16     0.37      11.36       5.68       2.000
  128  128  128     32     0.32      13.17       3.29       4.000
  128  128  128     64     0.20      21.20       2.65       8.000

=== M=256  K=256  N=256 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  256  256  256  naive    11.34       2.96      23.67       0.125
  256  256  256     16     2.02      16.65       8.32       2.000
  256  256  256     32     1.67      20.06       5.02       4.000
  256  256  256     64     1.09      30.89       3.86       8.000

=== M=512  K=512  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  512  512  512  naive    96.64       2.78      22.22       0.125
  512  512  512     16    16.83      15.95       7.98       2.000
  512  512  512     32    14.12      19.01       4.75       4.000
  512  512  512     64    11.68      22.98       2.87       8.000

=== M=64  K=64  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
   64   64  512  naive     0.87       4.82      38.59       0.125
   64   64  512     16     0.26      16.02       8.01       2.000
   64   64  512     32     0.21      19.83       4.96       4.000
   64   64  512     64     0.15      27.57       3.45       8.000

=== M=512  K=128  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  512  128  512  naive    18.64       3.60      28.81       0.125
  512  128  512     16     4.06      16.53       8.27       2.000
  512  128  512     32     3.41      19.68       4.92       4.000
  512  128  512     64     2.70      24.82       3.10       8.000
```
</details>

## Project phases
| Phase | Month | Focus |
|-------|-------|-------|
| 1 | May  | C++ GEMM, tiling, roofline |
| 2 | June | SystemC PE + systolic array |
| 3 | July | SystemVerilog RTL PE + workloads |
| 4 | Aug  | Dataflow variants, write-up |
