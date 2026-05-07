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

| M | K | N | AI (FLOP/B) | Naive (GFLOP/s) | Tiled-64 (GFLOP/s) | Speedup |
|--:|--:|--:|------------:|----------------:|-------------------:|--------:|
| 64 | 64 | 64 | 10.7 | 2.09 | 28.75 | **13.8×** |
| 128 | 128 | 128 | 21.3 | 3.32 | 19.78 | **6.0×** |
| 256 | 256 | 256 | 42.7 | 2.83 | 29.54 | **10.4×** |
| 512 | 512 | 512 | 85.3 | 2.79 | 22.94 | **8.2×** |
| 64 | 64 | 512 | 15.1 | 4.84 | 25.81 | **5.3×** |
| 512 | 128 | 512 | 42.7 | 3.51 | 26.81 | **7.6×** |

**Takeaways**
- **AI is problem-defined, not implementation-defined** — naive and tiled have the same arithmetic intensity because they touch the same data for the same FLOPs.
- **Naive bandwidth collapses with size** — 0.20 GB/s at 64³ → 0.03 GB/s at 512³; data spills past L2 and the CPU stalls waiting on DRAM.
- **Tiling recovers cache reuse** — tile=64 achieves 8–14× speedup by keeping each tile in L1/L2 for the full inner loop.
- **All configs sit above the ridge point** (~10–15 FLOP/B on M-series) — tiled GEMM is compute-bound in principle; remaining headroom requires AVX/AMX or multi-threading (future phases).

<details>
<summary>Full tile-size sweep</summary>

```
=== M=64  K=64  N=64 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
   64   64   64  naive     0.25       2.09       0.20     10.667
   64   64   64     16     0.05       9.91       0.93     10.667
   64   64   64     32     0.03      18.55       1.74     10.667
   64   64   64     64     0.02      28.75       2.70     10.667

=== M=128  K=128  N=128 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  128  128  128  naive     1.26       3.32       0.16     21.333
  128  128  128     16     0.41      10.31       0.48     21.333
  128  128  128     32     0.34      12.24       0.57     21.333
  128  128  128     64     0.21      19.78       0.93     21.333

=== M=256  K=256  N=256 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  256  256  256  naive    11.85       2.83       0.07     42.667
  256  256  256     16     2.07      16.18       0.38     42.667
  256  256  256     32     1.67      20.07       0.47     42.667
  256  256  256     64     1.14      29.54       0.69     42.667

=== M=512  K=512  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  512  512  512  naive    96.26       2.79       0.03     85.333
  512  512  512     16    17.22      15.59       0.18     85.333
  512  512  512     32    14.13      19.00       0.22     85.333
  512  512  512     64    11.70      22.94       0.27     85.333

=== M=64  K=64  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
   64   64  512  naive     0.87       4.84       0.32     15.059
   64   64  512     16     0.26      16.23       1.08     15.059
   64   64  512     32     0.23      18.03       1.20     15.059
   64   64  512     64     0.16      25.81       1.71     15.059

=== M=512  K=128  N=512 ===
    M    K    N   tile       ms    GFLOP/s       GB/s  AI(FLOP/B)
-----------------------------------------------------------------
  512  128  512  naive    19.13       3.51       0.08     42.667
  512  128  512     16     4.09      16.41       0.38     42.667
  512  128  512     32     3.38      19.87       0.47     42.667
  512  128  512     64     2.50      26.81       0.63     42.667
```
</details>

## Project phases
| Phase | Month | Focus |
|-------|-------|-------|
| 1 | May  | C++ GEMM, tiling, roofline |
| 2 | June | SystemC PE + systolic array |
| 3 | July | SystemVerilog RTL PE + workloads |
| 4 | Aug  | Dataflow variants, write-up |
