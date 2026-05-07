#include <chrono>
#include <iostream>
#include <iomanip>
#include <functional>
#include "gemm.hpp"

// For an M x K x N GEMM:
//   FLOPs  = 2 * M * K * N   (one multiply + one add per inner iteration)
//   Bytes  = sizeof(float) * (M*K + K*N + M*N)  (read A, B; write C)
//   Arith intensity (AI) = FLOPs / Bytes
//   Bandwidth (GB/s)     = Bytes / seconds / 1e9
//
// When AI is high  → compute-bound  (limited by FLOP/s)
// When AI is low   → memory-bound   (limited by bandwidth)
// The crossover point on the roofline is: peak_FLOPS / peak_BW

struct BenchResult {
    int M, K, N;
    double ms;
    double gflops;
    double bandwidth_gbs;   // GB/s
    double arith_intensity; // FLOP / byte
};

BenchResult benchmark(int M, int K, int N,
                      std::function<void()> run_fn,
                      int reps = 5)
{
    run_fn(); // warm up

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++)
        run_fn();
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs  = std::chrono::duration<double>(t1 - t0).count() / reps;
    double flops = 2.0 * M * K * N;
    double bytes = sizeof(float) * double(M*K + K*N + M*N);

    return { M, K, N,
             secs * 1e3,
             flops / secs / 1e9,
             bytes / secs / 1e9,
             flops / bytes };
}

static void print_header() {
    std::cout << std::setw(5)  << "M"
              << std::setw(5)  << "K"
              << std::setw(5)  << "N"
              << std::setw(7)  << "tile"
              << std::setw(9)  << "ms"
              << std::setw(11) << "GFLOP/s"
              << std::setw(11) << "GB/s"
              << std::setw(12) << "AI(FLOP/B)"
              << "\n" << std::string(65, '-') << "\n";
}

static void print_row(const BenchResult& r, int tile) {
    std::cout << std::setw(5)  << r.M
              << std::setw(5)  << r.K
              << std::setw(5)  << r.N
              << std::setw(7)  << (tile == 0 ? std::string("naive") : std::to_string(tile))
              << std::setw(9)  << std::fixed << std::setprecision(2) << r.ms
              << std::setw(11) << std::setprecision(2) << r.gflops
              << std::setw(11) << std::setprecision(2) << r.bandwidth_gbs
              << std::setw(12) << std::setprecision(3) << r.arith_intensity
              << "\n";
}

int main() {
    // Configs: {M, K, N}
    const std::initializer_list<std::tuple<int,int,int>> configs = {
        { 64,  64,  64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        { 64,  64, 512},   // attention-like: small M, large N
        {512, 128, 512},   // rectangular
    };

    // Tile sizes to sweep for gemm_tiled
    const std::initializer_list<int> tile_sizes = {16, 32, 64};

    for (auto [M, K, N] : configs) {
        std::cout << "\n=== M=" << M << "  K=" << K << "  N=" << N << " ===\n";
        print_header();

        // --- naive ---
        {
            std::vector<float> A(M*K, 1.0f), B(K*N, 1.0f), C(M*N, 0.0f);
            auto r = benchmark(M, K, N, [&]{
                std::fill(C.begin(), C.end(), 0.0f);
                gemm_naive(A, B, C, M, K, N);
            });
            print_row(r, 0);
        }

        // --- tiled, various tile sizes ---
        for (int ts : tile_sizes) {
            if (ts > std::min({M, K, N})) continue; // skip nonsensical tile sizes
            std::vector<float> A(M*K, 1.0f), B(K*N, 1.0f), C(M*N, 0.0f);
            auto r = benchmark(M, K, N, [&]{
                std::fill(C.begin(), C.end(), 0.0f);
                gemm_tiled(A, B, C, M, K, N, ts);
            });
            print_row(r, ts);
        }
    }

    std::cout << "\nPaste these numbers into analysis/roofline_plot.py\n";
    return 0;
}
