#include <chrono>
#include <iostream>
#include <iomanip>
#include "gemm.hpp"

// For an M x K x N GEMM:
//   FLOPs  = 2 * M * K * N   (one multiply + one add per inner iteration)
//   Bytes  = sizeof(float) * (M*K + K*N + M*N)  (read A, B; write C)
//   Arith intensity = FLOPs / Bytes
//
// When intensity is high  → compute-bound  (limited by FLOP/s)
// When intensity is low   → memory-bound   (limited by bandwidth)
// The crossover point on the roofline is: peak_FLOPS / peak_BW

struct BenchResult {
    int M, K, N;
    double ms;
    double gflops;
    double arith_intensity; // FLOP / byte
};

BenchResult benchmark(int M, int K, int N, int reps = 5) {
    std::vector<float> A(M*K, 1.0f), B(K*N, 1.0f), C(M*N, 0.0f);

    // Warm up
    gemm_naive(A, B, C, M, K, N);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) {
        std::fill(C.begin(), C.end(), 0.0f);
        gemm_naive(A, B, C, M, K, N);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count() / reps;
    double flops = 2.0 * M * K * N;
    double bytes = sizeof(float) * double(M*K + K*N + M*N);

    return { M, K, N,
             secs * 1e3,
             flops / secs / 1e9,
             flops / bytes };
}

int main() {
    std::cout << std::setw(6)  << "M"
              << std::setw(6)  << "K"
              << std::setw(6)  << "N"
              << std::setw(10) << "ms"
              << std::setw(12) << "GFLOP/s"
              << std::setw(16) << "AI (FLOP/B)"
              << "\n" << std::string(56, '-') << "\n";

    for (auto [M, K, N] : std::initializer_list<std::tuple<int,int,int>>{
            {64,  64,  64},
            {128, 128, 128},
            {256, 256, 256},
            {512, 512, 512},
            {64,  64,  512},   // attention-like: small M, large N
        })
    {
        auto r = benchmark(M, K, N);
        std::cout << std::setw(6)  << r.M
                  << std::setw(6)  << r.K
                  << std::setw(6)  << r.N
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.ms
                  << std::setw(12) << std::setprecision(2) << r.gflops
                  << std::setw(16) << std::setprecision(2) << r.arith_intensity
                  << "\n";
    }

    std::cout << "\nPaste these numbers into analysis/roofline_plot.py\n";
    return 0;
}
