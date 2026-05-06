#pragma once
#include <vector>

// C = A * B
// A: M x K, B: K x N, C: M x N (row-major, zero-initialized before call)
void gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N
);

void gemm_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N,
    int tile_size
);
