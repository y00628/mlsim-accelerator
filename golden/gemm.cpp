#include "gemm.hpp"

// Naive GEMM: three nested loops, no cache optimization.
// Access pattern for B is column-wise → cache-unfriendly for row-major storage.
// This is your baseline; week 2 replaces this with gemm_tiled.
void gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i*N + j] += A[i*K + k] * B[k*N + j];
}

// Tiled GEMM — implement in week 2
// Breaks A, B, C into tile_size x tile_size blocks so each block
// fits in L1/L2 cache, dramatically reducing cache misses.
void gemm_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N,
    int tile_size)
{
    // TODO week 2: three outer loops over tiles, three inner loops within tile
    (void)A; (void)B; (void)C;
    (void)M; (void)K; (void)N; (void)tile_size;
}
