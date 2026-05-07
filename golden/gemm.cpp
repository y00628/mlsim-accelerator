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


// Tiled GEMM — processes tile_size×tile_size blocks so data stays in L1/L2.
// Breaks A, B, C into tile_size x tile_size blocks so each block
// fits in L1/L2 cache, dramatically reducing cache misses.
//
// Outer loops (i, j, k): step through matrix in tile_size chunks, picking tile origin.
// Inner loops (ii, kk, jj): work within one tile.
//
// Inner loop order is ii → kk → jj (not ii → jj → kk). Why jj is innermost:
//   B[kk*N + jj] increments jj by 1 → stride-1, sequential memory → prefetcher happy.
//   If jj and kk were swapped, kk increments → stride N → cache miss every step.
//
// Why kk is middle (not outermost inner):
//   For fixed (ii, kk), jj sweeps all columns: A[ii*K+kk] is the same value every step.
//   It stays in a register across the whole jj loop — free reuse with no memory traffic.
//   Example: tile_size=2, K=N=4, outer tile i=j=k=0, ii=0, kk=0:
//     A[ii*K+kk] = A[0] is loaded once into a register, then the jj loop fires:
//       jj=0: C[0] += A[0] * B[0]   (B col 0)
//       jj=1: C[1] += A[0] * B[1]   (B col 1)
//     A[0] never re-fetched; B strides by 1 (sequential) → both reads are free.
//
//   What-if: same tile but inner order ii → jj → kk (kk innermost):
//     ii=0, jj=0, kk=0: C[0] += A[0]   * B[0]
//     ii=0, jj=0, kk=1: C[0] += A[1]   * B[4]   ← B jumps by stride N=4 (cache miss)
//     ii=0, jj=1, kk=0: C[1] += A[0]   * B[1]   ← A[0] reloaded (evicted between jj iters)
//     ii=0, jj=1, kk=1: C[1] += A[1]   * B[5]   ← B jumps by stride N=4 again
//   Every kk step strides B by N; every jj step reloads A from scratch. Poor reuse.
//
// std::min clamps handle matrices where dimensions aren't multiples of tile_size.


void gemm_tiled(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N,
    int tile_size)
{
    // three outer loops over tiles, three inner loops within tile

    for (int i = 0; i < M; i += tile_size)
        for (int j = 0; j < N; j += tile_size)
            for (int k = 0; k < K; k += tile_size)
                for (int ii = i; ii < std::min(i + tile_size, M); ii++)
                    for (int kk = k; kk < std::min(k + tile_size, K); kk++)
                        for (int jj = j; jj < std::min(j + tile_size, N); jj++)
                            C[ii*N + jj] += A[ii*K + kk] * B[kk*N + jj];
}
