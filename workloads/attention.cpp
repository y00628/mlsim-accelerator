#include <vector>
#include <cmath>
// Scaled dot-product attention (Phase 3, July)
//
// attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// This decomposes into two GEMMs:
//   1. S = Q * K^T        shape: (seq_len x seq_len)
//   2. out = softmax(S/sqrt(d_k)) * V   shape: (seq_len x d_v)
//
// Arithmetic intensity:
//   GEMM 1: 2 * seq_len^2 * d_k FLOPs, modest reuse
//   GEMM 2: 2 * seq_len^2 * d_v FLOPs
//   Softmax: memory-bound (elementwise) — pulls AI down
//
// This is how you'll explain to interviewers why attention is
// memory-bandwidth-limited at long sequence lengths.

// TODO July: implement using gemm_naive as backend,
// measure arithmetic intensity, add to roofline plot
