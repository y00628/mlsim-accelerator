// Systolic array testbench — Phase 2 (June)
//
// Flow:
//   1. Generate random A (M x K) and B (K x N)
//   2. Compute reference C = gemm_naive(A, B)
//   3. Run systolic array simulation
//   4. Compare simulation output to reference
//   5. Print cycle count and utilization stats
