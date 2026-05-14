#pragma once
// NxN Systolic Array — Phase 2 (June)
//
// Instantiates an N x N grid of PE modules.
// Weight-stationary: weights are loaded once, activations stream left-to-right,
// partial sums accumulate top-to-bottom.
//
// After Phase 2 this model should:
//   1. Accept a matrix A (activations) and B (weights)
//   2. Simulate the cycle-by-cycle data movement
//   3. Log total cycles, utilization %, stall cycles
//   4. Output matrix C that matches golden/gemm.cpp within 1e-4
