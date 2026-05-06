#include <cassert>
#include <cmath>
#include <iostream>
#include "gemm.hpp"

// Helper: compare two matrices element-wise within epsilon
static void check_equal(const std::vector<float>& got,
                        const std::vector<float>& expected,
                        const char* test_name,
                        float eps = 1e-4f)
{
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); i++) {
        if (std::abs(got[i] - expected[i]) >= eps) {
            std::cerr << test_name << " FAILED at index " << i
                      << ": got " << got[i] << " expected " << expected[i] << "\n";
            std::exit(1);
        }
    }
    std::cout << test_name << " passed\n";
}

// 2x2 hand-computed case
// A = [[1,2],[3,4]], B = [[5,6],[7,8]]
// C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
void test_2x2() {
    std::vector<float> A = {1,2,3,4};
    std::vector<float> B = {5,6,7,8};
    std::vector<float> C(4, 0.0f);
    gemm_naive(A, B, C, 2, 2, 2);
    check_equal(C, {19,22,43,50}, "test_2x2");
}

// A * I = A  (identity matrix check)
void test_identity() {
    int N = 4;
    std::vector<float> A(N*N, 0.0f), I(N*N, 0.0f), C(N*N, 0.0f);
    for (int i = 0; i < N; i++) { A[i*N+i] = float(i+1); I[i*N+i] = 1.0f; }
    gemm_naive(A, I, C, N, N, N);
    check_equal(C, A, "test_identity");
}

// Non-square: M=2, K=3, N=4
void test_nonsquare() {
    // A(2x3) * B(3x4) = C(2x4)
    std::vector<float> A = {1,2,3, 4,5,6};
    std::vector<float> B = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    std::vector<float> C(2*4, 0.0f);
    gemm_naive(A, B, C, 2, 3, 4);
    // C should equal A padded with a zero column
    std::vector<float> expected = {1,2,3,0, 4,5,6,0};
    check_equal(C, expected, "test_nonsquare");
}

int main() {
    test_2x2();
    test_identity();
    test_nonsquare();
    std::cout << "All tests passed.\n";
    return 0;
}
