#include "pe.h"
#include <systemc.h>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// PE unit tests
//
// All tests share one sc_clock and one PE instance (SystemC does not allow
// re-elaboration after sc_start).  Between tests we reset PE::acc directly
// and drain/fill the fifos as needed.
//
// Test list:
//   1. Basic MAC correctness
//   2a. Edge — zero weight
//   2b. Edge — zero activations
//   2c. Edge — negative weight / activations
//   2d. Edge — unit weight (weight == 1.0f)
//   3.  Pass-through integrity
//   4.  Cumulative accumulation
//   5.  Numerical precision (K=8, mixed signs)
//   6.  K boundary — exactly K cycles consumed / forwarded
// ---------------------------------------------------------------------------

static const float EPS = 1e-4f;
static int g_pass = 0;
static int g_fail = 0;

static void check(const char* name, bool ok) {
    if (ok) { std::printf("  PASS  %s\n", name); ++g_pass; }
    else     { std::printf("  FAIL  %s\n", name); ++g_fail; }
}

// ---------------------------------------------------------------------------
// Top-level SC_MODULE that holds the single shared DUT
// ---------------------------------------------------------------------------
SC_MODULE(TB) {
    sc_clock        clk;
    sc_fifo<float>  act_in;
    sc_fifo<float>  act_out;
    PE              dut;

    SC_CTOR(TB)
        : clk("clk", 1, SC_NS)
        , act_in("act_in", 64)   // depth 64 — enough for the largest K used (K=8)
        , act_out("act_out", 64)
        , dut("dut")
    {
        dut.clk(clk);
        dut.act_in(act_in);
        dut.act_out(act_out);
        SC_THREAD(run_all);  // must run inside a process so wait() is legal
    }

    // -----------------------------------------------------------------------
    // Drive K activations, advance K clock cycles, drain act_out.
    // Called from SC_THREAD only — uses wait(), not sc_start().
    // -----------------------------------------------------------------------
    float run_test(float weight, const float* acts, int K, float* passthrough)
    {
        dut.weight = weight;
        dut.acc    = 0.0f;  // manual reset between tests

        for (int i = 0; i < K; ++i)
            act_in.write(acts[i]);

        wait(K, SC_NS);  // let PE's SC_CTHREAD run K cycles

        for (int i = 0; i < K; ++i)
            passthrough[i] = act_out.read();

        return dut.acc;
    }

    // -----------------------------------------------------------------------
    // 1. Correctness — basic MAC
    //    Feed K integer activations, verify acc == sum(weight * act[i]).
    // -----------------------------------------------------------------------
    void test_basic_mac() {
        std::printf("\n[1] Basic MAC correctness\n");
        const int K = 6;
        const float W = 3.0f;
        const float acts[K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        float ref = 0.0f;
        for (int i = 0; i < K; ++i) ref += W * acts[i];

        float passthrough[K];
        float got = run_test(W, acts, K, passthrough);

        check("acc == sum(weight * act[i])", std::fabs(got - ref) < EPS);
    }

    // -----------------------------------------------------------------------
    // 2a. Edge — zero weight
    // -----------------------------------------------------------------------
    void test_zero_weight() {
        std::printf("\n[2a] Zero weight\n");
        const int K = 4;
        const float acts[K] = {1.0f, 2.0f, 3.0f, 4.0f};

        float passthrough[K];
        float got = run_test(0.0f, acts, K, passthrough);

        check("acc == 0 with zero weight", got == 0.0f);
    }

    // -----------------------------------------------------------------------
    // 2b. Edge — zero activations
    // -----------------------------------------------------------------------
    void test_zero_activations() {
        std::printf("\n[2b] Zero activations\n");
        const int K = 4;
        const float acts[K] = {0.0f, 0.0f, 0.0f, 0.0f};

        float passthrough[K];
        float got = run_test(5.0f, acts, K, passthrough);

        check("acc == 0 with zero activations", got == 0.0f);
    }

    // -----------------------------------------------------------------------
    // 2c. Edge — negative weight and/or negative activations
    // -----------------------------------------------------------------------
    void test_negative_values() {
        std::printf("\n[2c] Negative weight / activations\n");
        const int K = 4;
        const float W = -2.0f;
        const float acts[K] = {1.0f, -3.0f, 2.0f, -4.0f};

        float ref = 0.0f;
        for (int i = 0; i < K; ++i) ref += W * acts[i];

        float passthrough[K];
        float got = run_test(W, acts, K, passthrough);

        check("acc correct with negative weight/activations", std::fabs(got - ref) < EPS);
    }

    // -----------------------------------------------------------------------
    // 2d. Edge — weight == 1.0f (acc should equal sum of activations)
    // -----------------------------------------------------------------------
    void test_unit_weight() {
        std::printf("\n[2d] Unit weight\n");
        const int K = 5;
        const float acts[K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        float ref = 0.0f;
        for (int i = 0; i < K; ++i) ref += acts[i];

        float passthrough[K];
        float got = run_test(1.0f, acts, K, passthrough);

        check("acc == sum(acts) for weight=1", std::fabs(got - ref) < EPS);
    }

    // -----------------------------------------------------------------------
    // 3. Pass-through integrity
    //    Every act_in value must appear on act_out unchanged and in order.
    //    Specifically guards against accidentally forwarding acc or weight.
    // -----------------------------------------------------------------------
    void test_passthrough() {
        std::printf("\n[3] Pass-through integrity\n");
        const int K = 6;
        const float W = 7.0f;
        const float acts[K] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

        float passthrough[K];
        run_test(W, acts, K, passthrough);

        bool value_ok = true;
        bool order_ok = true;
        for (int i = 0; i < K; ++i) {
            if (passthrough[i] != acts[i]) { value_ok = false; order_ok = false; }
        }
        check("act_out values match act_in values exactly", value_ok);
        check("act_out order matches act_in order",         order_ok);
    }

    // -----------------------------------------------------------------------
    // 4. Accumulation across cycles
    //    After K cycles acc must reflect all K MACs — not reset each cycle.
    //    Uses all-ones activations so partial sums are easy to reason about.
    // -----------------------------------------------------------------------
    void test_cumulative_accumulation() {
        std::printf("\n[4] Cumulative accumulation\n");
        const int K = 5;
        const float W = 2.0f;
        const float acts[K] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

        // After K cycles: acc == K * W * 1.0 == 10.0
        float expected = (float)K * W;

        float passthrough[K];
        float got = run_test(W, acts, K, passthrough);

        check("acc cumulative after K cycles (not reset per cycle)",
              std::fabs(got - expected) < EPS);
    }

    // -----------------------------------------------------------------------
    // 5. Numerical precision — K=8, mixed positive/negative values
    //    Reference computed in plain C++ in the same left-to-right order.
    //    Uses EPS = 1e-4f (not exact equality) — float has ~7 decimal digits.
    // -----------------------------------------------------------------------
    void test_numerical_precision() {
        std::printf("\n[5] Numerical precision (K=8, mixed signs)\n");
        const int K = 8;
        const float W = 1.5f;
        const float acts[K] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};

        float ref = 0.0f;
        for (int i = 0; i < K; ++i) ref += W * acts[i];  // same order as PE::run()

        float passthrough[K];
        float got = run_test(W, acts, K, passthrough);

        check("acc within 1e-4f of C++ reference (mixed signs)",
              std::fabs(got - ref) < EPS);
    }

    // -----------------------------------------------------------------------
    // 6. K boundary — exactly K cycles consumed and forwarded
    //    After draining act_out, both fifos must be empty.
    //    Confirms testbench timing matches what the array will expect.
    // -----------------------------------------------------------------------
    void test_k_boundary() {
        std::printf("\n[6] K boundary — exactly K cycles\n");
        const int K = 4;
        const float acts[K] = {1.0f, 2.0f, 3.0f, 4.0f};

        float passthrough[K];
        run_test(1.0f, acts, K, passthrough);

        // run_test drains act_out after wait(); both fifos should now be empty
        check("act_in empty after K cycles",           act_in.num_available()  == 0);
        check("act_out empty after draining K values", act_out.num_available() == 0);
    }

    // -----------------------------------------------------------------------
    // SC_THREAD entry — sequences all tests, then stops simulation
    // -----------------------------------------------------------------------
    void run_all() {
        test_basic_mac();
        test_zero_weight();
        test_zero_activations();
        test_negative_values();
        test_unit_weight();
        test_passthrough();
        test_cumulative_accumulation();
        test_numerical_precision();
        test_k_boundary();
        sc_stop();  // PE runs forever; stop once tests finish
    }
};

// ---------------------------------------------------------------------------
// sc_main — elaborate once, run all tests
// ---------------------------------------------------------------------------
int sc_main(int /*argc*/, char* /*argv*/[]) {
    std::printf("=== tb_pe: PE unit tests ===\n");

    TB tb("tb");
    sc_start();  // runs until run_all() calls sc_stop()

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
