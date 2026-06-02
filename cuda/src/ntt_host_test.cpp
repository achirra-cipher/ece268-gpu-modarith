// ============================================================================
//  ntt_host_test.cpp  --  Validate the CPU NTT (no GPU) before porting to CUDA.
//    1) round-trip: INTT(NTT(x)) == x
//    2) NTT convolution == naive O(n^2) convolution
//  Build:  make ntt_host_test
// ============================================================================
#include "ntt_cpu.h"
#include <cstdio>
#include <random>

using ma::u64;

int main() {
    std::mt19937_64 rng(2024);
    int failures = 0;

    // ---- round-trip over several sizes ----
    for (int log2n = 1; log2n <= 14; ++log2n) {
        size_t n = (size_t)1 << log2n;
        std::vector<u64> x(n), orig(n);
        for (auto &v : x) v = rng() % ma::P32;
        orig = x;
        ma::ntt_cpu(x, false);
        ma::ntt_cpu(x, true);
        for (size_t i = 0; i < n; ++i)
            if (x[i] != orig[i]) { ++failures; break; }
    }
    std::printf("round-trip (n up to 2^14): %s\n", failures ? "FAIL" : "OK");

    // ---- convolution vs naive (small n where O(n^2) is cheap) ----
    int conv_fail = 0;
    for (int log2n = 1; log2n <= 9; ++log2n) {
        size_t n = (size_t)1 << log2n;
        std::vector<u64> x(n), y(n);
        for (auto &v : x) v = rng() % ma::P32;
        for (auto &v : y) v = rng() % ma::P32;
        auto fast = ma::ntt_convolution(x, y);
        auto slow = ma::naive_convolution(x, y);
        for (size_t i = 0; i < n; ++i)
            if (fast[i] != slow[i]) { ++conv_fail; break; }
    }
    std::printf("convolution vs naive (n up to 2^9): %s\n", conv_fail ? "FAIL" : "OK");

    failures += conv_fail;
    std::printf(failures ? "\nNTT TESTS FAILED.\n" : "\nNTT TESTS PASSED.\n");
    return failures ? 1 : 0;
}
