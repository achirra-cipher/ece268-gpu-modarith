// ============================================================================
//  host_test.cpp  --  Local correctness harness (NO GPU REQUIRED).
//
//  Compiles modarith.cuh as plain C++ and checks the exact Barrett / modmul /
//  modexp / modinv logic against ground-truth __uint128_t arithmetic. This is
//  the same code that will run on the GPU, so passing here gives high
//  confidence before deploying to Colab.
//
//  Build:  make host_test     (or: clang++ -std=c++17 -O2 -Icuda/include ...)
// ============================================================================
#include "modarith.cuh"
#include "params.h"

#include <cstdio>
#include <cstdint>
#include <random>

using ma::u64;
using ma::u128;

static int g_failures = 0;

static void check(bool cond, const char *name, u64 a, u64 b, u64 got, u64 want) {
    if (!cond) {
        if (g_failures < 10)
            std::printf("  FAIL %-18s a=%llu b=%llu got=%llu want=%llu\n",
                        name, (unsigned long long)a, (unsigned long long)b,
                        (unsigned long long)got, (unsigned long long)want);
        ++g_failures;
    }
}

// Reference modexp using u128 reduction (independent of Barrett).
static u64 ref_powmod(u64 a, u64 e, u64 p) {
    u128 result = 1, base = (u128)(a % p);
    while (e) {
        if (e & 1) result = (result * base) % p;
        base = (base * base) % p;
        e >>= 1;
    }
    return (u64)result;
}

int main() {
    std::mt19937_64 rng(12345);
    const int N = 2'000'000;

    // ---- P32 (30-bit prime): Barrett32 ------------------------------------
    {
        ma::Barrett32 b = ma::barrett32_make(ma::P32);
        const u64 p = ma::P32;
        std::printf("[P32 = %llu]  running %d random checks...\n",
                    (unsigned long long)p, N);

        // boundary + random operands
        for (int i = 0; i < N; ++i) {
            u64 a = (i == 0) ? 0 : (i == 1 ? p - 1 : rng() % p);
            u64 c = (i == 1) ? p - 1 : rng() % p;

            u64 got_mul = ma::mulmod32(a, c, b);
            u64 want_mul = (u64)(((u128)a * c) % p);
            check(got_mul == want_mul, "mulmod32", a, c, got_mul, want_mul);

            u64 got_add = ma::addmod32(a, c, p);
            u64 want_add = (u64)(((u128)a + c) % p);
            check(got_add == want_add, "addmod32", a, c, got_add, want_add);

            u64 got_sub = ma::submod32(a, c, p);
            u64 want_sub = (u64)(((u128)a + p - c) % p);
            check(got_sub == want_sub, "submod32", a, c, got_sub, want_sub);
        }
        // also test full-range reduce (x up to 2^64-1)
        for (int i = 0; i < N; ++i) {
            u64 x = rng();
            u64 got = ma::barrett32_reduce(x, b);
            u64 want = x % p;
            check(got == want, "barrett32_reduce", x, 0, got, want);
        }
        // modexp + Fermat inverse
        for (int i = 0; i < 100000; ++i) {
            u64 a = 1 + rng() % (p - 1);
            u64 e = rng();
            u64 got = ma::powmod32(a, e, b);
            u64 want = ref_powmod(a, e, p);
            check(got == want, "powmod32", a, e, got, want);

            u64 inv = ma::invmod32(a, b);
            u64 prod = ma::mulmod32(a, inv, b);
            check(prod == 1, "invmod32", a, inv, prod, 1);
        }
        std::printf("  P32 done. failures so far = %d\n", g_failures);
    }

    // ---- P64 (Goldilocks 64-bit prime): Barrett64 -------------------------
    {
        ma::Barrett64 b = ma::barrett64_make(ma::P64);
        const u64 p = ma::P64;
        std::printf("[P64 = %llu]  running %d random checks...\n",
                    (unsigned long long)p, N);

        for (int i = 0; i < N; ++i) {
            // include hard boundary operands near p
            u64 a = (i == 0) ? 0 : (i == 1 ? p - 1 : (i == 2 ? p - 2 : rng() % p));
            u64 c = (i == 1) ? p - 1 : (i == 2 ? p - 1 : rng() % p);

            u64 got_mul = ma::mulmod64(a, c, b);
            u64 want_mul = (u64)(((u128)a * c) % p);
            check(got_mul == want_mul, "mulmod64", a, c, got_mul, want_mul);

            u64 got_add = ma::addmod64(a, c, p);
            u64 want_add = (u64)(((u128)a + c) % p);
            check(got_add == want_add, "addmod64", a, c, got_add, want_add);
        }
        for (int i = 0; i < 100000; ++i) {
            u64 a = 1 + rng() % (p - 1);
            u64 e = rng();
            u64 got = ma::powmod64(a, e, b);
            u64 want = ref_powmod(a, e, p);
            check(got == want, "powmod64", a, e, got, want);

            u64 inv = ma::invmod64(a, b);
            u64 prod = ma::mulmod64(a, inv, b);
            check(prod == 1, "invmod64", a, inv, prod, 1);
        }
        std::printf("  P64 done. failures so far = %d\n", g_failures);
    }

    if (g_failures == 0) {
        std::printf("\nALL TESTS PASSED.\n");
        return 0;
    }
    std::printf("\n%d FAILURES.\n", g_failures);
    return 1;
}
