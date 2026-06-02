#pragma once
#include "modarith.cuh"
#include "params.h"
#include <vector>
#include <cstddef>
#include <cstdint>

// ============================================================================
//  CPU reference Number Theoretic Transform (NTT) over the prime field P32,
//  using the Barrett-reduction modular unit. This is the "CPU equivalent" the
//  GPU is benchmarked against, and the correctness oracle for the GPU kernels.
//
//  Iterative radix-2 Cooley-Tukey (decimation-in-time) with bit-reversal.
//  Length n must be a power of two dividing 2^P32_TWO_ADICITY.
// ============================================================================

namespace ma {

inline u32 bit_reverse(u32 x, int log2n) {
    u32 r = 0;
    for (int i = 0; i < log2n; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

// Forward (inverse_ntt=false) or inverse (true) NTT, in place, on n elements.
// a[i] in [0, P32). Uses Barrett modmul throughout.
inline void ntt_cpu(std::vector<u64> &a, bool inverse_ntt) {
    const Barrett32 b = barrett32_make(P32);
    const u64 p = P32;
    const u32 n = (u32)a.size();
    int log2n = 0; while ((1u << log2n) < n) ++log2n;

    // bit-reversal permutation
    for (u32 i = 0; i < n; ++i) {
        u32 j = bit_reverse(i, log2n);
        if (i < j) { u64 t = a[i]; a[i] = a[j]; a[j] = t; }
    }

    // base primitive n-th root of unity: g^((p-1)/n); inverse uses its inverse
    u64 root = powmod32(P32_ROOT, (p - 1) / n, b);
    if (inverse_ntt) root = invmod32(root, b);

    for (u32 len = 2; len <= n; len <<= 1) {
        // w_len = root^(n/len) is a primitive len-th root of unity
        u64 wlen = powmod32(root, n / len, b);
        for (u32 i = 0; i < n; i += len) {
            u64 w = 1;
            for (u32 k = 0; k < len / 2; ++k) {
                u64 u = a[i + k];
                u64 v = mulmod32(a[i + k + len / 2], w, b);
                a[i + k]           = addmod32(u, v, p);
                a[i + k + len / 2] = submod32(u, v, p);
                w = mulmod32(w, wlen, b);
            }
        }
    }

    if (inverse_ntt) {
        u64 inv_n = invmod32(n % p, b);
        for (u32 i = 0; i < n; ++i) a[i] = mulmod32(a[i], inv_n, b);
    }
}

// Cyclic convolution of x and y (length n, power of two) via NTT:
//   z = INTT( NTT(x) .* NTT(y) ).
inline std::vector<u64> ntt_convolution(std::vector<u64> x, std::vector<u64> y) {
    const Barrett32 b = barrett32_make(P32);
    ntt_cpu(x, false);
    ntt_cpu(y, false);
    std::vector<u64> z(x.size());
    for (size_t i = 0; i < x.size(); ++i) z[i] = mulmod32(x[i], y[i], b);
    ntt_cpu(z, true);
    return z;
}

// Naive O(n^2) cyclic convolution, ground truth for correctness checks.
inline std::vector<u64> naive_convolution(const std::vector<u64> &x,
                                          const std::vector<u64> &y) {
    const Barrett32 b = barrett32_make(P32);
    const u64 p = P32;
    const size_t n = x.size();
    std::vector<u64> z(n, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            size_t k = (i + j) % n;          // cyclic
            z[k] = addmod32(z[k], mulmod32(x[i], y[j], b), p);
        }
    return z;
}

} // namespace ma
