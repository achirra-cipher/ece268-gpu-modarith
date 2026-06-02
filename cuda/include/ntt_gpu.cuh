#pragma once
#include "modarith.cuh"
#include "params.h"
#include <vector>
#include <cstdint>

// ============================================================================
//  GPU (pure CUDA) Number Theoretic Transform over P32, using the Barrett
//  modular unit on-device. Iterative radix-2 Cooley-Tukey: one kernel launch
//  per stage. All twiddles come from a single precomputed table of n powers of
//  the primitive n-th root of unity, so a stage of length `len` reads
//  twiddle = roots[(n/len) * k].
//
//  This header is CUDA-only (compiled by nvcc). It is included by the .cu
//  drivers (ntt.cu, bench.cu).
// ============================================================================

namespace ma {

// ---- device kernels --------------------------------------------------------

__global__ void k_bitrev(u64 *a, u32 n, int log2n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // reverse the low log2n bits of i
    u32 x = i, r = 0;
    for (int s = 0; s < log2n; ++s) { r = (r << 1) | (x & 1u); x >>= 1; }
    if (i < r) { u64 t = a[i]; a[i] = a[r]; a[r] = t; }
}

// One butterfly stage. nb = n/2 butterflies total.
__global__ void k_ntt_stage(u64 *a, const u64 *roots, u32 n, u32 len,
                            Barrett32 b, u64 p) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    u32 nb = n >> 1;
    if (t >= nb) return;
    u32 half  = len >> 1;
    u32 k     = t & (half - 1);
    u32 group = t / half;
    u32 i     = group * len + k;
    u64 tw    = roots[(n / len) * k];
    u64 u = a[i];
    u64 v = mulmod32(a[i + half], tw, b);
    a[i]        = addmod32(u, v, p);
    a[i + half] = submod32(u, v, p);
}

__global__ void k_scale(u64 *a, u32 n, u64 inv_n, Barrett32 b) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = mulmod32(a[i], inv_n, b);
}

// ---- host helpers ----------------------------------------------------------

// Build the n-power twiddle table on the host (forward or inverse direction).
inline std::vector<u64> build_roots(u32 n, bool inverse_ntt) {
    Barrett32 b = barrett32_make(P32);
    u64 root = powmod32(P32_ROOT, (P32 - 1) / n, b);
    if (inverse_ntt) root = invmod32(root, b);
    std::vector<u64> roots(n);
    u64 cur = 1;
    for (u32 j = 0; j < n; ++j) { roots[j] = cur; cur = mulmod32(cur, root, b); }
    return roots;
}

// In-place NTT of a device array d_a (length n, power of two). d_roots must hold
// build_roots(n, inverse_ntt). Pure kernel launches; no host<->device traffic.
inline void ntt_gpu_inplace(u64 *d_a, const u64 *d_roots, u32 n,
                            bool inverse_ntt, int threads = 256) {
    Barrett32 b = barrett32_make(P32);
    int log2n = 0; while ((1u << log2n) < n) ++log2n;

    u32 gridN  = (n + threads - 1) / threads;
    u32 gridNB = ((n >> 1) + threads - 1) / threads;

    k_bitrev<<<gridN, threads>>>(d_a, n, log2n);
    for (u32 len = 2; len <= n; len <<= 1)
        k_ntt_stage<<<gridNB, threads>>>(d_a, d_roots, n, len, b, P32);

    if (inverse_ntt) {
        u64 inv_n = invmod32(n % P32, b);
        k_scale<<<gridN, threads>>>(d_a, n, inv_n, b);
    }
}

} // namespace ma
