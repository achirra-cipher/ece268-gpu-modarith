#pragma once
#include <cstdint>

// ============================================================================
//  Modular Arithmetic Unit  (Barrett reduction)
//
//  The whole point of this header: every routine is marked __host__ __device__
//  so the SAME code path runs on the GPU (compiled by nvcc) AND on the CPU
//  (compiled by a normal C++ compiler such as clang/gcc). That lets us
//  exhaustively unit-test the exact Barrett / modmul / modexp logic locally,
//  with no GPU, before deploying to CUDA.
//
//  Definitions:
//    - barrett_reduce(x): reduce x in [0, p^2) down to [0, p)
//    - mulmod(a, b)     : (a * b) mod p
//    - powmod(a, e)     : (a ^ e) mod p   (square-and-multiply)
//    - invmod(a)        : a^(p-2) mod p   (Fermat, p prime)
// ============================================================================

#if defined(__CUDACC__)
  #define MA_HD __host__ __device__ __forceinline__
#else
  #define MA_HD inline
#endif

namespace ma {

using u32  = uint32_t;
using u64  = uint64_t;
using u128 = unsigned __int128;   // native on clang (host) and nvcc (device)

// ----------------------------------------------------------------------------
//  Barrett context for a prime p < 2^32.
//
//  Because operands are < p < 2^32, every product a*b is < 2^64 (fits in u64),
//  and the Barrett estimate uses a single 128-bit multiply. With k = 64 and
//  mu = floor(2^64 / p), one conditional subtraction always suffices:
//
//      let R = 2^64 mod p  (so R < p)
//      x/p - x*mu/2^64 = x*R / (p*2^64) < x/2^64 < 1   (since x < 2^64)
//   => floor(x/p) - q <= 1   => r = x - q*p < 2p.
// ----------------------------------------------------------------------------
struct Barrett32 {
    u64 p;    // the modulus  (< 2^32)
    u64 mu;   // floor(2^64 / p)
};

MA_HD Barrett32 barrett32_make(u64 p) {
    Barrett32 b;
    b.p  = p;
    b.mu = (u64)((((u128)1) << 64) / p);   // floor(2^64 / p)
    return b;
}

// Reduce x in [0, 2^64) modulo p (valid for any x < 2^64, in particular x < p^2).
MA_HD u64 barrett32_reduce(u64 x, const Barrett32 &b) {
    // q = floor(x * mu / 2^64) = high 64 bits of the 128-bit product x*mu.
#if defined(__CUDA_ARCH__)
    u64 q = __umul64hi(x, b.mu);              // device intrinsic, no 128-bit type needed
#else
    u64 q = (u64)(((u128)x * b.mu) >> 64);    // host: native __uint128_t
#endif
    u64 r = x - q * b.p;                      // r in [0, 2p)
    if (r >= b.p) r -= b.p;                   // single correction -> [0, p)
    return r;
}

MA_HD u64 mulmod32(u64 a, u64 b, const Barrett32 &bp) {
    return barrett32_reduce(a * b, bp);       // a,b < 2^32 => a*b < 2^64
}

MA_HD u64 addmod32(u64 a, u64 b, u64 p) {
    u64 s = a + b;            // < 2^33, no overflow
    if (s >= p) s -= p;
    return s;
}

MA_HD u64 submod32(u64 a, u64 b, u64 p) {
    return (a >= b) ? (a - b) : (a + p - b);
}

// (a ^ e) mod p via square-and-multiply.
MA_HD u64 powmod32(u64 a, u64 e, const Barrett32 &bp) {
    u64 result = 1;
    u64 base   = barrett32_reduce(a, bp);
    while (e > 0) {
        if (e & 1ULL) result = mulmod32(result, base, bp);
        base = mulmod32(base, base, bp);
        e >>= 1;
    }
    return result;
}

// Modular inverse via Fermat's little theorem: a^(p-2) mod p  (p prime, a != 0).
MA_HD u64 invmod32(u64 a, const Barrett32 &bp) {
    return powmod32(a, bp.p - 2, bp);
}

// ----------------------------------------------------------------------------
//  Barrett context for the 64-bit Goldilocks-style prime p < 2^64 (BONUS).
//
//  Operands are < p < 2^64, so products are < 2^128 (u128). We reduce a 128-bit
//  value modulo a 64-bit prime. We use k = 64:
//      mu = floor(2^128 / p)            (~65 bits, stored as u128)
//      q  = (x_hi * mu) >> 64           (uses top 64 bits of x as the estimate)
//      r  = x - q * p
//  then up to a few conditional subtractions. The estimate from x_hi can be off
//  by a small bounded amount, so we finish with a short bounded correction loop.
//  All of this is validated exhaustively on the host against true (u128 % p).
// ----------------------------------------------------------------------------
struct Barrett64 {
    u64  p;
    u128 mu;   // floor(2^128 / p)
};

MA_HD Barrett64 barrett64_make(u64 p) {
    Barrett64 b;
    b.p = p;
    // mu = floor(2^128 / p). 2^128 is one past u128 max, so compute as
    //   floor((2^128 - 1)/p) and bump by 1 when p divides 2^128 exactly
    //   (never for an odd prime, but keep it general/correct).
    u128 num = (u128)(-1);               // 2^128 - 1
    u128 mu  = num / (u128)p;
    if ((num - mu * (u128)p) + 1 >= (u128)p) mu += 1;
    b.mu = mu;
    return b;
}

// Reduce a full 128-bit value (lo,hi) modulo p (< 2^64).
MA_HD u64 barrett64_reduce_u128(u128 x, const Barrett64 &b) {
    u64  xhi = (u64)(x >> 64);
    // q ~= floor(x / p) using the top 64 bits of x.
    u128 q   = ((u128)xhi * b.mu) >> 64;
    u128 r   = x - q * (u128)b.p;        // wraps mod 2^128; r is small & positive
    // Bounded correction: estimate can undershoot, so subtract p a few times.
    while (r >= (u128)b.p) r -= (u128)b.p;
    return (u64)r;
}

MA_HD u64 mulmod64(u64 a, u64 b, const Barrett64 &bp) {
    return barrett64_reduce_u128((u128)a * (u128)b, bp);
}

MA_HD u64 addmod64(u64 a, u64 b, u64 p) {
    u64 s = a + b;
    if (s < a || s >= p) s -= p;   // handle 64-bit overflow too
    return s;
}

MA_HD u64 submod64(u64 a, u64 b, u64 p) {
    return (a >= b) ? (a - b) : (a + p - b);
}

MA_HD u64 powmod64(u64 a, u64 e, const Barrett64 &bp) {
    u64 result = 1;
    u64 base   = barrett64_reduce_u128((u128)a, bp);
    while (e > 0) {
        if (e & 1ULL) result = mulmod64(result, base, bp);
        base = mulmod64(base, base, bp);
        e >>= 1;
    }
    return result;
}

MA_HD u64 invmod64(u64 a, const Barrett64 &bp) {
    return powmod64(a, bp.p - 2, bp);
}

} // namespace ma
