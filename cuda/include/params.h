#pragma once
#include <cstdint>

// ============================================================================
// Prime-field parameters for the GPU Modular Arithmetic Unit.
//
// P32: NTT-friendly 30-bit prime. All products fit in 64 bits, so Barrett
//      reduction uses only native __uint128_t intermediates -> bulletproof.
// P64: Goldilocks prime 2^64 - 2^32 + 1, a genuinely large field used in real
//      zk-proof systems. Barrett needs wider intermediates (bonus path).
// ============================================================================

namespace ma {

// ---- Primary 30-bit NTT prime: 998244353 = 119 * 2^23 + 1 -------------------
// Multiplicative generator g = 3. Group order p-1 = 2^23 * 7 * 17.
// => supports power-of-two NTT lengths up to 2^23.
static constexpr uint64_t P32       = 998244353ULL;
static constexpr uint64_t P32_ROOT  = 3ULL;          // primitive root mod P32
static constexpr uint32_t P32_TWO_ADICITY = 23;       // 2^23 | (P32 - 1)

// ---- Bonus large prime: Goldilocks p = 2^64 - 2^32 + 1 ----------------------
// = 18446744069414584321. Generator g = 7. 2-adicity = 32.
static constexpr uint64_t P64       = 0xFFFFFFFF00000001ULL;
static constexpr uint64_t P64_ROOT  = 7ULL;
static constexpr uint32_t P64_TWO_ADICITY = 32;

} // namespace ma
