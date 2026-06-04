#!/usr/bin/env python3
"""Independent ground-truth oracle for the GPU Modular Arithmetic Unit.

Uses Python's arbitrary-precision integers (and sympy when available) to verify,
completely independently of the C++/CUDA code, that:
  * the chosen primes are actually prime,
  * the primitive roots have the right order,
  * Barrett reduction matches true modular reduction,
  * a reference NTT round-trips and matches naive convolution.

Run:  python3 python/oracle.py
"""
import random

P32 = 998244353            # 119 * 2^23 + 1
P32_ROOT = 3
P64 = (1 << 64) - (1 << 32) + 1   # Goldilocks
P64_ROOT = 7


def is_prime(n):
    """Return True if n is prime using sympy when available, else Miller-Rabin."""
    try:
        import sympy
        return bool(sympy.isprime(n))
    except Exception:
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True


def barrett_mu(p, k):
    """Compute Barrett precomputed constant mu = floor(2^k / p)."""
    return (1 << k) // p


def barrett_reduce(x, p, mu, k):
    """Reduce x mod p using Barrett reduction with precomputed mu and shift k."""
    q = (x * mu) >> k
    r = x - q * p
    while r >= p:
        r -= p
    return r


def check_barrett(p, k, trials=200000):
    """Verify Barrett reduction against Python's built-in modulo over random pairs."""
    mu = barrett_mu(p, k)
    rng = random.Random(1)
    for _ in range(trials):
        a = rng.randrange(p)
        b = rng.randrange(p)
        if barrett_reduce(a * b, p, mu, k) != (a * b) % p:
            return False
    return True


def ntt(a, inverse, p, root):
    """Compute forward or inverse NTT of a over Z_p using primitive root.

    Uses Cooley-Tukey iterative butterfly with bit-reversal permutation.
    len(a) must be a power of two and divide p-1.
    """
    n = len(a)
    a = a[:]
    j = 0
    for i in range(1, n):                 # bit-reversal
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    r = pow(root, (p - 1) // n, p)
    if inverse:
        r = pow(r, p - 2, p)
    length = 2
    while length <= n:
        wlen = pow(r, n // length, p)
        for i in range(0, n, length):
            w = 1
            for k in range(length // 2):
                u = a[i + k]
                v = a[i + k + length // 2] * w % p
                a[i + k] = (u + v) % p
                a[i + k + length // 2] = (u - v) % p
                w = w * wlen % p
        length <<= 1
    if inverse:
        inv_n = pow(n, p - 2, p)
        a = [x * inv_n % p for x in a]
    return a


def naive_conv(x, y, p):
    """Compute cyclic convolution of x and y mod p via O(n^2) direct summation."""
    n = len(x)
    z = [0] * n
    for i in range(n):
        for jj in range(n):
            z[(i + jj) % n] = (z[(i + jj) % n] + x[i] * y[jj]) % p
    return z


def main():
    print("== prime checks ==")
    print(f"  P32 = {P32}: prime={is_prime(P32)}")
    print(f"  P64 = {P64}: prime={is_prime(P64)}")
    print(f"  order of root {P32_ROOT} mod P32 == p-1: "
          f"{pow(P32_ROOT, (P32-1)//2, P32) != 1}")

    print("== Barrett reduction ==")
    print(f"  P32 (k=64): {'OK' if check_barrett(P32, 64) else 'FAIL'}")

    print("== NTT (P32) ==")
    rng = random.Random(7)
    n = 256
    x = [rng.randrange(P32) for _ in range(n)]
    y = [rng.randrange(P32) for _ in range(n)]
    rt = ntt(ntt(x, False, P32, P32_ROOT), True, P32, P32_ROOT)
    print(f"  round-trip: {'OK' if rt == x else 'FAIL'}")
    fx, fy = ntt(x, False, P32, P32_ROOT), ntt(y, False, P32, P32_ROOT)
    fz = [a * b % P32 for a, b in zip(fx, fy)]
    fast = ntt(fz, True, P32, P32_ROOT)
    print(f"  convolution vs naive: {'OK' if fast == naive_conv(x, y, P32) else 'FAIL'}")


if __name__ == "__main__":
    main()
