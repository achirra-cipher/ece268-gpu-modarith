// ============================================================================
//  modexp_demo.cu  --  Batched modular exponentiation over the LARGE prime
//  field P64 (Goldilocks 2^64-2^32+1), using Barrett64.
//
//  This is the kernel shape behind RSA decryption / Diffie-Hellman / batch
//  signature verification: many *independent* exponentiations base^exp mod p,
//  one per thread. Perfectly parallel -> great GPU fit, even though a single
//  exponentiation is sequential (square-and-multiply).
//
//  Validates GPU == CPU and reports GPU vs CPU throughput.
//  Build: make modexp_demo   Run: ./build/modexp_demo
// ============================================================================
#include "modarith.cuh"
#include "params.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include <sys/stat.h>

using ma::u32;
using ma::u64;
using ma::Barrett64;

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _cuda_err_ = (call);                                   \
    if (_cuda_err_ != cudaSuccess) {                                   \
        std::printf("CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_cuda_err_), __FILE__, __LINE__);\
        std::exit(1);                                                  \
    } } while (0)

__global__ void k_batch_modexp(const u64 *base, const u64 *exp, u64 *out,
                               u32 n, Barrett64 bp) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = ma::powmod64(base[i], exp[i], bp);
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    mkdir("results", 0755);
    std::mt19937_64 rng(2025);
    Barrett64 bp = ma::barrett64_make(ma::P64);
    const u32 n = 1u << 20;             // ~1M independent exponentiations
    const int threads = 256;

    std::vector<u64> base(n), exp(n), gpu(n), cpu(n);
    for (u32 i = 0; i < n; ++i) {
        base[i] = 1 + rng() % (ma::P64 - 1);
        exp[i]  = rng();               // full 64-bit exponents
    }

    u64 *db, *de, *doo;
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&de, n * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&doo, n * sizeof(u64)));
    CUDA_CHECK(cudaMemcpy(db, base.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(de, exp.data(),  n * sizeof(u64), cudaMemcpyHostToDevice));

    u32 grid = (n + threads - 1) / threads;
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    k_batch_modexp<<<grid, threads>>>(db, de, doo, n, bp);   // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    k_batch_modexp<<<grid, threads>>>(db, de, doo, n, bp);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, s, e);
    CUDA_CHECK(cudaMemcpy(gpu.data(), doo, n * sizeof(u64), cudaMemcpyDeviceToHost));

    // CPU reference (same Barrett64 path)
    double t0 = now_ms();
    for (u32 i = 0; i < n; ++i) cpu[i] = ma::powmod64(base[i], exp[i], bp);
    double cpu_ms = now_ms() - t0;

    u32 mism = 0;
    for (u32 i = 0; i < n; ++i) if (gpu[i] != cpu[i]) ++mism;

    std::printf("[modexp P64] n=%u  GPU=%.3f ms  CPU=%.1f ms  speedup=%.1fx  "
                "mismatches=%u\n", n, gpu_ms, cpu_ms, cpu_ms / gpu_ms, mism);

    FILE *csv = std::fopen("results/bench_modexp.csv", "w");
    std::fprintf(csv, "n,gpu_ms,cpu_ms,speedup,mismatches\n");
    std::fprintf(csv, "%u,%.4f,%.4f,%.2f,%u\n", n, gpu_ms, cpu_ms, cpu_ms / gpu_ms, mism);
    std::fclose(csv);
    std::printf("Wrote results/bench_modexp.csv\n");

    cudaFree(db); cudaFree(de); cudaFree(doo);
    return mism ? 1 : 0;
}
