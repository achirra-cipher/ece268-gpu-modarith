// ============================================================================
//  bench.cu  --  GPU vs CPU benchmarks for the Barrett modular unit (Colab T4).
//    1) Pointwise modular multiply throughput (raw modmul/Barrett op rate).
//    2) Forward NTT latency across sizes.
//  Emits CSV to results/ for plotting. CUDA events time the GPU; std::chrono
//  times the single-thread CPU reference (same Barrett code path).
//
//  Build: make bench   Run: ./build/bench
// ============================================================================
#include "ntt_gpu.cuh"
#include "ntt_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <sys/stat.h>

using ma::u32;
using ma::u64;
using ma::Barrett32;

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _cuda_err_ = (call);                                   \
    if (_cuda_err_ != cudaSuccess) {                                   \
        std::printf("CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_cuda_err_), __FILE__, __LINE__);\
        std::exit(1);                                                  \
    } } while (0)

// Pointwise c[i] = a[i] * b[i] mod p, repeated `iters` times (fused chain to
// keep the compiler honest), exercising raw Barrett modmul throughput.
__global__ void k_modmul(const u64 *a, const u64 *b, u64 *c, u32 n,
                         int iters, Barrett32 bp) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 x = a[i];
    u64 y = b[i];
    for (int t = 0; t < iters; ++t) x = ma::mulmod32(x, y, bp);
    c[i] = x;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    mkdir("results", 0755);
    std::mt19937_64 rng(99);
    Barrett32 bp = ma::barrett32_make(ma::P32);
    const int threads = 256;

    // ===================== (1) modmul throughput =========================
    {
        FILE *csv = std::fopen("results/bench_modmul.csv", "w");
        std::fprintf(csv, "n,iters,gpu_ms,cpu_ms,gpu_gops,cpu_gops,speedup\n");
        const u32 n = 1u << 22;          // 4M lanes
        const int iters = 64;            // mults per lane
        std::vector<u64> a(n), b(n), c(n);
        for (u32 i = 0; i < n; ++i) { a[i] = rng() % ma::P32; b[i] = rng() % ma::P32; }

        u64 *da, *db, *dc;
        CUDA_CHECK(cudaMalloc(&da, n * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&db, n * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&dc, n * sizeof(u64)));
        CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(u64), cudaMemcpyHostToDevice));

        u32 grid = (n + threads - 1) / threads;
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        k_modmul<<<grid, threads>>>(da, db, dc, n, iters, bp);   // warmup
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(s);
        k_modmul<<<grid, threads>>>(da, db, dc, n, iters, bp);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, s, e);

        // CPU reference (same Barrett path)
        double t0 = now_ms();
        for (u32 i = 0; i < n; ++i) {
            u64 x = a[i], y = b[i];
            for (int t = 0; t < iters; ++t) x = ma::mulmod32(x, y, bp);
            c[i] = x;
        }
        double cpu_ms = now_ms() - t0;

        double ops = (double)n * iters;
        double gpu_gops = ops / (gpu_ms * 1e6);
        double cpu_gops = ops / (cpu_ms * 1e6);
        std::printf("[modmul] n=%u iters=%d  GPU=%.3f ms (%.2f Gops/s)  "
                    "CPU=%.1f ms (%.2f Gops/s)  speedup=%.1fx\n",
                    n, iters, gpu_ms, gpu_gops, cpu_ms, cpu_gops, cpu_ms / gpu_ms);
        std::fprintf(csv, "%u,%d,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                     n, iters, gpu_ms, cpu_ms, gpu_gops, cpu_gops, cpu_ms / gpu_ms);
        std::fclose(csv);
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    // ===================== (2) forward NTT latency =======================
    {
        FILE *csv = std::fopen("results/bench_ntt.csv", "w");
        std::fprintf(csv, "log2n,n,gpu_kernel_ms,gpu_total_ms,cpu_ms,speedup_kernel,speedup_total\n");
        for (int log2n = 10; log2n <= 22; ++log2n) {
            u32 n = 1u << log2n;
            std::vector<u64> h(n);
            for (auto &v : h) v = rng() % ma::P32;

            u64 *d_a, *d_r;
            CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(u64)));
            CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(u64)));
            auto rf = ma::build_roots(n, false);
            CUDA_CHECK(cudaMemcpy(d_r, rf.data(), n * sizeof(u64), cudaMemcpyHostToDevice));

            cudaEvent_t s, e, s2, e2;
            cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventCreate(&s2); cudaEventCreate(&e2);
            std::vector<u64> back(n);

            // warmup
            CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
            ma::ntt_gpu_inplace(d_a, d_r, n, false);
            CUDA_CHECK(cudaDeviceSynchronize());

            // (A) kernel-only window: data already resident on device.
            cudaEventRecord(s);
            ma::ntt_gpu_inplace(d_a, d_r, n, false);
            cudaEventRecord(e);
            CUDA_CHECK(cudaEventSynchronize(e));

            // (B) end-to-end window: H2D + one NTT + D2H (realistic offload cost).
            cudaEventRecord(s2);
            CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
            ma::ntt_gpu_inplace(d_a, d_r, n, false);
            CUDA_CHECK(cudaMemcpy(back.data(), d_a, n * sizeof(u64), cudaMemcpyDeviceToHost));
            cudaEventRecord(e2);
            CUDA_CHECK(cudaEventSynchronize(e2));

            float kernel_ms = 0, total_ms = 0;
            cudaEventElapsedTime(&kernel_ms, s, e);
            cudaEventElapsedTime(&total_ms, s2, e2);

            // CPU reference
            std::vector<u64> hc = h;
            double t0 = now_ms();
            ma::ntt_cpu(hc, false);
            double cpu_ms = now_ms() - t0;

            std::printf("[ntt] n=2^%-2d  GPU kernel=%.3f ms total=%.3f ms  "
                        "CPU=%.2f ms  speedup(kernel)=%.1fx total=%.1fx\n",
                        log2n, kernel_ms, total_ms, cpu_ms,
                        cpu_ms / kernel_ms, cpu_ms / total_ms);
            std::fprintf(csv, "%d,%u,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                         log2n, n, kernel_ms, total_ms, cpu_ms,
                         cpu_ms / kernel_ms, cpu_ms / total_ms);

            cudaFree(d_a); cudaFree(d_r);
        }
        std::fclose(csv);
    }

    std::printf("\nWrote results/bench_modmul.csv and results/bench_ntt.csv\n");
    return 0;
}
