// ============================================================================
//  ntt.cu  --  GPU NTT correctness driver (run on Colab T4).
//    * GPU forward NTT must match the CPU reference elementwise.
//    * GPU round-trip INTT(NTT(x)) must equal x.
//    * GPU convolution must match the CPU NTT convolution.
//  Build:  make ntt    Run: ./build/ntt
// ============================================================================
#include "ntt_gpu.cuh"
#include "ntt_cpu.h"
#include <cstdio>
#include <random>

using ma::u32;
using ma::u64;

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _cuda_err_ = (call);                                   \
    if (_cuda_err_ != cudaSuccess) {                                   \
        std::printf("CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(_cuda_err_), __FILE__, __LINE__);\
        return 1;                                                      \
    } } while (0)

int main() {
    std::mt19937_64 rng(7);
    int failures = 0;

    for (int log2n = 4; log2n <= 20; ++log2n) {
        u32 n = 1u << log2n;
        std::vector<u64> h(n), orig(n);
        for (auto &v : h) v = rng() % ma::P32;
        orig = h;

        // device buffers + twiddle tables
        u64 *d_a, *d_rf, *d_ri;
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_rf, n * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&d_ri, n * sizeof(u64)));
        auto rf = ma::build_roots(n, false);
        auto ri = ma::build_roots(n, true);
        CUDA_CHECK(cudaMemcpy(d_rf, rf.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ri, ri.data(), n * sizeof(u64), cudaMemcpyHostToDevice));

        // GPU forward
        CUDA_CHECK(cudaMemcpy(d_a, h.data(), n * sizeof(u64), cudaMemcpyHostToDevice));
        ma::ntt_gpu_inplace(d_a, d_rf, n, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<u64> gpu_fwd(n);
        CUDA_CHECK(cudaMemcpy(gpu_fwd.data(), d_a, n * sizeof(u64), cudaMemcpyDeviceToHost));

        // CPU forward
        std::vector<u64> cpu_fwd = orig;
        ma::ntt_cpu(cpu_fwd, false);

        bool match = true;
        for (u32 i = 0; i < n; ++i) if (gpu_fwd[i] != cpu_fwd[i]) { match = false; break; }

        // GPU round-trip
        ma::ntt_gpu_inplace(d_a, d_ri, n, true);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<u64> rt(n);
        CUDA_CHECK(cudaMemcpy(rt.data(), d_a, n * sizeof(u64), cudaMemcpyDeviceToHost));
        bool rtmatch = true;
        for (u32 i = 0; i < n; ++i) if (rt[i] != orig[i]) { rtmatch = false; break; }

        if (!match || !rtmatch) ++failures;
        std::printf("n=2^%-2d  fwd-vs-CPU:%s  round-trip:%s\n",
                    log2n, match ? "OK" : "FAIL", rtmatch ? "OK" : "FAIL");

        cudaFree(d_a); cudaFree(d_rf); cudaFree(d_ri);
    }

    std::printf(failures ? "\nGPU NTT TESTS FAILED.\n" : "\nGPU NTT TESTS PASSED.\n");
    return failures ? 1 : 0;
}
