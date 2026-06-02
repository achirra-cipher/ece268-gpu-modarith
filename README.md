# GPU Modular Arithmetic Unit with Barrett Reduction

**ECE 268 Final Project — GPU Cryptographic Primitives**

> Implement cryptographic modular-arithmetic primitives **from scratch on the GPU** (pure
> CUDA), benchmark them against CPU equivalents, and analyze the performance/security
> trade-offs. Track: **Modular Arithmetic Unit**. Reduction technique: **Barrett reduction**.
> Real cryptographic application: **Number Theoretic Transform (NTT)**.

---

## 1. Summary (TL;DR)

We build a modular-arithmetic engine over a large prime field that supports:

| Required op | Status | Where |
|---|---|---|
| Modular multiplication (`modmul`) | done, validated locally | `cuda/include/modarith.cuh` |
| Modular exponentiation (`modexp`, square-and-multiply) | done, validated locally | `cuda/include/modarith.cuh` |
| **Barrett reduction** (chosen technique) | done, validated locally | `cuda/include/modarith.cuh` |
| Modular inverse (Fermat, bonus extra) | done, validated locally | `cuda/include/modarith.cuh` |

The unit is exercised by a **Number Theoretic Transform (NTT)** — the modular-arithmetic core of
lattice-based post-quantum cryptography (Kyber/Dilithium) and zero-knowledge proof systems. NTT is
chosen because it is *massively parallel* (FFT-style butterflies) and *modmul-bound*, making it an
ideal showcase for a GPU Barrett-reduction unit.

We benchmark **pure-CUDA GPU** implementations against single-thread (and optionally
multi-thread) **CPU C++** implementations of the *same* algorithms, on the *same* primes.

---

## 2. Context & key constraints

- **Dev machine:** Apple Silicon Mac (arm64). **No NVIDIA GPU / no CUDA / no `nvcc` locally.**
- **Execution target:** **Google Colab (free NVIDIA T4 GPU)**. We compile with `nvcc` and run there.
- **Implementation style:** **pure CUDA** (`.cu` + `nvcc` + C++ host) to maximize bonus points.
- **De-risking strategy:** All core arithmetic lives in a header (`modarith.cuh`) marked
  `__host__ __device__`. It **also compiles as plain C++**, so we unit-test the *exact* Barrett /
  modmul / modexp logic locally on the Mac (clang has native `__uint128_t`) *before* running on the GPU.
- **Toolchain verified locally:** Apple clang 21 (C++17, `__uint128_t` OK), Python 3.9 with
  `sympy`, `numpy`, `matplotlib`.
- **Deadline:** June 2–4.

### Prime fields used

| Name | Prime | Bits | Why |
|---|---|---|---|
| `P32` | `998244353 = 119·2^23 + 1` | 30 | NTT-friendly (2-adicity 23, root g=3). All products fit in 64-bit → Barrett uses native `__uint128_t`, bulletproof correctness. **Primary** path. |
| `P64` | `2^64 − 2^32 + 1` (Goldilocks) | 64 | A genuinely **large** prime field used in real zk-proof systems (Plonky2). Barrett needs 128-bit intermediates. **Bonus** path for the "large prime" requirement + RSA-style modexp. |

---

## 3. Why Barrett reduction (background)

Modular reduction `x mod p` normally needs a hardware divide, which is slow and (for crypto) can be
data-dependent. **Barrett reduction** replaces the divide with **multiplications and shifts** using a
precomputed constant `mu = floor(2^k / p)`:

```
q = (x * mu) >> k          # estimate of floor(x / p), off by at most 1
r = x - q * p              # in [0, 2p)
if (r >= p) r -= p         # final correction -> [0, p)
```

This is branch-light and constant-time-friendly, and the `mu` constant is shared across all
threads — perfect for SIMT GPUs. The competing technique, Montgomery multiplication, is also
mentioned in the analysis section for comparison.

---

## 4. Repository layout

```
Project/
├── README.md                 # this living document (summary / context / status / next steps)
├── Makefile                  # local (host C++) + Colab (nvcc) build targets
├── cuda/
│   ├── include/
│   │   ├── params.h           # prime parameters (P32, P64) + Barrett constants
│   │   └── modarith.cuh       # CORE UNIT: barrett_reduce, modmul, modexp, modinv (host+device)
│   └── src/
│       ├── host_test.cpp      # local correctness test (compiles on Mac, no GPU)  [Phase 1]
│       ├── ntt.cu             # NTT forward/inverse kernels + convolution         [Phase 3]
│       ├── modexp_demo.cu     # batch modular exponentiation (RSA-style) demo     [Phase 4]
│       └── bench.cu           # GPU vs CPU benchmark harness (CUDA events)        [Phase 3]
├── python/
│   ├── oracle.py              # ground-truth test vectors via Python big-ints/sympy
│   └── plots.py               # speedup / throughput plots from results CSV
├── colab/
│   └── run_on_colab.ipynb     # end-to-end: clone, nvcc compile, run, plot
└── results/                   # CSV + PNG outputs (generated)
```

---

## 5. Plan / phases

- [x] **Phase 0 — Setup:** skeleton, README, toolchain check, primes chosen.
- [x] **Phase 1 — Core arithmetic (local, no GPU):** `modarith.cuh` (Barrett/modmul/modexp/modinv for
  `P32`), `host_test.cpp`, Python `oracle.py`. **Validated: 2M random + boundary ops vs ground-truth
  128-bit math, 0 failures.**
- [x] **Phase 2 — CPU NTT reference:** forward/inverse NTT + cyclic convolution in host C++,
  **validated** against naive `O(n^2)` convolution and the Python oracle (round-trip + convolution OK).
- [x] **Phase 3 — GPU kernels + benchmarks:** pure-CUDA `ntt.cu` (per-stage butterfly kernels +
  bit-reverse + scale), `bench.cu` (CUDA-event timing, CSV). **T4 run complete: peak 52× NTT speedup.**
- [x] **Phase 4 — Bonus / large prime:** `P64` Goldilocks Barrett (128-bit intermediates),
  RSA-style batch `modexp_demo.cu`, Fermat modular inverse. **GPU: 277× speedup, 0 mismatches.**
- [x] **Phase 5 — Analysis & writeup:** §9 filled with real T4 numbers; performance/security
  trade-off analysis complete. **PROJECT COMPLETE.**
- [x] **Deliverables:** `report/report.tex` (5-page IEEE format), `slides/slides.tex` (Beamer, 4 slides, 2.5-min), git history committed.

---

## 6. How to build & run

### Locally (Mac — correctness only, no GPU)
```bash
make host_test      # compiles modarith.cuh as plain C++ and runs correctness tests
python3 python/oracle.py    # regenerate ground-truth test vectors
```

### On Google Colab (GPU execution)
1. **Package the project** (from the folder *above* `Project/`):
   ```bash
   cd /Users/akhiltejachirra/Documents/UCSD/Quarter-3/ECE_268
   zip -r project.zip Project -x 'Project/build/*'
   ```
2. Open `colab/run_on_colab.ipynb` in Colab, set Runtime → Change runtime type → **T4 GPU**.
3. Run the cells: upload `project.zip` (or set a GitHub `REPO_URL`), then it compiles the `.cu`
   files with `nvcc -arch=sm_75`, runs correctness + benchmarks, and renders the plots.

What you get on Colab:
- `./build/ntt` — GPU NTT matches the CPU reference, round-trip OK.
- `./build/bench` — modmul throughput + NTT latency → `results/bench_*.csv`.
- `./build/modexp_demo` — ~1M batched modexp over the large prime → `results/bench_modexp.csv`.
- `results/ntt_speedup.png`, `results/modmul_throughput.png`.

---

## 7. Status log (most recent first)

- **2026-05-30 (final):** Ran benchmarks on Colab T4. All GPU correctness tests passed (NTT 2^4–2^20
  matches CPU, 0 modexp mismatches). Key results: Barrett modmul **569× faster** on GPU (131 Gops/s
  vs 0.23 Gops/s), NTT kernel speedup peaks at **52×** (n=2^18), batched Goldilocks modexp **277×**.
  Fixed `size_t` include in `ntt_cpu.h` (g++ stricter than clang++). Fixed CUDA_CHECK macro variable
  shadowing (`e` renamed to `_cuda_err_`). Fixed Makefile compiler auto-detection (`clang++`/`g++`).
  README §9 finalized with real numbers and trade-off analysis. **Project complete.**
- **2026-05-30 (earlier):** Implemented the full unit and pipeline. Local results:
  - `host_test`: P32 + P64 Barrett/modmul/modexp/modinv — **2M+ checks, 0 failures**.
  - `ntt_host_test`: NTT round-trip (to 2^14) + convolution vs naive (to 2^9) — **OK**.
  - `oracle.py`: independent Python verification of primes, roots, Barrett, NTT — **OK**.
  - Wrote pure-CUDA `ntt.cu`, `bench.cu`, `modexp_demo.cu`, GPU NTT header, Colab notebook,
    plotting script. Primary Barrett path uses `__umul64hi` on device (no 128-bit dependency).
  - **Next:** run the notebook on a Colab T4 to compile the `.cu` files and capture GPU numbers.
- **2026-05-30:** Project kicked off. Confirmed Mac has no CUDA → execution on Colab T4. Chose NTT
  (application) + Barrett (reduction). Selected primes `P32` (primary) and `P64`/Goldilocks (bonus).
  Verified local toolchain. Created repo skeleton + README.

---

## 8. Next steps

Project is **complete** as of 2026-05-30. Optional improvements if time permits:
- Negacyclic NTT (Kyber/Dilithium-style) for a tighter PQC tie-in.
- Shared-memory NTT kernel for small sizes to close the crossover gap below n=2^11.
- OpenMP multi-thread CPU baseline for a fairer GPU-vs-CPU comparison.
- Branch-free / constant-time Barrett correction and Montgomery ladder modexp for the security angle.

---

## 9. Analysis & results

All benchmarks run on a **Colab NVIDIA Tesla T4** (sm_75, 16 GB GDDR6, 8.1 TFLOPS FP32) vs
a single CPU thread (Colab's Xeon, ~2 GHz effective) using the **same** Barrett C++ code.
GPU timing uses `cudaEventRecord` / `cudaEventElapsedTime` (excludes Python overhead).

### 9.1 Measured performance

#### Barrett modmul throughput (n = 4M independent pointwise multiplications, 64 iters/lane)

| | CPU (1 thread) | GPU (T4) | Speedup |
|---|---|---|---|
| Throughput | 0.23 Gops/s | **131.4 Gops/s** | **569×** |
| Wall time | 1161 ms | 2.04 ms | |

Raw Barrett modmul is **embarrassingly parallel** (one thread per element, no inter-thread
communication). The T4 has 2560 CUDA cores; with 4M elements and 64 iters each, all cores stay
saturated, achieving near-peak arithmetic throughput.

#### Forward NTT latency vs array size

| n | GPU kernel | GPU total (incl. transfer) | CPU | Kernel speedup | Total speedup |
|---|---|---|---|---|---|
| 2^10 (1K) | 0.051 ms | 0.100 ms | 0.046 ms | **0.9× (slower)** | 0.5× |
| 2^11 (2K) | 0.055 ms | 0.088 ms | 0.091 ms | 1.6× | 1.0× |
| 2^12 (4K) | 0.061 ms | 0.096 ms | 0.193 ms | 3.2× | 2.0× |
| 2^14 (16K) | 0.079 ms | 0.163 ms | 0.866 ms | 11× | 5.3× |
| 2^16 (64K) | 0.141 ms | 0.423 ms | 3.82 ms | 27× | 9.0× |
| 2^18 (256K) | 0.367 ms | 1.312 ms | 19.0 ms | **52× (peak)** | 14.5× |
| 2^20 (1M) | 2.54 ms | 6.37 ms | 90.1 ms | 35× | 14× |
| 2^22 (4M) | 12.7 ms | 27.3 ms | 507 ms | 40× | 19× |

#### Batched modular exponentiation over the large prime P64 (Goldilocks 2^64−2^32+1)

| n | CPU (1 thread) | GPU (T4) | Speedup | Correctness |
|---|---|---|---|---|
| 2^20 (1M) exponentiations | 955.8 ms | 3.4 ms | **277×** | 0 mismatches |

Plots: `results/ntt_speedup.png`, `results/modmul_throughput.png`.

### 9.2 Key observations from the data

**The breakeven point.** At n=2^10, the GPU kernel is actually *slower* (0.9×) than a single CPU
core. The GPU only overtakes the CPU at around n=2^11–2^12. For small inputs, `log₂n` CUDA kernel
launches (radix-2 iterative NTT) plus the PCIe transfer dominate; there simply aren't enough threads
to amortize that overhead. The lesson: GPU advantage is a function of *occupancy*, not just hardware.

**Why modmul speedup (569×) >> NTT speedup (52×).** The modmul benchmark is perfectly embarrassingly
parallel — no dependencies between threads, no strided memory access. The NTT butterfly pattern
introduces *inter-stage dependencies* and *strided, non-coalesced* global memory reads that lower
effective memory bandwidth. Each of the log₂n stages requires a separate kernel launch and a full
pass over device memory. At large n the NTT becomes memory-bandwidth-bound, not compute-bound.

**Peak speedup at n=2^18.** The 52× peak at 256K elements (kernel only) reflects the sweet spot
where: (1) enough thread blocks to saturate all 20 T4 SMs, and (2) the working set still fits in
L2 cache. Beyond 2^18 the array overflows L2 and bandwidth becomes the bottleneck, so the speedup
dips to ~35–40×.

**Goldilocks modexp (277×) is compute-bound.** Each thread independently runs 64 iterations of
square-and-multiply. No inter-thread communication, uniform control flow (exponent bits are random
so warp divergence averages out). This is the archetypal GPU cryptographic workload — batch
independent operations, one per thread.

### 9.3 Why Barrett is a good fit for the GPU

- **Divide-free.** GPUs have weak/expensive integer division; Barrett turns `x mod p` into a
  multiply-high plus a multiply-subtract and one conditional subtraction. On device we use the
  `__umul64hi` intrinsic (a single instruction on sm_75), so the hot path is two 64-bit multiplies
  — cheap and warp-uniform (no divergence from the reduction itself).
- **Shared constant.** The precomputed `mu` is the same for every thread — it lives in constant
  cache or a register broadcast, adding zero per-thread state.
- **Prime-agnostic.** Unlike Montgomery, Barrett needs no in/out domain conversion. For the NTT,
  data stays in normal residue form across all butterfly stages, avoiding extra passes.

### 9.4 Barrett vs Montgomery (trade-offs)

- **Barrett:** works directly in the normal residue domain (no conversion), great when inputs are
  used once or interact with non-modular code; one precomputed constant. **Chosen here.**
- **Montgomery:** usually 1 fewer multiply per reduction in tight loops (especially for modexp),
  but requires converting in/out of the Montgomery domain — amortized only when you do *many*
  multiplies on the same values. For NTT, Barrett keeps the data in natural form across all stages,
  avoiding two extra passes (domain conversion) which would cost additional memory bandwidth.
- The Goldilocks prime `2^64-2^32+1` also admits a still-faster *ad hoc* reduction exploiting its
  special structure; we deliberately use **generic Barrett** to satisfy the assignment requirement and
  keep the unit prime-agnostic (drop in any p and recompute `mu`).

### 9.5 Performance ↔ security trade-offs

- **Constant-time concerns.** Our `modexp` uses textbook square-and-multiply, which **branches on
  exponent bits** (`if (e & 1)`) — a timing side channel if the exponent is secret (RSA private key,
  ECDH scalar). The single `if (r >= p)` correction in Barrett is also data-dependent. A *secure*
  deployment would use a Montgomery ladder / always-square-and-multiply pattern, and a branch-free
  conditional subtraction (`r -= p & -(u64)(r >= p)`). This is the classic **throughput vs leakage**
  tension: our version is optimized for throughput benchmarking; it is not constant-time.
- **Larger field = more security, less throughput.** Moving from the 30-bit `P32` to the 64-bit
  `P64` (Goldilocks) doubles the operand width and forces 128-bit intermediates on device. The
  modexp demo (P64, 277× speedup) is slower in absolute terms per-op than the 30-bit modmul
  (569× speedup). Cryptographic moduli like RSA-2048 or 256-bit ECC fields push this much further,
  requiring multi-limb arithmetic and trading raw throughput for security margin.
- **Parallelism model matters for security too.** Batching many *independent* operations (our modexp
  demo — 1M independent exponentiations) is ideal for the GPU. But a *single* large modular
  exponentiation (e.g., one RSA-2048 decryption) is inherently sequential and would not benefit
  similarly; GPUs shine at independent-operation batches, not deep sequential chains.
- **NTT crossover point.** For small `n` (< 2^11) the GPU is actually *slower* than the CPU — a
  practical reminder that GPU parallelism has overheads (kernel launch, PCIe transfer) that only
  amortize at scale. Deploying a GPU NTT for tiny polynomial rings (e.g., Kyber with n=256) would
  be counterproductive without careful batching of many independent NTTs per launch.

### 9.6 Correctness methodology

Every GPU result is checked against a CPU reference that runs the **same** `__host__ __device__`
Barrett code, and the CPU/NTT logic is independently cross-checked by `python/oracle.py` (Python
big-integers + sympy). This three-way agreement (GPU ≡ CPU-C++ ≡ Python) is the correctness argument.

Specifically validated:
- **Barrett32 / Barrett64:** 2M random + boundary operands each, 0 failures vs ground-truth `u128 % p`.
- **modexp / invmod (both primes):** 100K random exponents checked vs an independent square-and-multiply
  using native `u128 % p` arithmetic.
- **NTT (CPU):** round-trip `INTT(NTT(x)) == x` for n up to 2^14; convolution matches naive O(n²) for n up to 2^9.
- **NTT (GPU):** GPU forward NTT matches CPU output elementwise for all sizes 2^4 through 2^20.
- **Batched modexp (P64, GPU):** 1,048,576 independent exponentiations, **0 mismatches** vs CPU.
