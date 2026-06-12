# 10-Minute Presentation Recording Script (Group 12)

**Deck:** `ECE268_GPU_ModArith_Presentation_2.pptx`  
**Target:** ~10:00 total (9:30–10:30 is fine)  
**Split:** ~3 min Akhil · ~3 min Sameera · ~4 min Abhijit (adjust to slide count)

Record via Zoom (gallery + screen share) or OBS. Each speaker should introduce themselves once.

---

## Akhil (~3:00) — Problem, design, implementation

**Slide: Title**
> “Hi, we’re Group 12 — Akhil, Sameera, and Abhijit. Our project is a GPU modular arithmetic unit using Barrett reduction, applied to the Number Theoretic Transform for post-quantum crypto workloads.”

**Slide: Motivation / NTT**
> “Lattice schemes like Kyber and Dilithium rely on polynomial multiplication in a prime field. NTT turns that convolution into O(n log n) butterfly stages, and every butterfly is dominated by modular multiply-modulo a prime. That’s why a fast modmul engine on the GPU matters.”

**Slide: Barrett reduction**
> “We chose Barrett reduction over Montgomery because it fits our NTT data layout: we reduce products in place without Montgomery’s extra transform step. For P32 we use 30-bit primes with 64-bit products; for P64 we use the Goldilocks field with 128-bit intermediates.”

**Slide: Architecture**
> “All core math lives in `modarith.cuh` as `__host__ __device__` code — the same functions compile for CPU tests on my Mac and for CUDA kernels on Colab’s T4. NTT uses three kernels: bit-reverse, staged butterflies with precomputed roots, and scaling. We validate against a Python oracle and exhaustive local C++ tests before touching the GPU.”

**Handoff:**
> “Sameera ran our Colab benchmarks — I’ll pass it to her for the numbers.”

---

## Sameera (~3:00) — Benchmarks and results

**Slide: Experimental setup**
> “We benchmarked on a Colab Tesla T4 versus a single CPU thread, same Barrett code path. GPU times use CUDA events on the kernel; we also report end-to-end time including host-device transfer.”

**Slide: Modmul throughput**
> “For four million independent Barrett multiplications, the GPU hit about 131 giga-ops per second versus 0.23 on one CPU thread — roughly a 569× speedup. This is embarrassingly parallel: one thread per element, no synchronization.”

**Slide: NTT scaling**
> “NTT is more interesting. Below about n equals 2048, PCIe transfer and kernel launch dominate, so CPU wins on total time. Past that, GPU kernel time wins — we saw about 52× at n equals 262144. The lesson: modular arithmetic alone is a huge win; whole transforms need large n or batching to amortize transfer.”

**Slide: Modexp (if present)**
> “We also batched modular exponentiation on the 64-bit Goldilocks prime — about 277× over CPU with zero mismatches against the oracle.”

**Handoff:**
> “Abhijit will cover security trade-offs and wrap up.”

---

## Abhijit (~4:00) — Security, limitations, conclusion

**Slide: Security / side channels**
> “This is a performance prototype, not a hardened crypto library. Our Barrett correction uses conditional branches and variable-time arithmetic — fine for benchmarking, but an attacker with timing access could learn secrets. Production code would need constant-time reduction, careful memory access patterns, and likely Montgomery or specialized intrinsics.”

**Slide: Limitations**
> “Single-thread CPU baseline isn’t a fair upper bound — OpenMP would narrow the gap. We didn’t implement negacyclic NTT or shared-memory tiling, which would help small n. Results are tied to one T4; newer GPUs would shift the crossover.”

**Slide: Contributions / repo**
> “Code, tests, Colab notebook, and CSV results are on GitHub — link on the slide. See `CONTRIBUTIONS.md` for who did what.”

**Slide: Conclusion**
> “We built a from-scratch CUDA Barrett engine, proved it with CPU and Python oracles, and showed when GPU modular arithmetic pays off for NTT-scale workloads. Thanks — happy to take questions if this were live; for the recording, that’s our project.”

---

## Recording tips

- **Pace:** ~130 words/min → ~400 words per speaker for 3 min.
- **Audio:** One mic per person or a quiet room; test levels before the full take.
- **Video:** Course may require faces on camera — confirm syllabus.
- **Retries:** Do one dry run with a phone timer; re-record only the speaker who stumbles.
- **Export:** MP4, 1080p, upload to Gradescope under the presentation assignment.

## Short backup (if deck has fewer slides)

Combine Akhil’s design + Sameera’s top number (569× modmul, 52× NTT peak) + Abhijit’s security caveat in one pass each; skip modexp slide if over time.
