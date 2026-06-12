# ECE 268 Final Project — Submission Checklist (Group 12)

**Repo:** https://github.com/achirra-cipher/ece268-gpu-modarith  
**Team:** Akhil Teja Chirra, Vagda Sameera Talari, Abhijit Kumar Gupta

---

## Gradescope uploads

| Item | File | Status |
|---|---|---|
| IEEE report (PDF) | `ECE268_GPU_ModArith_Report_2.pdf` | **You** — apply report feedback, re-export PDF, upload |
| 10-minute presentation recording | MP4/WebM from Zoom or screen recorder | **Pending** — use script in `slides/RECORDING_10MIN.md` |

## In-class presentation (June 4)

| Item | File | Status |
|---|---|---|
| 5-slide deck (course template) | `slides/12_Project_Akhil_Sameera_Abhijit.pptx` | Ready — upload to course Google Drive if required |

## GitHub (required deliverable)

| Item | Location | Status |
|---|---|---|
| Source code + tests | `cuda/`, `python/`, `Makefile` | Done |
| Colab reproduction | `colab/run_on_colab.ipynb` | Done |
| Benchmark CSVs | `results/*.csv` | Done |
| Team contributions | `CONTRIBUTIONS.md` | Done |
| Final report PDF | `ECE268_GPU_ModArith_Report_2.pdf` | Commit after your edits |
| Recording deck | `ECE268_GPU_ModArith_Presentation_2.pptx` | Done |

## Before you submit the report

Apply these fixes to `ECE268_GPU_ModArith_Report_2.pdf` (or its LaTeX source):

1. Remove duplicate Section VI.B / VI.C (both titled “Roofline Analysis and Occupancy”).
2. Fix citation [4] — AES reference should not point to Barrett’s paper.
3. Align code listings with the repo (`barrett32_reduce`, `k_ntt_stage`, precomputed `roots[]`).
4. Correct modexp iteration count wording (not “64 iterations = prime bit-length”).
5. Verify Table II PCIe row against `results/bench_ntt.csv` (kernel ~2.04 ms).
6. Remove or qualify “constant-time friendly” in the intro (contradicts security section).
7. CPU compiler on Colab is **g++**, not clang.
8. Table V: “3329-bit Kyber field” → “Kyber modulus 3329”; rephrase “277× attacker advantage”.

## Recording (10 minutes)

1. Open `ECE268_GPU_ModArith_Presentation_2.pptx`.
2. Follow `slides/RECORDING_10MIN.md` — ~3 min per speaker.
3. Record with all three on camera or voice (course policy); share screen on slides.
4. Export MP4; upload to Gradescope.

## Local verification (optional)

```bash
make local          # host + NTT tests on Mac (no CUDA)
python3 python/oracle.py
python3 python/plots.py   # requires matplotlib
```

GPU benchmarks: open `colab/run_on_colab.ipynb` on Colab with GPU runtime.
