# Team Contributions

## Project: GPU-Accelerated Modular Arithmetic Unit
**Course:** ECE 268 — Security of Hardware Embedded Systems  
**Team:** Group 12 — [ece268-gpu-modarith](https://github.com/achirra-cipher/ece268-gpu-modarith)

---

## Akhil Teja Chirra (achirra@ucsd.edu)

- Designed and implemented Barrett modular arithmetic (`cuda/include/modarith.cuh`): `barrett32_reduce`, `barrett64_reduce`, `modmul`, `modexp`, and Fermat modular inverse.
- Implemented CPU and GPU Number Theoretic Transform (`cuda/include/ntt_cpu.h`, `cuda/include/ntt_gpu.cuh`, `cuda/src/ntt.cu`).
- Built the benchmark harness (`cuda/src/bench.cu`, `cuda/src/modexp_demo.cu`) and local host tests (`cuda/src/host_test.cpp`, `cuda/src/ntt_host_test.cpp`).
- Authored the Makefile, README, Colab notebook (`colab/run_on_colab.ipynb`), and IEEE report (`report/report.tex`, `ECE268_GPU_ModArith_Report_2.pdf`).
- Implemented the Python oracle (`python/oracle.py`) and plotting script (`python/plots.py`) for independent verification and benchmark visualization.

## Vagda Sameera Talari (vtalari@ucsd.edu)

- Ran GPU benchmarks on Google Colab (Tesla T4) and captured results in `results/bench_modmul.csv`, `results/bench_ntt.csv`, and `results/bench_modexp.csv`.
- Built the course-template presentation deck (`slides/12_Project_Akhil_Sameera_Abhijit.pptx`) and the 10-minute recording deck (`ECE268_GPU_ModArith_Presentation_2.pptx`).
- Contributed to slide content, in-class presentation (June 4), and the recorded presentation.

## Abhijit Kumar Gupta (akg008@ucsd.edu)

- Added docstrings to `python/oracle.py`: documented `is_prime`, `barrett_mu`, `barrett_reduce`, `check_barrett`, `ntt`, and `naive_conv` to clarify algorithm choices and parameter contracts.
- Added docstrings to `python/plots.py`: documented `read_csv`, `plot_ntt`, and `plot_modmul` to describe input sources and output artifacts.
- Created and maintained this `CONTRIBUTIONS.md` file summarizing team roles.
