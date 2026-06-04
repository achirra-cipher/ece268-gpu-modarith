# Team Contributions

## Project: GPU-Accelerated Modular Arithmetic Unit
**Course:** ECE 268 — GPU Architecture and Programming  
**Team:** achirra-cypher / ece268-gpu-modarith

---

## Abhijit Kumar (akg008@ucsd.edu)

- Added docstrings to `python/oracle.py`: documented `is_prime`, `barrett_mu`, `barrett_reduce`, `check_barrett`, `ntt`, and `naive_conv` to clarify algorithm choices and parameter contracts.
- Added docstrings to `python/plots.py`: documented `read_csv`, `plot_ntt`, and `plot_modmul` to describe input sources and output artifacts.
- Created this `CONTRIBUTIONS.md` file summarizing team roles.

## Achira (GitHub: achirra-cypher)

- Designed and implemented the CUDA kernel for Barrett modular multiplication (`cuda/`).
- Set up the benchmark harness that produces `results/bench_ntt.csv` and `results/bench_modmul.csv`.
- Wrote the core NTT GPU kernel and CPU reference implementation.
- Authored the Makefile, README, and project report (`report/`).
- Implemented the Python oracle (`python/oracle.py`) for ground-truth verification of the CUDA results.
- Implemented the plotting script (`python/plots.py`) for benchmark visualization.
