# Benchmark results (Google Colab, Tesla T4)

Benchmarks were run with `make bench` and `make modexp_demo` after compiling with
`nvcc -arch=sm_75`. GPU times use CUDA events; CPU times use a single-thread
reference running the same Barrett code path as the GPU.

## Files

| File | Description |
|------|-------------|
| `bench_modmul.csv` | Pointwise Barrett modmul throughput (n=2^22 lanes, 64 iters each) |
| `bench_ntt.csv` | Forward NTT latency vs transform size (log2n 10–22) |
| `bench_modexp.csv` | Batched modexp over P64 Goldilocks (2^20 independent ops) |

## How to regenerate

1. Open `colab/run_on_colab.ipynb` with a T4 GPU runtime.
2. Run through compile and `./build/bench` / `./build/modexp_demo`.
3. CSV files are written under `results/`.

Plots: `python3 python/plots.py` (reads these CSVs; PNGs are gitignored).
