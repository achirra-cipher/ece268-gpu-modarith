#!/usr/bin/env python3
"""Plot GPU-vs-CPU benchmark results produced by the CUDA harness.

Reads results/bench_ntt.csv and results/bench_modmul.csv and writes PNGs into
results/. Run after the benchmarks (on Colab or any machine with the CSVs):

    python3 python/plots.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"  (skip) {name} not found")
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_ntt():
    rows = read_csv("bench_ntt.csv")
    if not rows:
        return
    log2n = [int(r["log2n"]) for r in rows]
    gpu_k = [float(r["gpu_kernel_ms"]) for r in rows]
    gpu_t = [float(r["gpu_total_ms"]) for r in rows]
    cpu = [float(r["cpu_ms"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(log2n, cpu, "o-", label="CPU (1 thread)")
    ax1.plot(log2n, gpu_t, "s-", label="GPU total (incl. transfer)")
    ax1.plot(log2n, gpu_k, "^-", label="GPU kernel only")
    ax1.set_yscale("log")
    ax1.set_xlabel("NTT size (log2 n)")
    ax1.set_ylabel("time (ms, log scale)")
    ax1.set_title("Forward NTT latency: GPU vs CPU")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    sp_k = [c / g for c, g in zip(cpu, gpu_k)]
    sp_t = [c / g for c, g in zip(cpu, gpu_t)]
    ax2.plot(log2n, sp_k, "^-", label="vs GPU kernel")
    ax2.plot(log2n, sp_t, "s-", label="vs GPU total")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("NTT size (log2 n)")
    ax2.set_ylabel("speedup (x)")
    ax2.set_title("GPU speedup over CPU")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RESULTS, "ntt_speedup.png")
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


def plot_modmul():
    rows = read_csv("bench_modmul.csv")
    if not rows:
        return
    r = rows[0]
    labels = ["CPU (1 thread)", "GPU (T4)"]
    gops = [float(r["cpu_gops"]), float(r["gpu_gops"])]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, gops, color=["#888", "#2a7"])
    ax.set_ylabel("modular mults / sec (Gops/s)")
    ax.set_title(f"Barrett modmul throughput\n(speedup {float(r['speedup']):.0f}x)")
    for b, v in zip(bars, gops):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom")
    fig.tight_layout()
    out = os.path.join(RESULTS, "modmul_throughput.png")
    fig.savefig(out, dpi=130)
    print(f"  wrote {out}")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    plot_ntt()
    plot_modmul()
