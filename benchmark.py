"""
QxBin CUDA v1.1 - GPU Benchmarks

Simple but effective benchmark comparing the three backends:
- Native CUDA (fastest)
- CuPy
- CPU (NumPy fallback)

Run with:
    python benchmark.py

Requirements:
    - For full comparison: build the native bindings + have CuPy installed
    - CPU fallback always works
"""

import time
import numpy as np
from qxbin import QxBin

def benchmark_backend(backend: str, num_cubits: int, grid_size: int, iterations: int = 5):
    """Benchmark a single backend."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {backend.upper()} | {num_cubits} cubits | {grid_size}x{grid_size} grid")
    print('='*60)

    try:
        qx = QxBin(num_cubits=num_cubits, grid_size=grid_size, backend=backend)
    except Exception as e:
        print(f"  Failed to initialize {backend}: {e}")
        return None

    # Warmup
    qx.evolve_chains()

    # Benchmark evolve_chains
    evolve_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        qx.evolve_chains()
        evolve_times.append(time.perf_counter() - start)

    avg_evolve = np.mean(evolve_times) * 1000  # ms
    std_evolve = np.std(evolve_times) * 1000

    print(f"  evolve_chains()     : {avg_evolve:.2f} ± {std_evolve:.2f} ms")

    # Benchmark optimize_to_target (lighter version)
    opt_times = []
    for _ in range(3):  # fewer iterations for optimize
        start = time.perf_counter()
        qx.optimize_to_target(target_mean=0.65, max_steps=20)
        opt_times.append(time.perf_counter() - start)

    avg_opt = np.mean(opt_times)
    print(f"  optimize_to_target(): {avg_opt:.2f} seconds (20 steps)")

    cubits_per_sec = num_cubits / (avg_evolve / 1000)
    print(f"  Throughput          : {cubits_per_sec:,.0f} cubits/second")

    return {
        "backend": backend,
        "num_cubits": num_cubits,
        "avg_evolve_ms": avg_evolve,
        "avg_optimize_s": avg_opt,
        "throughput": cubits_per_sec
    }


def run_full_benchmark():
    print("\n" + "="*70)
    print("QxBin CUDA v1.1 - Performance Benchmark")
    print("="*70)
    print("Comparing Native CUDA vs CuPy vs CPU backends\n")

    configs = [
        (256, 6),
        (1024, 8),
        (4096, 8),
    ]

    all_results = []

    for num_cubits, grid_size in configs:
        for backend in ["native", "cupy", "cpu"]:
            result = benchmark_backend(backend, num_cubits, grid_size)
            if result:
                all_results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY - Average evolve_chains() time (ms)")
    print("="*70)
    print(f"{'Backend':<12} {'256 cubits':>12} {'1024 cubits':>14} {'4096 cubits':>14}")
    print("-" * 54)

    for backend in ["native", "cupy", "cpu"]:
        row = [backend]
        for num_cubits, _ in configs:
            matches = [r for r in all_results if r["backend"] == backend and r["num_cubits"] == num_cubits]
            if matches:
                row.append(f"{matches[0]['avg_evolve_ms']:.1f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<12} {row[1]:>12} {row[2]:>14} {row[3]:>14}")

    print("\nNote: Lower is better. Native CUDA should be significantly faster than CuPy and CPU.")
    print("Run this on your target GPU for accurate numbers.\n")


if __name__ == "__main__":
    run_full_benchmark()