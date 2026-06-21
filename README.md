# QxBin CUDA v1.1

**High-Performance GPU Acceleration Layer for QxBin**

Binary Probability Matrix evolution on NVIDIA GPUs using the same fractional superposition math that makes room-temperature quantum-inspired simulation possible.

CuPy (Python) + Native CUDA C++ kernel + Python Bindings (pybind11)

---

## What is this?

This is the dedicated GPU acceleration repository for the [QxBin framework](https://github.com/pikk-qxbin/qxbin).

It implements the **GPU / CUDA port** (roadmap item #3) as a clean, layered, production-ready package:

- **Same core math** as the main QxBin repo (fractional exponents `bias**n` / `(1-bias)**m`, blend, per-matrix normalization)
- Three tiers of execution for different needs
- Designed for scale: 1024+ cubit ensembles at interactive speeds

## The Three Layers

| Layer                    | File(s)                        | Performance     | Pythonic | Recommended for                  |
|--------------------------|--------------------------------|-----------------|----------|----------------------------------|
| **High-level (CuPy)**    | `qxbin_cuda.py`                | Excellent       | Yes      | Prototyping, visualization, quick experiments |
| **Native CUDA Kernel**   | `qxbin_cuda.cu`                | Best            | No       | Maximum speed, custom integrations, production kernels |
| **Python Bindings**      | `qxbin_cuda_pybind.cpp` + `setup.py` | Excellent    | Yes      | **Best overall** – native speed from Python |

All layers produce identical mathematical results.

## Quick Start

### 1. High-level CuPy (easiest to try)

```bash
pip install cupy-cuda12x   # or cupy-cuda11x matching your CUDA version
python qxbin_cuda.py
```

### 2. Python Bindings + Native Kernel (recommended)

```bash
pip install pybind11 numpy

# Build the binding
python setup.py build_ext --inplace

# Use it
python -c "
from qxbin_cuda import QxBinNative
qx = QxBinNative(num_cubits=1024, grid_size=8)
print(qx)
qx.optimize_to_target(target_mean=0.72)
"
```

### 3. Native CUDA Kernel only

```bash
nvcc -o qxbin_cuda qxbin_cuda.cu -arch=sm_80   # change to your GPU arch
./qxbin_cuda
```

## CI Status

[![CI](https://github.com/pikk-qxbin/qxbin-cuda/actions/workflows/ci.yml/badge.svg)](https://github.com/pikk-qxbin/qxbin-cuda/actions/workflows/ci.yml)

## Installation Notes

- **CuPy**: `pip install cupy-cuda12x` (or matching your CUDA toolkit)
- **pybind11**: `pip install pybind11`
- **CUDA Toolkit**: Required for native kernel and bindings (nvcc)
- Works on Linux with NVIDIA GPUs (RTX 30/40 series, A100, H100, etc.)

## Why QxBin CUDA matters

We didn't wait for fault-tolerant quantum computers.
We changed the mathematical representation (Binary Probability Matrices + fractional states) so that **today's GPUs** can run rich, evolving, probabilistic "spinning coin" simulations at scale.

This is democratized quantum-inspired computing:
- Personal labs on every developer machine
- Edge + cloud scale
- Foundation for uncertainty-aware AI and optimization

## Version

**v1.1** – Initial public release of the full layered GPU stack (June 2026)

## Related

- Main QxBin framework: https://github.com/pikk-qxbin/qxbin
- Author: Rupesh Malpani (@rupeshmalpani) | pikk.company

## License

MIT License (same as main QxBin repo)

Fork it. Extend it. Ship faster.
