# QxBin CUDA v1.1.0 Release Notes

**Released:** June 21, 2026

QxBin CUDA v1.1 is the first official release of the dedicated high-performance GPU acceleration layer for the QxBin framework.

## Highlights

- **Full layered GPU stack** now available in one clean repository
- **Automatic backend selection** — just `from qxbin import QxBin` and it picks the fastest available backend
- Production-ready with CI, changelog, and proper versioning
- Same powerful fractional superposition math, now running at full GPU speed

## What's Included

### Three Execution Layers
| Layer              | Performance | Ease of Use | Best For                     |
|--------------------|-------------|-------------|------------------------------|
| Native CUDA        | Best        | Medium      | Maximum speed                |
| CuPy (Python)      | Excellent   | High        | Prototyping & visualization  |
| Unified `QxBin`    | Best        | Highest     | **Recommended for most users** |

### Key Features
- Automatic backend selection (`native` → `cupy` → `cpu`)
- Native CUDA C++ kernel with efficient parallel reduction
- Python bindings via pybind11
- High-level CuPy implementation
- CPU fallback always available
- GitHub Actions CI for reliable builds

## Installation

### Recommended (Unified Interface)
```bash
pip install pybind11 numpy
python setup.py build_ext --inplace

from qxbin import QxBin
qx = QxBin(num_cubits=1024, grid_size=8)
```

### Quick Try (CuPy only)
```bash
pip install cupy-cuda12x
python qxbin_cuda.py
```

## Links
- Repository: https://github.com/pikk-qxbin/qxbin-cuda
- Main QxBin Framework: https://github.com/pikk-qxbin/qxbin

## What's Next
- More advanced native kernel features
- Multi-GPU support
- Tighter integration with Qiskit / CUDA-Q
- Benchmarks and performance comparisons

---

**This release represents a major step toward democratizing quantum-inspired computing on classical hardware.**

Built with ❤️ by Rupesh Malpani / pikk.company