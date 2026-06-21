# Changelog

All notable changes to QxBin CUDA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-06-21

### Added
- Dedicated `qxbin-cuda` repository
- Full layered GPU acceleration stack:
  - High-level CuPy implementation (`qxbin_cuda.py`)
  - Native CUDA C++ kernel (`qxbin_cuda.cu`)
  - Python bindings via pybind11 (`qxbin_cuda_pybind.cpp` + `setup.py`)
- Comprehensive README with quick starts for all layers
- MIT License
- GitHub Actions CI workflow for building and testing bindings
- This CHANGELOG

### Changed
- Separated GPU acceleration into its own focused repository (v1.1)

[1.1.0]: https://github.com/pikk-qxbin/qxbin-cuda/releases/tag/v1.1.0