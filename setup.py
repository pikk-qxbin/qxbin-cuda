from setuptools import setup, Extension
import pybind11
import os

cuda_include = os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/include'
cuda_lib = os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/lib64'

ext_modules = [
    Extension(
        "qxbin_cuda",
        sources=["qxbin_cuda_pybind.cpp"],
        include_dirs=[pybind11.get_include(), cuda_include],
        library_dirs=[cuda_lib],
        libraries=["cudart"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    )
]

setup(
    name="qxbin_cuda",
    version="1.1.0",
    author="Rupesh Malpani / pikk.company",
    description="QxBin CUDA v1.1 - Native Python Bindings for high-performance Binary Probability Matrix evolution on GPU",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.10", "numpy"],
    python_requires=">=3.8",
    zip_safe=False,
)