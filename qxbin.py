"""
QxBin Unified Interface v1.1

Automatic backend selection for maximum performance + ease of use.

Tries in this order:
1. Native CUDA bindings (qxbin_cuda) - fastest
2. CuPy (qxbin_cuda.py) - excellent GPU performance, pure Python
3. CPU fallback (NumPy) - always works

Usage:
    from qxbin import QxBin

    qx = QxBin(num_cubits=1024, grid_size=8)   # auto-selects best backend
    qx.evolve_chains()
    qx.optimize_to_target(0.7)
"""

import importlib
import warnings

class QxBin:
    """
    Unified QxBin interface with automatic backend selection.
    
    This is the recommended way to use QxBin CUDA in most cases.
    """

    def __init__(self, num_cubits: int = 256, grid_size: int = 8, backend: str = "auto"):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        self.backend = None
        self._impl = None

        if backend == "auto":
            self._select_backend()
        elif backend == "native":
            self._load_native()
        elif backend == "cupy":
            self._load_cupy()
        elif backend == "cpu":
            self._load_cpu()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _select_backend(self):
        # Try native CUDA bindings first (best performance)
        try:
            self._load_native()
            return
        except Exception:
            pass

        # Then try CuPy
        try:
            self._load_cupy()
            return
        except Exception:
            pass

        # Fallback to CPU (NumPy)
        warnings.warn("No GPU backend available. Falling back to CPU (NumPy). "
                      "For best performance install cupy-cudaXX or build the native bindings.")
        self._load_cpu()

    def _load_native(self):
        from qxbin_cuda import QxBinNative
        self._impl = QxBinNative(self.num_cubits, self.grid_size)
        self.backend = "native"
        print(f"[QxBin] Using native CUDA backend ({self.num_cubits} cubits)")

    def _load_cupy(self):
        import importlib.util
        import os
        
        spec = importlib.util.spec_from_file_location("qxbin_cupy", 
            os.path.join(os.path.dirname(__file__), "qxbin_cuda.py"))
        qx_cupy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qx_cupy_module)
        
        self._impl = qx_cupy_module.QxBinCUDA(self.num_cubits, self.grid_size)
        self.backend = "cupy"
        print(f"[QxBin] Using CuPy backend ({self.num_cubits} cubits)")

    def _load_cpu(self):
        import numpy as np
        
        class _CPUQxBin:
            def __init__(self, num_cubits, grid_size):
                self.num_cubits = num_cubits
                self.grid_size = grid_size
                self.states = np.random.rand(num_cubits, grid_size, grid_size).astype(np.float64)
                self.states /= self.states.sum(axis=(1,2), keepdims=True)
            
            def evolve_chains(self):
                biases = np.random.uniform(0.5, 0.85, self.num_cubits)
                ns = np.random.randint(1, 6, self.num_cubits)
                ms = np.random.randint(1, 6, self.num_cubits)
                
                frac = (biases ** ns)[:, None, None]
                tail = ((1 - biases) ** ms)[:, None, None]
                
                blended = (self.states * frac + (1 - self.states) * tail) * 0.5
                total = blended.sum(axis=(1,2), keepdims=True)
                self.states = np.where(total > 1e-12, blended / total, 
                                       np.ones_like(blended) / (self.grid_size**2))
                return self.states.mean(axis=0)
            
            def optimize_to_target(self, target_mean=0.7, max_steps=200):
                for step in range(max_steps):
                    agg = self.evolve_chains()
                    if abs(agg.mean() - target_mean) < 0.001:
                        print(f"Converged after {step} steps")
                        break
                return self.states.mean(axis=0)
        
        self._impl = _CPUQxBin(self.num_cubits, self.grid_size)
        self.backend = "cpu"

    def evolve_chains(self):
        return self._impl.evolve_chains()

    def optimize_to_target(self, target_mean: float = 0.7, max_steps: int = 200):
        return self._impl.optimize_to_target(target_mean, max_steps)

    def visualize(self, title=None):
        if hasattr(self._impl, "visualize"):
            self._impl.visualize(title or f"QxBin ({self.backend})")
        else:
            print("Visualize not available on current backend")

    def __repr__(self):
        return f"<QxBin backend={self.backend} num_cubits={self.num_cubits} grid_size={self.grid_size}>"


def create_qxbin(num_cubits=256, grid_size=8, backend="auto"):
    """Factory function for QxBin with automatic backend selection."""
    return QxBin(num_cubits, grid_size, backend)