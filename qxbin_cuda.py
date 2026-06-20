import cupy as cp
import matplotlib.pyplot as plt


class QxBinCUDA:
    """
    QxBin GPU / CUDA Simulator - High-Performance Cubit Ensemble on NVIDIA GPUs

    CuPy-powered drop-in evolution of QxBinCloud logic.
    Runs the Binary Probability Matrix evolution entirely on GPU for massive speedups
    on ensembles (hundreds to tens of thousands of cubits).

    Requires: pip install cupy-cuda12x   (or cupy-cuda11x matching your CUDA version)
    or conda install cupy

    This implements the GPU / CUDA port from the QxBin roadmap.
    Same fractional exponent superposition logic, now massively parallel on silicon.

    Perfect for:
    - Large-scale Monte Carlo / probabilistic simulation
    - Quantum-inspired optimization loops
    - Real-time ensemble evolution & feedback systems
    - Pikk edge + AI workloads needing fast uncertainty modeling
    """

    def __init__(self, num_cubits: int = 256, grid_size: int = 8):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        # Allocate directly on GPU
        self.states = cp.random.rand(
            num_cubits, grid_size, grid_size, dtype=cp.float64
        )
        # Normalize all matrices in one go
        s = self.states.sum(axis=(1, 2), keepdims=True)
        self.states /= cp.where(s > 0, s, 1.0)

    def evolve_chains(self, biases=None):
        """
        Evolve ALL cubit chains in parallel on the GPU using QxBin fractional logic.
        Vectorized blend + per-matrix normalization.
        """
        if biases is None:
            biases = cp.random.uniform(0.5, 0.85, self.num_cubits, dtype=cp.float64)
        else:
            biases = cp.asarray(biases, dtype=cp.float64)

        ns = cp.random.randint(1, 6, self.num_cubits, dtype=cp.int32)
        ms = cp.random.randint(1, 6, self.num_cubits, dtype=cp.int32)

        # Broadcasting magic: [num_cubits, 1, 1] for elementwise across grids
        frac = (biases ** ns.astype(cp.float64))[:, None, None]
        tail = ((1.0 - biases) ** ms.astype(cp.float64))[:, None, None]

        # Core QxBin blend (identical math to CPU version)
        blended = (self.states * frac + (1.0 - self.states) * tail) * 0.5

        # Per-matrix normalize on GPU
        total = blended.sum(axis=(1, 2), keepdims=True)
        self.states = cp.where(
            total > 1e-12,
            blended / total,
            cp.ones_like(blended) / (self.grid_size * self.grid_size)
        )
        # Return aggregate mean as numpy for easy viz / downstream CPU use
        return cp.asnumpy(self.states.mean(axis=0))

    def optimize_to_target(self, target_mean: float = 0.7, max_steps: int = 200):
        """GPU-accelerated probabilistic optimization loop."""
        for step in range(max_steps):
            agg = self.evolve_chains()
            current = float(agg.mean())
            if abs(current - target_mean) < 0.001:
                print(f"Converged after {step} steps | mean prob ≈ {current:.4f}")
                break
        return cp.asnumpy(self.states)

    def visualize(self, title="QxBin CUDA - GPU Probability Landscape"):
        agg = cp.asnumpy(self.states.mean(axis=0))
        plt.figure(figsize=(8, 6))
        plt.imshow(agg, cmap="inferno", interpolation="nearest")
        plt.colorbar(label="Average Probability Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        plt.show()

    def to_cpu(self):
        """Return current states as numpy array (for compatibility with other tiers)."""
        return cp.asnumpy(self.states)


if __name__ == "__main__":
    print("QxBin CUDA v1.1 - GPU Accelerated Cubit Ensemble")
    print("This is the CuPy layer of the dedicated qxbin-cuda repo.\n")

    cuda = QxBinCUDA(num_cubits=1024, grid_size=8)
    print(f"Initialized {cuda.num_cubits} cubit matrices on GPU...")

    print("Running optimization loop on GPU...")
    final_states = cuda.optimize_to_target(target_mean=0.72, max_steps=150)

    print("\nFinal aggregate probability landscape (mean across all cubits):")
    print(cp.asnumpy(final_states.mean()))

    cuda.visualize("QxBin CUDA v1.1 - 1024 Cubits Evolved on GPU")

    print("\n\u2705 CuPy layer ready. For maximum performance use the native bindings.")