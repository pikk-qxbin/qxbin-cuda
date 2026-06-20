/*
 * QxBin CUDA Native Kernel v1.1
 * High-performance Binary Probability Matrix evolution on NVIDIA GPUs
 *
 * Compilation:
 *   nvcc -o qxbin_cuda qxbin_cuda.cu -arch=sm_80   # or your GPU arch
 *   ./qxbin_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define GRID_SIZE 8
#define NUM_CUBITS 1024
#define MAX_STEPS 150
#define TARGET_MEAN 0.72f

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void evolve_kernel(
    float* __restrict__ states,
    const float* __restrict__ biases,
    const int* __restrict__ ns,
    const int* __restrict__ ms,
    int num_cubits,
    int grid_size
) {
    int cubit_id = blockIdx.x;
    if (cubit_id >= num_cubits) return;

    int matrix_size = grid_size * grid_size;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    float bias = biases[cubit_id];
    int n = ns[cubit_id];
    int m = ms[cubit_id];

    float frac = powf(bias, (float)n);
    float tail = powf(1.0f - bias, (float)m);

    float* my_matrix = states + cubit_id * matrix_size;

    float local_sum = 0.0f;
    for (int i = tid; i < matrix_size; i += total_threads) {
        float old_val = my_matrix[i];
        float new_val = (old_val * frac + (1.0f - old_val) * tail) * 0.5f;
        my_matrix[i] = new_val;
        local_sum += new_val;
    }

    float block_sum = blockReduceSum(local_sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = block_sum;
    __syncthreads();

    float norm = (shared_sum > 1e-12f) ? (1.0f / shared_sum) : (1.0f / matrix_size);

    for (int i = tid; i < matrix_size; i += total_threads) {
        my_matrix[i] *= norm;
    }
}

void evolve_step(float* d_states, float* d_biases, int* d_ns, int* d_ms, int num_cubits, int grid_size) {
    int threads_per_block = 256;
    int blocks = num_cubits;
    evolve_kernel<<<blocks, threads_per_block>>>(d_states, d_biases, d_ns, d_ms, num_cubits, grid_size);
    cudaDeviceSynchronize();
}

int main() {
    printf("QxBin CUDA Native Kernel v1.1\n");
    printf("Running on GPU with %d cubits, %dx%d grids\n\n", NUM_CUBITS, GRID_SIZE, GRID_SIZE);

    int matrix_size = GRID_SIZE * GRID_SIZE;
    size_t states_bytes = NUM_CUBITS * matrix_size * sizeof(float);
    size_t vec_bytes = NUM_CUBITS * sizeof(float);
    size_t int_vec_bytes = NUM_CUBITS * sizeof(int);

    float* h_states = (float*)malloc(states_bytes);
    float* h_biases = (float*)malloc(vec_bytes);
    int* h_ns = (int*)malloc(int_vec_bytes);
    int* h_ms = (int*)malloc(int_vec_bytes);

    srand(42);
    for (int c = 0; c < NUM_CUBITS; c++) {
        float sum = 0.0f;
        for (int i = 0; i < matrix_size; i++) {
            h_states[c * matrix_size + i] = (float)rand() / RAND_MAX;
            sum += h_states[c * matrix_size + i];
        }
        for (int i = 0; i < matrix_size; i++) h_states[c * matrix_size + i] /= sum;
        h_biases[c] = 0.5f + ((float)rand() / RAND_MAX) * 0.35f;
        h_ns[c] = 1 + rand() % 5;
        h_ms[c] = 1 + rand() % 5;
    }

    float *d_states, *d_biases;
    int *d_ns, *d_ms;
    cudaMalloc(&d_states, states_bytes);
    cudaMalloc(&d_biases, vec_bytes);
    cudaMalloc(&d_ns, int_vec_bytes);
    cudaMalloc(&d_ms, int_vec_bytes);

    cudaMemcpy(d_states, h_states, states_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, h_biases, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ns, h_ns, int_vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms, h_ms, int_vec_bytes, cudaMemcpyHostToDevice);

    printf("Evolving on GPU...\n");
    for (int step = 0; step < MAX_STEPS; step++) {
        evolve_step(d_states, d_biases, d_ns, d_ms, NUM_CUBITS, GRID_SIZE);

        if (step % 20 == 0 || step == MAX_STEPS - 1) {
            cudaMemcpy(h_states, d_states, states_bytes, cudaMemcpyDeviceToHost);
            float total_mean = 0.0f;
            for (int c = 0; c < NUM_CUBITS; c++) {
                float cubit_mean = 0.0f;
                for (int i = 0; i < matrix_size; i++) {
                    cubit_mean += h_states[c * matrix_size + i];
                }
                total_mean += cubit_mean / matrix_size;
            }
            total_mean /= NUM_CUBITS;
            printf("Step %3d | Mean probability: %.4f\n", step, total_mean);

            if (fabsf(total_mean - TARGET_MEAN) < 0.001f) {
                printf("Converged early at step %d\n", step);
                break;
            }
        }
    }

    cudaMemcpy(h_states, d_states, states_bytes, cudaMemcpyDeviceToHost);
    float final_agg = 0.0f;
    for (int c = 0; c < NUM_CUBITS; c++) {
        for (int i = 0; i < matrix_size; i++) final_agg += h_states[c * matrix_size + i];
    }
    final_agg /= (NUM_CUBITS * matrix_size);
    printf("\nFinal aggregate mean probability: %.4f\n", final_agg);

    printf("\n\u2705 Native CUDA v1.1 complete.\n");

    cudaFree(d_states); cudaFree(d_biases); cudaFree(d_ns); cudaFree(d_ms);
    free(h_states); free(h_biases); free(h_ns); free(h_ms);
    return 0;
}