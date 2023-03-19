#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"


__global__ void FW_simple_kernel(int* d_D, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (i < n && j < n) {
        int ij = d_D[i * n + j];
        int ik = d_D[i * n + k];
        int kj = d_D[k * n + j];

        if (ik + kj < ij)
            d_D[i * n + j] = ik + kj;
    }
}