#pragma once
#include <cuda.h>
#include <cuda_runtime.h>



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


//TODO: VALUTARE SE FARE UNA VERSIONE PER LMEM E NON (SENZA SYNCTHREAD)
__forceinline__
__device__ void updateBlocked(int* C, int* A, int* B, int bi, int bj, const int blockSize){
    for(int k = 0; k < blockSize; k++){
        if(C[bi * blockSize + bj] > A[bi * blockSize + k] + B[k * blockSize + bj])
            C[bi * blockSize + bj] = A[bi * blockSize + k] + B[k * blockSize + bj];
        __syncthreads();
    }
}
