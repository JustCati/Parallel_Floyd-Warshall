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


__forceinline__
__device__ void blockedUpdateFW(int* C, int* A, int* B, int i, int j, const int blockSize){
    for(int k = 0; k < blockSize; k++){
        int sum = A[i * blockSize + k] + B[k * blockSize + j];
        if(C[i * blockSize + j] > sum)
            C[i * blockSize + j] = sum;
        __syncthreads();
    }
}

// Aggiorna il blocco principale (k)
__global__ void blocked_FW_phase1(int* d_D, int n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ int lmem[];
    int* lmem_A = lmem;

    lmem_A[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

// Aggiorna i blocchi nella stessa riga e colonna del blocco principale (k)
__global__ void blocked_FW_phase2(int* d_D, int n, int k, const int blockSize){
    // Seleziona l'indice (diagonale) da cui poi andremo a osservare
    // i blocchi nella stessa riga e colonna del blocco principale
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    // Uno per il blocco da modificare e l'altro per il blocco di dipendenza
    extern __shared__ int lmem[];
    int* lmem_A = lmem;
    int* lmem_B = lmem + (blockSize * blockSize * sizeof(int));


    lmem_A[i * blockSize + j] = d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_B, i, j, blockSize);
    __syncthreads();

    // Aggiorno la matrice con le nuove dipendenze 
    d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];

    lmem_A[i * blockSize + j] = d_D[(k * blockSize * n) + (x * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_A, i, j, blockSize);
    __syncthreads();

    // Aggiorno la matrice con le nuove dipendenze
    d_D[(k * blockSize * n) + (x * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

// Aggiorna i blocchi restanti
__global__ void blocked_FW_phase3(int* d_D, int n, int k, const int blockSize){
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if(x == k && y == k)
        return;

    extern __shared__ int lmem[];
    int* lmem_A = lmem;
    int* lmem_B = lmem + (blockSize * blockSize * sizeof(int));
    int* lmem_C = lmem + (2 * blockSize * blockSize * sizeof(int));

    lmem_A[i * blockSize + j] = d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_C[i * blockSize + j] = d_D[(k * blockSize * n) + (y * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_C, i, j, blockSize);
    __syncthreads();

    d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}
