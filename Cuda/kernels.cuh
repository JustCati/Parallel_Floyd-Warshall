#pragma once
#include <cuda.h>
#include <cuda_runtime.h>



__global__ void FW_simple_kernel_pitch(short* d_D, int pitch, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    short* d_D_Pitch_i = (short*)((char*)d_D + i * pitch);
    short* d_D_Pitch_k = (short*)((char*)d_D + k * pitch);
  
    if (i < n && j < n) {
        short ij = d_D_Pitch_i[j];
        short ik = d_D_Pitch_i[k];
        short kj = d_D_Pitch_k[j];

        if (ik + kj < ij)
            d_D_Pitch_i[j] = ik + kj;
    }
}

__global__ void FW_simple_kernel(short* d_D, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (i < n && j < n) {
        short ij = d_D[i * n + j];
        short ik = d_D[i * n + k];
        short kj = d_D[k * n + j];

        if (ik + kj < ij)
            d_D[i * n + j] = ik + kj;
    }
}


__forceinline__
__device__ void blockedUpdateFW(short* C, short* A, short* B, int i, int j, const int blockSize){
    for(int k = 0; k < blockSize; k++){
        short sum = A[i * blockSize + k] + B[k * blockSize + j];
        if(C[i * blockSize + j] > sum)
            C[i * blockSize + j] = sum;
        __syncthreads();
    }
}

// Aggiorna il blocco principale (k)
__global__ void blocked_FW_phase1(short* d_D, int n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;

    lmem_A[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

// Aggiorna i blocchi nella stessa riga e colonna del blocco principale (k)
__global__ void blocked_FW_phase2(short* d_D, int n, int k, const int blockSize){
    // Seleziona l'indice (diagonale) da cui poi andremo a osservare
    // i blocchi nella stessa riga e colonna del blocco principale
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    // Uno per il blocco da modificare e l'altro per il blocco di dipendenza
    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);

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
__global__ void blocked_FW_phase3(short* d_D, int n, int k, const int blockSize){
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if(x == k && y == k)
        return;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);
    short* lmem_C = (short*)(&lmem_B[blockSize * blockSize]);

    lmem_A[i * blockSize + j] = d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_C[i * blockSize + j] = d_D[(k * blockSize * n) + (y * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_C, i, j, blockSize);
    __syncthreads();

    d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}
