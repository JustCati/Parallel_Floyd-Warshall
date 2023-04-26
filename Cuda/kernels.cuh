#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#define ll long long


__forceinline__ __host__ __device__ 
short4 operator+(const short4& a, const short4& b) {
    return make_short4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __host__ __device__
short4 checkWeight(const short4& a, const short4& b) {
    return make_short4(a.x < b.x ? a.x : b.x,
                        a.y < b.y ? a.y : b.y,
                        a.z < b.z ? a.z : b.z,
                        a.w < b.w ? a.w : b.w);
}


__global__ void FW_simple_kernel(short* d_D, ll n, int k) {
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

__global__ void FW_simple_kernel_pitch(short* d_D, ll pitch, ll n, int k) {
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

__global__ void FW_simple_kernel_vectorized(short4 *d_D, ll n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < (n >> 2)) {
        short tempIk;
        short4 ij, ik, kj;
        ll numElem = n >> 2;

        ij = d_D[i * numElem + j];
        kj = d_D[k * numElem + j];

        int mask = ~((~0) << 2);
        int lsb_2 = (k & mask);

#if 1 // "brutto ma veloce" (~0.5s più veloce con 2^13)
        tempIk = *(((short*)(d_D + i * numElem + (k >> 2))) + lsb_2);
#else // "pulito ma più lento"
        if(lsb_2 == 0)
            tempIk = d_D[i * numElem + (k >> 2)].x;
        if(lsb_2 == 1)
            tempIk = d_D[i * numElem + (k >> 2)].y;
        if(lsb_2 == 2)
            tempIk = d_D[i * numElem + (k >> 2)].z;
        if(lsb_2 == 3)
            tempIk = d_D[i * numElem + (k >> 2)].w;
#endif

        ik = make_short4(tempIk, tempIk, tempIk, tempIk);

        short4 res = ik + kj;
        d_D[i * numElem + j] = checkWeight(res, ij);
    }
}

__global__ void FW_simple_kernel_vectorized_pitch(short4* d_D, ll pitch, ll n, int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < (n >> 2)) {
        short tempIk;
        short4 ij, ik, kj;

        short4 *d_D_Pitch_i = (short4*)((char*)d_D + i * pitch);
        short4 *d_D_Pitch_k = (short4*)((char*)d_D + k * pitch);

        ij = d_D_Pitch_i[j];
        kj = d_D_Pitch_k[j];

        int mask = ~((~0) << 2);
        int lsb_2 = (k & mask);

#if 1 // "brutto ma veloce" (~0.5s più veloce con 2^13)
        tempIk = *(((short*)(d_D_Pitch_i + (k >> 2))) + lsb_2);
#else // "pulito ma più lento"
        if(lsb_2 == 0)
            tempIk = d_D_Pitch_i[(k >> 2)].x;
        if(lsb_2 == 1)
            tempIk = d_D_Pitch_i[(k >> 2)].y;
        if(lsb_2 == 2)
            tempIk = d_D_Pitch_i[(k >> 2)].z;
        if(lsb_2 == 3)
            tempIk = d_D_Pitch_i[(k >> 2)].w;
#endif
        ik = make_short4(tempIk, tempIk, tempIk, tempIk);

        short4 res = ik + kj;
        d_D_Pitch_i[j] = checkWeight(res, ij);
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
__global__ void blocked_FW_phase1(short* d_D, ll n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
 
    // A: blocco principale nella diagonale da aggiornare
    lmem_A[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

__global__ void blocked_FW_phase1_pitch(short* d_D, int pitch, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* d_D_Pitch_main = (short*)((char*)d_D + (k * blockSize * pitch) + (k * blockSize) + (i * pitch));

    lmem_A[i * blockSize + j] = d_D_Pitch_main[j];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_main[j] = lmem_A[i * blockSize + j];
}

// Aggiorna i blocchi nella stessa riga e colonna del blocco principale (k)
__global__ void blocked_FW_phase2(short* d_D, ll n, int k, const int blockSize){
    // Seleziona l'indice (diagonale) da cui poi andremo a osservare
    // i blocchi nella stessa riga e colonna del blocco principale
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    // A = blocco temporaneo da modificare usando B
    // B = blocco principale (k) da cui dipende A
    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);

    // Fase 2.1: carica in A il blocco nella stessa colonna di B (la riga (in blocchi) è indicizzata da x)
    lmem_A[i * blockSize + j] = d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_B, i, j, blockSize);
    __syncthreads();

    d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];

    // Fase 2.2: carica in A il blocco nella stessa riga di B (la colonna (in blocchi) è indicizzata da x)
    lmem_A[i * blockSize + j] = d_D[(k * blockSize * n) + (x * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D[(k * blockSize * n) + (x * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

__global__ void blocked_FW_phase2_pitch(short* d_D, int pitch, int k, const int blockSize){
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);


    short* d_D_Pitch_col = (short*)((char*)d_D + (x * blockSize * pitch) + (k * blockSize) + (i * pitch));
    short* d_D_Pitch_main = (short*)((char*)d_D + (k * blockSize * pitch) + (k * blockSize) + (i * pitch));
    short* d_D_Pitch_row = (short*)((char*)d_D + (k * blockSize * pitch) + (x * blockSize) + (i * pitch));


    lmem_A[i * blockSize + j] = d_D_Pitch_col[j];
    lmem_B[i * blockSize + j] = d_D_Pitch_main[j];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_A, lmem_B, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_col[j] = lmem_A[i * blockSize + j];

    lmem_A[i * blockSize + j] = d_D_Pitch_row[j];
    lmem_B[i * blockSize + j] = d_D_Pitch_main[j];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_A, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_row[j] = lmem_A[i * blockSize + j];
}

// Aggiorna i blocchi restanti, che non sono nella stessa riga o colonna del blocco principale (k)
// Ognuno di questi blocchi dipende dai corrispondenti blocchi 
// perpendicolari nella riga e colonna del blocco principale
__global__ void blocked_FW_phase3(short* d_D, ll n, int k, const int blockSize){
    // x: seleziona la riga (in blocchi)
    // y: seleziona la colonna (in blocchi)
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    // se x o y è uguale a k allora il blocco 
    // è stato già aggiornato nella fase 2
    if(x == k || y == k)
        return;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);
    short* lmem_C = (short*)(&lmem_B[blockSize * blockSize]);

    // A: blocco da aggiornare
    // B: blocco nella stessa colonna del blocco principale e nella stessa riga di A
    // C: blocco nella stessa riga del blocco principale e nella stessa colonna di A
    lmem_A[i * blockSize + j] = d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)];
    lmem_B[i * blockSize + j] = d_D[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_C[i * blockSize + j] = d_D[(k * blockSize * n) + (y * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_C, i, j, blockSize);
    __syncthreads();

    d_D[(x * blockSize * n) + (y * blockSize) + (i * n + j)] = lmem_A[i * blockSize + j];
}

__global__ void blocked_FW_phase3_pitch(short* d_D, int pitch, int k, const int blockSize){
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if(x == k || y == k)
        return;

    extern __shared__ short lmem[];
    short* lmem_A = (short*)lmem;
    short* lmem_B = (short*)(&lmem_A[blockSize * blockSize]);
    short* lmem_C = (short*)(&lmem_B[blockSize * blockSize]);

    short* d_D_Pitch_x = (short*)((char*)d_D + (x * blockSize * pitch) + (y * blockSize) + (i * pitch));
    short* d_D_Pitch_k = (short*)((char*)d_D + (x * blockSize * pitch) + (k * blockSize) + (i * pitch));
    short* d_D_Pitch_y = (short*)((char*)d_D + (k * blockSize * pitch) + (y * blockSize) + (i * pitch));

    lmem_A[i * blockSize + j] = d_D_Pitch_x[j];
    lmem_B[i * blockSize + j] = d_D_Pitch_k[j];
    lmem_C[i * blockSize + j] = d_D_Pitch_y[j];
    __syncthreads();

    blockedUpdateFW(lmem_A, lmem_B, lmem_C, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_x[j] = lmem_A[i * blockSize + j];
}
