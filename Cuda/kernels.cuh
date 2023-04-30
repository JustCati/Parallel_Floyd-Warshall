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
    return make_short4( a.x < b.x ? a.x : b.x,
                        a.y < b.y ? a.y : b.y,
                        a.z < b.z ? a.z : b.z,
                        a.w < b.w ? a.w : b.w );
}


__global__ void FW_simple_kernel(short *graph, ll n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
  
    if (i < n && j < n) {
        short ij = graph[i * n + j];
        short ik = graph[i * n + k];
        short kj = graph[k * n + j];

        if (ik + kj < ij)
            graph[i * n + j] = ik + kj;
    }
}

__global__ void FW_simple_kernel_pitch(short *graph, ll pitch, ll n, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    short *d_D_Pitch_i = (short*)((char*)graph + i * pitch);
    short *d_D_Pitch_k = (short*)((char*)graph + k * pitch);
  
    if (i < n && j < n) {
        short ij = d_D_Pitch_i[j];
        short ik = d_D_Pitch_i[k];
        short kj = d_D_Pitch_k[j];

        if (ik + kj < ij)
            d_D_Pitch_i[j] = ik + kj;
    }
}

__global__ void FW_simple_kernel_vectorized(short4 *graph, ll n, int k){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < (n >> 2)) {
        short tempIk;
        short4 ij, ik, kj;
        ll numElem = n >> 2;

        ij = graph[i * numElem + j];
        kj = graph[k * numElem + j];

        int mask = ~((~0) << 2);
        int lsb_2 = (k & mask);

        // if(lsb_2 == 0)
        //     tempIk = graph[i * numElem + (k >> 2)].x;
        // if(lsb_2 == 1)
        //     tempIk = graph[i * numElem + (k >> 2)].y;
        // if(lsb_2 == 2)
        //     tempIk = graph[i * numElem + (k >> 2)].z;
        // if(lsb_2 == 3)
        //     tempIk = graph[i * numElem + (k >> 2)].w;

        tempIk = *(((short*)(graph + i * numElem + (k >> 2))) + lsb_2);
        ik = make_short4(tempIk, tempIk, tempIk, tempIk);

        short4 res = ik + kj;
        graph[i * numElem + j] = checkWeight(res, ij);
    }
}

__global__ void FW_simple_kernel_vectorized_pitch(short4 *graph, ll pitch, ll n, int k){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < (n >> 2)) {
        short tempIk;
        short4 ij, ik, kj;

        short4 *d_D_Pitch_i = (short4*)((char*)graph + i * pitch);
        short4 *d_D_Pitch_k = (short4*)((char*)graph + k * pitch);

        ij = d_D_Pitch_i[j];
        kj = d_D_Pitch_k[j];

        int mask = ~((~0) << 2);
        int lsb_2 = (k & mask);

        // if(lsb_2 == 0)
        //     tempIk = d_D_Pitch_i[(k >> 2)].x;
        // if(lsb_2 == 1)
        //     tempIk = d_D_Pitch_i[(k >> 2)].y;
        // if(lsb_2 == 2)
        //     tempIk = d_D_Pitch_i[(k >> 2)].z;
        // if(lsb_2 == 3)
        //     tempIk = d_D_Pitch_i[(k >> 2)].w;

        tempIk = *(((short*)(d_D_Pitch_i + (k >> 2))) + lsb_2);
        ik = make_short4(tempIk, tempIk, tempIk, tempIk);

        short4 res = ik + kj;
        d_D_Pitch_i[j] = checkWeight(res, ij);
    }
}


__forceinline__
__device__ void blockedUpdateFW(short *graph_ij, short *graph_ik, short *graph_kj, int i, int j, const int blockSize){
    for(int k = 0; k < blockSize; k++){
        short sum = graph_ik[i * blockSize + k] + graph_kj[k * blockSize + j];
        if(graph_ij[i * blockSize + j] > sum)
            graph_ij[i * blockSize + j] = sum;
        __syncthreads();
    }
}


__forceinline__
__device__ void blockedUpdateFW_vectorized(short4 *graph_ij, short4 *graph_ik, short4 *graph_kj, int i, int j, const int blockSize){
    for(int k = 0; k < blockSize; k++){
        short4 ij = graph_ij[i * blockSize + j];
        short4 kj = graph_kj[k * blockSize + j];

        int mask = ~((~0) << 2);
        int lsb_2 = (k & mask);
        short tempIk = *(((short*)(graph_ik + i * blockSize + (k >> 2))) + lsb_2);

        short4 ik = make_short4(tempIk, tempIk, tempIk, tempIk);
        short4 sum = ik + kj;

        graph_ij[i * blockSize + j] = checkWeight(sum, ij);
        __syncthreads();
    }
}


// Aggiorna il blocco principale (k)
__global__ void blocked_FW_phase1(short *graph, ll n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem[];
 
    // lmem: blocco principale nella diagonale da aggiornare
    lmem[i * blockSize + j] = graph[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem, lmem, lmem, i, j, blockSize);
    __syncthreads();

    graph[(k * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem[i * blockSize + j];
}

__global__ void blocked_FW_phase1_pitch(short *graph, ll pitch, ll n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem[];

    // Porta i puntatori fino alla riga usando il pitch, l'offset per
    // le colonne viene gestito poi in "colonne" (pitch / sizeof(T))
    short *d_D_Pitch_main = (short*)((char*)graph + (k * blockSize * pitch));

    lmem[i * blockSize + j] = d_D_Pitch_main[(k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem, lmem, lmem, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_main[(k * blockSize) + (i * n + j)] = lmem[i * blockSize + j];
}

__global__ void blocked_FW_phase1_vectorized(short *graph, ll n, int k, const int blockSize){
    int i = threadIdx.y;
    int j = threadIdx.x;

    extern __shared__ short lmem_p[];
    short4 *lmem = (short4*)lmem_p;
    short4 *graph_Block = (short4*)&graph[(k * blockSize * n) + (k * blockSize) + (i * n)];

    lmem[i * blockSize + j] = graph_Block[j];
    __syncthreads();

    blockedUpdateFW_vectorized(lmem, lmem, lmem, i, j, blockSize);
    __syncthreads();

    graph_Block[j] = lmem[i * blockSize + j];
}

// Aggiorna i blocchi nella stessa riga e colonna del blocco principale (k)
__global__ void blocked_FW_phase2(short *graph, ll n, int k, const int blockSize){
    // Seleziona l'indice (diagonale) da cui poi andremo a osservare
    // i blocchi nella stessa riga e colonna del blocco principale
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    // block = blocco temporaneo da modificare usando il blocco principale (k)
    // main = blocco principale (k) da cui dipende il blocco "block" temporaneo
    extern __shared__ short lmem[];
    short *lmem_Block = (short*)lmem;
    short *lmem_Main = (short*)(&lmem_Block[blockSize * blockSize]);


    // Fase 2.1: carica in "block" il blocco nella stessa colonna di "main" (la riga (in blocchi) è indicizzata da x)
    lmem_Block[i * blockSize + j] = graph[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_Main[i * blockSize + j] = graph[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Block, lmem_Main, i, j, blockSize);
    __syncthreads();

    graph[(x * blockSize * n) + (k * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];

    // Fase 2.2: carica in "block" il blocco nella stessa riga di "main" (la colonna (in blocchi) è indicizzata da x)
    lmem_Block[i * blockSize + j] = graph[(k * blockSize * n) + (x * blockSize) + (i * n + j)];
    lmem_Main[i * blockSize + j] = graph[(k * blockSize * n) + (k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Main, lmem_Block, i, j, blockSize);
    __syncthreads();

    graph[(k * blockSize * n) + (x * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];
}

__global__ void blocked_FW_phase2_pitch(short *graph, ll pitch, ll n, int k, const int blockSize){
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    extern __shared__ short lmem[];
    short *lmem_Block = (short*)lmem;
    short *lmem_Main = (short*)(&lmem_Block[blockSize * blockSize]);

    // Porta i puntatori fino alla riga usando il pitch, l'offset per
    // le colonne viene gestito poi in "colonne" (pitch / sizeof(T))
    short *d_D_Pitch_col = (short*)((char*)graph + (x * blockSize * pitch));
    short *d_D_Pitch_main = (short*)((char*)graph + (k * blockSize * pitch));
    short *d_D_Pitch_row = (short*)((char*)graph + (k * blockSize * pitch));

    lmem_Block[i * blockSize + j] = d_D_Pitch_col[(k * blockSize) + (i * n + j)];
    lmem_Main[i * blockSize + j] = d_D_Pitch_main[(k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Block, lmem_Main, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_col[(k * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];

    lmem_Block[i * blockSize + j] = d_D_Pitch_row[(x * blockSize) + (i * n + j)];
    lmem_Main[i * blockSize + j] = d_D_Pitch_main[(k * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Main, lmem_Block, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_row[(x * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];
}

__global__ void blocked_FW_phase2_vectorized(short *graph, ll n, int k, const int blockSize){
    int x = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if (x == k)
        return;

    extern __shared__ short lmem[];
    short4 *lmem_Block = (short4*)lmem;
    short4 *lmem_Main = (short4*)&lmem_Block[blockSize * blockSize];

    short4 *graph_Block = (short4*)&graph[(x * blockSize * n) + (k * blockSize) + (i * n)];
    short4 *graph_Main = (short4*)&graph[(k * blockSize * n) + (k * blockSize) + (i * n)];

    lmem_Block[i * blockSize + j] = graph_Block[j];
    lmem_Main[i * blockSize + j] = graph_Main[j];
    __syncthreads();

    blockedUpdateFW_vectorized(lmem_Block, lmem_Block, lmem_Main, i, j, blockSize);
    __syncthreads();
    
    graph_Block[j] = lmem_Block[i * blockSize + j];

    graph_Block = (short4*)&graph[(k * blockSize * n) + (x * blockSize) + (i * n)];
    lmem_Block[i * blockSize + j] = graph_Block[j];
    lmem_Main[i * blockSize + j] = graph_Main[j];
    __syncthreads();
    
    blockedUpdateFW_vectorized(lmem_Block, lmem_Main, lmem_Block, i, j, blockSize);
    __syncthreads();

    graph_Block[j] = lmem_Block[i * blockSize + j];
}


// Aggiorna i blocchi restanti, che non sono nella stessa riga o colonna del blocco principale (k)
// Ognuno di questi blocchi dipende dai corrispondenti blocchi 
// perpendicolari nella riga e colonna del blocco principale
__global__ void blocked_FW_phase3(short *graph, ll n, int k, const int blockSize){
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
    short *lmem_Block = (short*)lmem;
    short *lmem_Col = (short*)(&lmem_Block[blockSize * blockSize]);
    short *lmem_Row = (short*)(&lmem_Col[blockSize * blockSize]);

    // block: blocco da aggiornare
    // row: blocco nella stessa colonna del blocco principale e nella stessa riga di "block"
    // col: blocco nella stessa riga del blocco principale e nella stessa colonna di "block"
    lmem_Block[i * blockSize + j] = graph[(x * blockSize * n) + (y * blockSize) + (i * n + j)];
    lmem_Col[i * blockSize + j] = graph[(x * blockSize * n) + (k * blockSize) + (i * n + j)];
    lmem_Row[i * blockSize + j] = graph[(k * blockSize * n) + (y * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Col, lmem_Row, i, j, blockSize);
    __syncthreads();

    graph[(x * blockSize * n) + (y * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];
}

__global__ void blocked_FW_phase3_pitch(short *graph, ll pitch, ll n, int k, const int blockSize){
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if(x == k || y == k)
        return;

    extern __shared__ short lmem[];
    short *lmem_Block = (short*)lmem;
    short *lmem_Col = (short*)(&lmem_Block[blockSize * blockSize]);
    short *lmem_Row = (short*)(&lmem_Col[blockSize * blockSize]);

    // Porta i puntatori fino alla riga usando il pitch, l'offset per
    // le colonne viene gestito poi in "colonne" (pitch / sizeof(T))
    short *d_D_Pitch_block = (short*)((char*)graph + (x * blockSize * pitch));
    short *d_D_Pitch_col = (short*)((char*)graph + (x * blockSize * pitch));
    short *d_D_Pitch_row = (short*)((char*)graph + (k * blockSize * pitch));

    lmem_Block[i * blockSize + j] = d_D_Pitch_block[(y * blockSize) + (i * n + j)];
    lmem_Col[i * blockSize + j] = d_D_Pitch_col[(k * blockSize) + (i * n + j)];
    lmem_Row[i * blockSize + j] = d_D_Pitch_row[(y * blockSize) + (i * n + j)];
    __syncthreads();

    blockedUpdateFW(lmem_Block, lmem_Col, lmem_Row, i, j, blockSize);
    __syncthreads();

    d_D_Pitch_block[(y * blockSize) + (i * n + j)] = lmem_Block[i * blockSize + j];
}

__global__ void blocked_FW_phase3_vectorized(short *graph, ll n, int k, const int blockSize){
    int x = blockIdx.y;
    int y = blockIdx.x;

    int i = threadIdx.y;
    int j = threadIdx.x;

    if(x == k || y == k)
        return;

    extern __shared__ short lmem[];
    short4 *lmem_Block = (short4*)lmem;
    short4 *lmem_Col = (short4*)&lmem_Block[blockSize * blockSize];
    short4 *lmem_Row = (short4*)&lmem_Col[blockSize * blockSize];

    short4 *graph_Block = (short4*)(&graph[(x * blockSize * n) + (y * blockSize) + (i * n)]);
    short4 *graph_Col = (short4*)(&graph[(x * blockSize * n) + (k * blockSize) + (i * n)]);
    short4 *graph_Row = (short4*)(&graph[(k * blockSize * n) + (y * blockSize) + (i * n)]);

    lmem_Block[i * blockSize + j] = graph_Block[j];
    lmem_Col[i * blockSize + j] = graph_Col[j];
    lmem_Row[i * blockSize + j] = graph_Row[j];
    __syncthreads();

    blockedUpdateFW_vectorized(lmem_Block, lmem_Col, lmem_Row, i, j, blockSize);
    __syncthreads();

    graph_Block[j] = lmem_Block[i * blockSize + j];
}
