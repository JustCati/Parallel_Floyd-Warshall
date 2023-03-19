#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "CudaFunctions.cuh"
#include "../Graph/graph.hpp"





int* simple_parallel_FW(const Graph& g){
    size_t memsize = g.getMatrixSize();
    const int* matrix = g.getAdjMatrix();

    int* d_matrix;
    cuda(cudaMalloc(&d_matrix, memsize));
    cuda(cudaMemcpy(d_matrix, matrix, memsize, cudaMemcpyHostToDevice));

    //* ----- INIT AND CALL KERNEL ------
    dim3 dimBlock = dim3(32, 32);
    dim3 numBlock = dim3((g.getNumVertices() + dimBlock.x - 1) / dimBlock.x, (g.getNumVertices() + dimBlock.x - 1) / dimBlock.y);

    for(int k = 0; k < g.getNumVertices(); k++)
        FW_simple_kernel<<<numBlock, dimBlock>>>(d_matrix, g.getNumVertices(), k);
    //* ----------------------------------

    int* h_matrix;
    cuda(cudaMallocHost(&h_matrix, memsize));
    cuda(cudaMemcpy(h_matrix, d_matrix, memsize, cudaMemcpyDeviceToHost));

    cuda(cudaFree(d_matrix));
    return h_matrix;
}
