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
    float elapsedTime;
    cudaEvent_t start, stop;

    cuda(cudaEventCreate(&start));
    cuda(cudaEventCreate(&stop));


    cuda(cudaEventRecord(start));
    cuda(cudaMalloc(&d_matrix, memsize));
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "CudaMalloc time: " << elapsedTime << " ms" << std::endl;

    cuda(cudaEventRecord(start));
    cuda(cudaMemcpy(d_matrix, matrix, memsize, cudaMemcpyHostToDevice));
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));    
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "CudaMemCpy to device time: " << elapsedTime << " ms" << std::endl;


    float totalElapsedTime = 0;
    //* ----- INIT AND CALL KERNEL ------
    dim3 dimBlock = dim3(32, 32);
    dim3 numBlock = dim3((g.getNumVertices() + dimBlock.x - 1) / dimBlock.x, (g.getNumVertices() + dimBlock.x - 1) / dimBlock.y);
    
    for(int k = 0; k < g.getNumVertices(); k++){
        cuda(cudaEventRecord(start));
        FW_simple_kernel<<<numBlock, dimBlock>>>(d_matrix, g.getNumVertices(), k);
        cuda(cudaEventRecord(stop));
        cuda(cudaEventSynchronize(stop));
        cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

        totalElapsedTime += elapsedTime;
    }
    //* ----------------------------------

    std::cout << "Total kernel time: " << totalElapsedTime << " ms" << std::endl;

    int* h_matrix;
    cuda(cudaEventRecord(start));
    cuda(cudaMallocHost(&h_matrix, memsize));
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "CudaMallocHost time: " << elapsedTime << " ms" << std::endl;

    cuda(cudaEventRecord(start));
    cuda(cudaMemcpy(h_matrix, d_matrix, memsize, cudaMemcpyDeviceToHost));
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));
    std::cout << "CudaMemCpy to host time: " << elapsedTime << " ms" << std::endl;

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}
