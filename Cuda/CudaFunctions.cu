#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "CudaFunctions.cuh"
#include "../Graph/graph.hpp"


void printMetrics(std::string title, std::vector<std::string> outputs, std::vector<float> times){
    if (outputs.size() != times.size()){
        std::cout << "ERROR: outputs and times vectors are not the same size" << std::endl;
        return;
    }
    std::cout << title << std::endl << std::endl;
    for(int i = 0; i < outputs.size(); i++)
        std::cout << outputs[i] << times[i] << " ms" << std::endl;
}


int* simple_parallel_FW(const Graph& g, bool cache){
    int* d_matrix;
    size_t memsize = g.getMatrixMemSize();
    const int* matrix = g.getAdjMatrix();

    float elapsedTime;
    std::vector<float> times;
    std::vector<std::string> outputs;

    cudaEvent_t start, stop;
    cuda(cudaEventCreate(&start));
    cuda(cudaEventCreate(&stop));


    cuda(cudaEventRecord(start));
    cuda(cudaMalloc(&d_matrix, memsize)); //* allocate memory on device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMalloc: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    cuda(cudaMemcpy(d_matrix, matrix, memsize, cudaMemcpyHostToDevice)); //* copy matrix to device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to device: ");
    times.push_back(elapsedTime);
    
    //* ----- INIT AND CALL KERNEL ------
    const int blockSize = g.getBlockSize();
    dim3 dimBlock = dim3(blockSize, blockSize);
    dim3 numBlock = dim3((g.getNumVertices() + dimBlock.x - 1) / dimBlock.x, (g.getNumVertices() + dimBlock.x - 1) / dimBlock.y);

    cuda(cudaEventRecord(start));

    for(int k = 0; k < g.getNumVertices(); k++)
        FW_simple_kernel<<<numBlock, dimBlock>>>(d_matrix, g.getNumVertices(), k); //* call kernel
    
    //* ----------------------------------
    
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("Total kernel: ");
    times.push_back(elapsedTime);

    int* h_matrix;
    cuda(cudaEventRecord(start));
    cuda(cudaMallocHost(&h_matrix, memsize)); //* allocate memory on host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMallocHost: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    cuda(cudaMemcpy(h_matrix, d_matrix, memsize, cudaMemcpyDeviceToHost)); //* copy matrix to host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to host: ");
    times.push_back(elapsedTime);

    if(!cache)
        printMetrics("Starting SIMPLE FW KERNEL", outputs, times); //* print metrics

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}
