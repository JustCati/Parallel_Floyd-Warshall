#include <iostream>
#include <vector>
#include <numeric>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "CudaFunctions.cuh"
#include "../Graph/graph.hpp"

#define DEFAULT_BLOCK_SIZE 32



void printMetrics(std::string title, std::vector<std::string> outputs, std::vector<float> times){
    if (outputs.size() != times.size()){
        std::cout << "ERROR: outputs and times vectors are not the same size" << std::endl;
        return;
    }
    std::cout << title << std::endl << std::endl;
    for(int i = 0; i < outputs.size(); i++)
        std::cout << outputs[i] << times[i] << " ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Total time: " << std::accumulate(times.begin(), times.end(), 0.0) / 1000 << " s" << std::endl;
}


int* simple_parallel_FW(const int* g, int numVertices, int blockSize){
    int* d_matrix;
    size_t memsize = numVertices * numVertices * sizeof(int);

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
    cuda(cudaMemcpy(d_matrix, g, memsize, cudaMemcpyHostToDevice)); //* copy matrix to device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to device: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));

    //* ---------------------- KERNEL ---------------------- *//
    dim3 dimBlock = dim3(blockSize, blockSize);
    dim3 numBlock = dim3((numVertices + dimBlock.x - 1) / dimBlock.x, (numVertices + dimBlock.x - 1) / dimBlock.y);

    for(int k = 0; k < numVertices; k++)
        FW_simple_kernel<<<numBlock, dimBlock>>>(d_matrix, numVertices, k); //* call kernel
    
    //* ---------------------------------------------------- *//
    
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

    std::string title =  "Starting SIMPLE FW KERNEL with " + std::to_string(numVertices) + " nodes";
    printMetrics(title, outputs, times); //* print metrics

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}


int* blocked_parallel_FW(const int* g, int numVertices, int blockSize){
    int* d_matrix;
    size_t memsize = numVertices * numVertices * sizeof(int);

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
    cuda(cudaMemcpy(d_matrix, g, memsize, cudaMemcpyHostToDevice)); //* copy matrix to device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to device: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));

    //* ---------------------- KERNEL ---------------------- *//
    const int numBlocks = (numVertices + blockSize - 1) / blockSize;
    dim3 dimBlock = dim3(blockSize, blockSize);
    dim3 dimBlock_phase3 = dim3(numBlocks, numBlocks);
    size_t sharedMemSize = blockSize * blockSize * sizeof(int);

    for(int k = 0; k < numBlocks; k++){
        blocked_FW_phase1<<<1, dimBlock, sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
        blocked_FW_phase2<<<numBlocks, dimBlock, 2 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
        blocked_FW_phase3<<<dimBlock_phase3, dimBlock, 3 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
    }
    //* ------------------------------------------------------ *//
    
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

    std::string title =  "Starting BLOCKED FW KERNEL with " + std::to_string(numVertices) + " nodes";
    printMetrics(title, outputs, times); //* print metrics

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}
