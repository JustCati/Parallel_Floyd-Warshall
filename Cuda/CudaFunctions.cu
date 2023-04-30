#include <iostream>
#include <vector>
#include <numeric>

#include <cuda.h>
#include <cuda_runtime.h>

#include "kernels.cuh"
#include "CudaFunctions.cuh"
#include "../Graph/graph.hpp"

#define ll long long


void printMetrics(std::string title, std::vector<std::string> outputs, std::vector<float> times){
    if (outputs.size() != times.size())
        throw std::runtime_error("Outputs e times vectors sono di dimensioni diverse");
    
    std::cout << title << std::endl << std::endl;
    for(int i = 0; i < outputs.size(); i++){
        std::cout << outputs[i] << times[i];
        if(outputs[i].find("Bandwidth") == std::string::npos)
            std::cout << " ms";
        else
            std::cout << " GB/s";
        std::cout << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "Total Function time: " << std::accumulate(times.begin(), times.end(), 0.0) / 1000 << " s" << std::endl;
}

short* simple_parallel_FW(const short *g, ll numVertices, int blockSize, bool usePitch, bool vectorize, bool debug){
    size_t pitch = 0;
    short *d_matrix, *h_matrix;
    size_t singleRow_memsize, memsize;

    if (vectorize){
        singleRow_memsize = (numVertices >> 2) * sizeof(short4);
        memsize = (numVertices >> 2) * numVertices * sizeof(short4);
    }
    else{
        singleRow_memsize = numVertices * sizeof(short);
        memsize = numVertices * numVertices * sizeof(short);
    }

    float elapsedTime;
    std::vector<float> times;
    std::vector<std::string> outputs;

    cudaEvent_t start, stop;
    cuda(cudaEventCreate(&start));
    cuda(cudaEventCreate(&stop));

    cuda(cudaEventRecord(start));
    if (usePitch){
        cuda(cudaMallocPitch(&d_matrix, &pitch, singleRow_memsize, numVertices)); //* allocate memory on device
    }
    else
        cuda(cudaMalloc(&d_matrix, memsize)); //* allocate memory on device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMalloc: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    if (usePitch){
        cuda(cudaMemcpy2D(d_matrix, pitch, g, singleRow_memsize, singleRow_memsize, numVertices, cudaMemcpyHostToDevice)); //* copy matrix to device
    }
    else
        cuda(cudaMemcpy(d_matrix, g, memsize, cudaMemcpyHostToDevice)); //* copy matrix to device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to device: ");
    times.push_back(elapsedTime);
    outputs.push_back("CudaMemCpy to device Bandwidth: ");
    times.push_back(memsize / elapsedTime / 1.0e6);

    cuda(cudaEventRecord(start));

    //* ---------------------- KERNEL ---------------------- *//
    dim3 dimBlock = dim3(blockSize, blockSize);
    dim3 numBlock = dim3((numVertices + dimBlock.x - 1) / dimBlock.x, (numVertices + dimBlock.y - 1) / dimBlock.y);

    if(!vectorize){
        if(usePitch)
            for(int k = 0; k < numVertices; k++)
                FW_simple_kernel_pitch<<<numBlock, dimBlock>>>(d_matrix, pitch, numVertices, k); //* call kernel
        else
            for(int k = 0; k < numVertices; k++)
                FW_simple_kernel<<<numBlock, dimBlock>>>(d_matrix, numVertices, k); //* call kernel
    }
    else{ //* vectorize with short4 type (default)
        dimBlock = dim3(blockSize >> 2, blockSize);
        if(usePitch)
            for(int k = 0; k < numVertices; k++)
                FW_simple_kernel_vectorized_pitch<<<numBlock, dimBlock>>>((short4*)d_matrix, pitch, numVertices >> 2, k); //* call kernel
        else
            for(int k = 0; k < numVertices; k++)
                FW_simple_kernel_vectorized<<<numBlock, dimBlock>>>((short4*)d_matrix, numVertices >> 2, k); //* call kernel
    }

    //* ---------------------------------------------------- *//

    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("Total kernel call: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    cuda(cudaMallocHost(&h_matrix, memsize)); //* allocate memory on host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMallocHost: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    if (usePitch){
        cuda(cudaMemcpy2D(h_matrix, singleRow_memsize, d_matrix, pitch, singleRow_memsize, numVertices, cudaMemcpyDeviceToHost)); //* copy matrix to host
    }
    else
        cuda(cudaMemcpy(h_matrix, d_matrix, memsize, cudaMemcpyDeviceToHost)); //* copy matrix to host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to Host: ");
    times.push_back(elapsedTime);
    outputs.push_back("CudaMemCpy to Host Bandwidth: ");
    times.push_back(memsize / elapsedTime / 1.0e6);

    if(!debug){
        std::string title =  "Starting SIMPLE FW KERNEL with " + std::to_string(numVertices) +\
        " nodes" + (usePitch ? " with pitch," : "") + (vectorize ? " with vectorization" : "");
        printMetrics(title, outputs, times); //* print metrics
    }

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}

short* blocked_parallel_FW(const short *g, ll numVertices, int blockSize, bool usePitch, bool vectorize){
    size_t pitch = 0;
    short *d_matrix, *h_matrix;
    size_t singleRow_memsize, memsize;

    if (vectorize){
        singleRow_memsize = (numVertices >> 2) * sizeof(short4);
        memsize = (numVertices >> 2) * numVertices * sizeof(short4);
    }
    else{
        singleRow_memsize = numVertices * sizeof(short);
        memsize = numVertices * numVertices * sizeof(short);
    }

    float elapsedTime;
    std::vector<float> times;
    std::vector<std::string> outputs;

    cudaEvent_t start, stop;
    cuda(cudaEventCreate(&start));
    cuda(cudaEventCreate(&stop));

    cuda(cudaEventRecord(start));
    if(usePitch){
        cuda(cudaMallocPitch(&d_matrix, &pitch, singleRow_memsize, numVertices)); //* allocate memory on device
    }
    else
        cuda(cudaMalloc(&d_matrix, memsize)); //* allocate memory on device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMalloc: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    if(usePitch){
        cuda(cudaMemcpy2D(d_matrix, pitch, g, singleRow_memsize, singleRow_memsize, numVertices, cudaMemcpyHostToDevice)); //* copy matrix to device
    }
    else
        cuda(cudaMemcpy(d_matrix, g, memsize, cudaMemcpyHostToDevice)); //* copy matrix to device
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to device: ");
    times.push_back(elapsedTime);
    outputs.push_back("CudaMemCpy to device Bandwidth: ");
    times.push_back(memsize / elapsedTime / 1.0e6);

    cuda(cudaEventRecord(start));

    //* ---------------------- KERNEL ---------------------- *//
    const int numBlocks = (numVertices + blockSize - 1) / blockSize;

    dim3 dimBlock = dim3(blockSize, blockSize);
    dim3 dimBlock_phase3 = dim3(numBlocks, numBlocks);
    const size_t sharedMemSize = blockSize * blockSize * sizeof(short);

    if(vectorize){
        dimBlock = dim3(blockSize >> 2, blockSize);
        std::cout << "dimBlock: " << dimBlock.x << " " << dimBlock.y << std::endl;
        std::cout << "dimBlock_phase3: " << dimBlock_phase3.x << " " << dimBlock_phase3.y << std::endl;

        for(int k = 0; k < numBlocks; k++){
            blocked_FW_phase1_vectorized<<<1, dimBlock, sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
            blocked_FW_phase2_vectorized<<<numBlocks, dimBlock, 2 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
            blocked_FW_phase3_vectorized<<<dimBlock_phase3, dimBlock, 3 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
        }
    }
    else{
        if(usePitch){
            for(int k = 0; k < numBlocks; k++){
                blocked_FW_phase1_pitch<<<1, dimBlock, sharedMemSize>>>(d_matrix, pitch, pitch / sizeof(short), k, blockSize);
                blocked_FW_phase2_pitch<<<numBlocks, dimBlock, 2 * sharedMemSize>>>(d_matrix, pitch, pitch / sizeof(short), k, blockSize);
                blocked_FW_phase3_pitch<<<dimBlock_phase3, dimBlock, 3 * sharedMemSize>>>(d_matrix, pitch, pitch / sizeof(short), k, blockSize);
            }
        }
        else{
            for(int k = 0; k < numBlocks; k++){
                blocked_FW_phase1<<<1, dimBlock, sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
                blocked_FW_phase2<<<numBlocks, dimBlock, 2 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
                blocked_FW_phase3<<<dimBlock_phase3, dimBlock, 3 * sharedMemSize>>>(d_matrix, numVertices, k, blockSize);
            }
        }
    }
    //* ------------------------------------------------------ *//

    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("Total kernel call: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    cuda(cudaMallocHost(&h_matrix, memsize)); //* allocate memory on host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMallocHost: ");
    times.push_back(elapsedTime);

    cuda(cudaEventRecord(start));
    if(usePitch){
        cuda(cudaMemcpy2D(h_matrix, singleRow_memsize, d_matrix, pitch, singleRow_memsize, numVertices, cudaMemcpyDeviceToHost)); //* copy matrix to host
    }
    else
        cuda(cudaMemcpy(h_matrix, d_matrix, memsize, cudaMemcpyDeviceToHost)); //* copy matrix to host
    cuda(cudaEventRecord(stop));
    cuda(cudaEventSynchronize(stop));
    cuda(cudaEventElapsedTime(&elapsedTime, start, stop));

    outputs.push_back("CudaMemCpy to host: ");
    times.push_back(elapsedTime);
    outputs.push_back("CudaMemCpy to host Bandwidth: ");
    times.push_back(memsize / elapsedTime / 1.0e6);

    std::string title =  "Starting BLOCKED FW KERNEL with " + std::to_string(numVertices) +\
    " nodes" + (usePitch ? " with pitch," : ""); // + (vectorize ? " with vectorization" : "");
    printMetrics(title, outputs, times); //* print metrics

    cuda(cudaEventDestroy(start));
    cuda(cudaEventDestroy(stop));
    cuda(cudaFree(d_matrix));
    return h_matrix;
}
