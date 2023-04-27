#include <iostream>
#include <fstream>
#include <cuda_runtime.h>


__global__ void kernel(short *A, size_t pitch, int n){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    short *arr = (short*)((char*)A + i * pitch);
    if(i >= n || j >= pitch)
        arr[j] = 5;
    else
        arr[j] = 1;
}


int main(){
    const int n = 4;
    size_t pitch;
    short *A;

    cudaMallocPitch(&A, &pitch, n * sizeof(short), n);

    std::cout << "pitch: " << pitch  << " widht: "<< 10*sizeof(short) <<  std::endl;

    dim3 block(16, 16);
    dim3 grid(pitch, n);
    kernel<<<grid, block>>>(A, pitch, n);

    cudaDeviceSynchronize();

    short *h_A;
    cudaMallocHost(&h_A, pitch * 10);
    cudaMemcpy2D(h_A, pitch, A, pitch, pitch, n, cudaMemcpyDeviceToHost);
    
    std::ofstream out("out.txt");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < pitch; j++)
            out << h_A[i * pitch + j] << "\t";
        out << std::endl;
    }
    out.close();
    
    cudaFreeHost(h_A);
    cudaFree(A);
    return 0;
}

