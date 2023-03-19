#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void FW_simple_kernel(int* d_D, int n, int k);

