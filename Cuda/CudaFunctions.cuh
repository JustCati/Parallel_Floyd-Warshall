#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../Graph/graph.hpp"

#define DEFAULT_BLOCK_SIZE 32


#define cuda(fun) { cudaCheck(fun, __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        std::cerr << "Errore " << cudaGetErrorName(err) << " nel file " << file << " alla riga " \
                << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}


short* simple_parallel_FW(const short* g, int numVertices, bool usePitch = false, int blockSize = DEFAULT_BLOCK_SIZE, bool debug = false);

short* simple_parallel_FW_vec(const short* g, int numVertices, int blockSize = DEFAULT_BLOCK_SIZE, bool debug = false);

short* blocked_parallel_FW(const short* g, int numVertices, int blockSize = DEFAULT_BLOCK_SIZE);
