#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../Graph/graph.hpp"


#define cuda(fun) { cudaCheck(fun, __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t err, const char *file, int line){
    if(err != cudaSuccess){
        std::cerr << "Errore " << cudaGetErrorName(err) << " nel file " << file << " alla riga " \
                << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}


int* simple_parallel_FW(const Graph& g);


