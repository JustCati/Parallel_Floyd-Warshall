#include <iostream>
#include <unistd.h>
#include <map>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"

#define ll long long
#define DEFAULT_SEED 1234
#define DEFAULT_BLOCK_SIZE 16

/*
    -s: seed
    -c: check / verify results
    -V: (verbose) print results matrix
    -v: vectorized if possible with short4
    -b <block size>: Set block size for GPU
    -p <percentage>: Set percentage for Erdos-Renyi graph generation
    -a <algorithm>: Set algorithm to use (1: cpu, 1: simple, 2: blocked)
*/
int main(int argc, char **argv){
    int perc = 50, blockSize = DEFAULT_BLOCK_SIZE, algorithm = 0, seed = DEFAULT_SEED;
    bool toVerify = false, printResults = false, vectorize = false;

    if(argc < 2 || argc > 13)
        throw std::invalid_argument("Utilizzo comando: ./parallel_fw [-s] seed [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-V] [-v] numVertices");
        
    short *graph = nullptr;
    const ll numVertices = atoll(argv[argc - 1]);

    int opt;
    extern char *optarg;
    std::map<short, short> sqrts = {{1024, 32}, {256, 16}, {64, 8}, {16, 4}};
    while((opt = getopt(argc, argv, "s:p:b:a:cvV")) != -1){
        switch(opt){
            case 's':
                seed = atoi(optarg);
                break;
            case 'p':
                perc = atoi(optarg);
                if(perc <= 0 || perc >= 100)
                    throw std::invalid_argument("Inserire percentuale compreso tra 0 e 100 (estremi esclusi)");
                break;
            case 'b':
                blockSize = atoi(optarg);
                if(sqrts.find(blockSize) == sqrts.end())
                    throw std::invalid_argument("Invalid block size: 1024, 256, 64, 16");

                blockSize = sqrts[blockSize];
                break;
            case 'a':
                algorithm = atoi(optarg);
                if(algorithm == 0 || (algorithm != 1 && algorithm != 2 && algorithm != 3))
                    throw std::invalid_argument("Inserire 1 per FW su CPU, 2 per FW parallelizzato su global memory, 3 per FW parallelizzato su shared memory (blocked)");
                break;
            case 'c':
                toVerify = true;
                break;
            case 'v':
                vectorize = true;
                break;
            case 'V':
                printResults = true;
                break;
            default:
                throw std::invalid_argument("Utilizzo comando: ./parallel_fw [-s] seed [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-V] [-v] numVertices");
        }
    }

    if(vectorize && (numVertices & 3))
        throw std::invalid_argument("Il numero di vertici deve essere multiplo di 4 per poter utilizzare la versione vectorized");

    ll numCol = numVertices;
    if(algorithm == 3){
        const int remainder = numVertices - blockSize * (numVertices / blockSize);
        if (remainder)
            numCol = numVertices + blockSize - remainder;

        graph = blockedGraphInit(numVertices, perc, blockSize, seed);
    }
    else
        graph = graphInit(numVertices, perc, seed);

    //! ------------ PARALLEL FLOYD WARSHALL ON GPU -----

    short *w_GPU = nullptr;
    switch (algorithm){
        case 1:
            w_GPU = FloydWarshallCPU(graph, numVertices, numCol);
            break;
        case 2:
            w_GPU = simple_parallel_FW(graph, numCol, blockSize, vectorize);
            break;
        case 3:
            w_GPU = blocked_parallel_FW(graph, numCol, blockSize, vectorize);
            break;
    }

    //! ----------------------------------------------

    //! ------------------ VERIFY --------------------
    
    if(toVerify){
        short *resultsForVerify = FloydWarshallCPU(graph, numVertices, numCol);
        
        verify(resultsForVerify, numVertices, w_GPU, numCol);
        delete[] resultsForVerify;
    }

    if (printResults){
        std::cout << "Originale: " << std::endl;
        printMatrix(graph, numVertices, numCol);
        std::cout << "Risultato: " << std::endl;
        printMatrix(w_GPU, numVertices, numCol);
    }

    //! -----------------------------------------------------------

    if(algorithm == 1)
        delete[] w_GPU;
    else
        cuda(cudaFreeHost(w_GPU));

    delete[] graph;
    exit(0);
}
