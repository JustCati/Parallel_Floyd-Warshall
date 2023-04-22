#include <iostream>
#include <unistd.h>
#include <fstream>
#include <map>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"

#define CPU_VERT_LIMIT 4096
#define DEFAULT_BLOCK_SIZE 16

/*
    -P: use pitch
    -c: check / verify results
    -V: (verbose) print results matrix
    -v: vectorized if possible with short4
    -b <block size>: Set block size for GPU execution on Blocked Floyd-Warshall
    -p <percentage>: Set percentage for Erdos-Renyi graph generation
    -a <algorithm>: Set algorithm to use (0: simple, 1: blocked)
*/
int main(int argc, char **argv){
    int perc = 50, blockSize = 0, algorithm = 0;
    bool usePitch = false, vectorize = false;
    bool toVerify = false, printResults = false;

    if(argc < 2 || argc > 12)
        throw std::invalid_argument("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-V] [-v] [-P]");
        
    short* graph = nullptr;
    int numVertices = atoi(argv[argc - 1]);

    int opt;
    extern char *optarg;
    std::map<short, short> sqrts = {{1024, 32}, {256, 16}, {64, 8}, {16, 4}, {4, 2}};
    while((opt = getopt(argc, argv, "p:b:a:cvVP")) != -1){
        switch(opt){
            case 'p':
                perc = atoi(optarg);
                if(perc <= 0 || perc >= 100)
                    throw std::invalid_argument("Inserire percentuale compreso tra 0 e 100 (estremi esclusi)");
                break;
            case 'b':
                blockSize = atoi(optarg);
                if(sqrts.find(blockSize) == sqrts.end())
                    throw std::invalid_argument("Invalid block size for blocked parallel FW algorithm");

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
            case 'P':
                usePitch = true;
                break;
            case 'v':
                vectorize = true;
                break;
            case 'V':
                printResults = true;
                break;
            default:
                throw std::invalid_argument("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-V] [-v] [-P]");
        }
    }

    if(vectorize && (numVertices & 3))
        throw std::invalid_argument("Il numero di vertici deve essere multiplo di 4 per poter utilizzare la versione vectorized");

    int numCol = numVertices;
    if(algorithm == 3){
        blockSize = (blockSize == 0) ? DEFAULT_BLOCK_SIZE : blockSize;
        int remainder = numVertices - blockSize * (numVertices / blockSize);
        if (remainder)
            numCol = numVertices + blockSize - remainder;

        graph = blockedGraphInit(numVertices, perc, blockSize);
    }
    else
        graph = graphInit(numVertices, perc);

    //! ------------ PARALLEL FLOYD WARSHALL ON GPU -----

    short* w_GPU = nullptr;
    switch (algorithm){
        case 1:
            w_GPU = FloydWarshallCPU(graph, numVertices, numCol);
            break;
        case 2:
            w_GPU = simple_parallel_FW(graph, numCol, blockSize, usePitch, vectorize);
            break;
        case 3:
            w_GPU = blocked_parallel_FW(graph, numCol, blockSize);
            break;
    }

    //! ----------------------------------------------

    //! ------------------ VERIFY --------------------
    if(toVerify){
        bool cpuExec = true;
        short *resultsForVerify = nullptr;
        
        if (numVertices < CPU_VERT_LIMIT)
            resultsForVerify = FloydWarshallCPU(graph, numVertices, numCol);
        else{
            cpuExec = false;
            resultsForVerify = simple_parallel_FW(graph, numCol, DEFAULT_BLOCK_SIZE, false, false, true);
        }

        verify(resultsForVerify, numVertices, w_GPU, numCol);

        if(cpuExec)
            delete[] resultsForVerify;
        else
            cuda(cudaFreeHost(resultsForVerify));
    }

    if (printResults)
        printMatrix(w_GPU, numVertices, numCol);

    //! -----------------------------------------------------------

    if(algorithm == 1)
        delete[] w_GPU;
    else
        cuda(cudaFreeHost(w_GPU));

    delete[] graph;
    exit(0);
}
