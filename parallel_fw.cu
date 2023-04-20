#include <iostream>
#include <string>
#include <fstream>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"

#define CPU_VERT_LIMIT 4096

/*
    -v: verify
    -pit: use pitch
    -vec: vectorized if possible
    -verbose: print results matrix
    -c: if necessary save results to file (cache)
    -b <block size>: Set block size for GPU execution on Blocked Floyd-Warshall
    -p <percentage>: Set percentage for Erdos-Renyi graph generation
    -a <algorithm>: Set algorithm to use (0: simple, 1: blocked)
*/
int main(int argc, char **argv){
    int perc = 50, blockSize = 0, algorithm = 0;
    bool usePitch = false, vectorize = false;
    bool saveToCache = false, toVerify = false, printResults = false;

    if(argc < 2 || argc > 13)
        err("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-vec] [-v] [-pit]  [-verbose]");
    
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-p") == 0){
            perc = atoi(argv[i + 1]);
            if(perc <= 0 || perc >= 100)
                err("Inserire percentuale compreso tra 0 e 100 (estremi esclusi)");
        }
        if(strcmp(argv[i], "-b") == 0)
            blockSize = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-c") == 0)
            saveToCache = true;
        if(strcmp(argv[i], "-a") == 0){
            algorithm = atoi(argv[i + 1]);
            if(algorithm == 0 || (algorithm != 1 && algorithm != 2 && algorithm != 3))
                err("Inserire 1 per FW su CPU, 2 per FW parallelizzato su global memory, 3 per FW parallelizzato su shared memory (blocked)");
        }
        if(strcmp(argv[i], "-v") == 0)
            toVerify = true;
        if(strcmp(argv[i], "-verbose") == 0)
            printResults = true;
        if(strcmp(argv[i], "-pit") == 0)
            usePitch = true;
        if(strcmp(argv[i], "-vec") == 0)
            vectorize = true;
    }

    short* graph = nullptr;
    int numVertices = atoi(argv[1]), numCol = numVertices;

    if(algorithm == 3 && blockSize != 0){
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
        short* resultsCached = nullptr;
        
        std::ifstream in;
        std::string graphFilename = "cachedResults/results_" + std::to_string(numVertices) + "_" + std::to_string(perc) + ".txt";

        in.open(graphFilename, std::ifstream::in);
        if(in.is_open()){
            resultsCached = new short[numVertices * numVertices];
            for(int i = 0; i < numVertices; i++)
                for(int j = 0; j < numVertices; j++)
                    in >> resultsCached[i * numVertices + j];
            in.close();
        }
        else if (numVertices < CPU_VERT_LIMIT)
            resultsCached = FloydWarshallCPU(graph, numVertices, numCol);
        else{
            cpuExec = false;
            resultsCached = simple_parallel_FW(graph, numCol, DEFAULT_BLOCK_SIZE, false, false, true);
        }

        if(saveToCache)
            writeToFile(resultsCached, numVertices, numVertices, graphFilename);

        verify(resultsCached, numVertices, w_GPU, numCol);

        if(cpuExec)
            delete[] resultsCached;
        else
            cuda(cudaFreeHost(resultsCached));
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
