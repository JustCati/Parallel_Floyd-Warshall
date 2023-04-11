#include <iostream>
#include <string>
#include <fstream>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"

#define CPU_VERT_LIMIT 4096

/*
    -v: verify
    -c: if necessary save results to file (cache)
    -b <block size>: Set block size for GPU execution on Blocked Floyd-Warshall
    -p <percentage>: Set percentage for Erdos-Renyi graph generation
    -a <algorithm>: Set algorithm to use (0: simple, 1: blocked)
*/
int main(int argc, char **argv){
    int perc = 50, blockSize = 0;
    bool algorithm = 0, saveToCache = false, toVerify = false;

    if(argc < 2 || argc > 10)
        err("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage [-b] BlockSize [-a] algorithm [-c] [-v]");
    
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
        if(strcmp(argv[i], "-a") == 0)
            if(atoi(argv[i + 1]) != 0 && atoi(argv[i + 1]) != 1)
                err("Inserire 0 per algoritmo semplice, 1 per algoritmo bloccato");
            else
                algorithm = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-v") == 0)
            toVerify = true;
    }

    short* graph = nullptr;
    int numVertices = atoi(argv[1]), numCol = numVertices;

    if(algorithm){
        int remainder = numVertices - blockSize * (numVertices / blockSize);
        if (remainder)
            numCol = numVertices + blockSize - remainder;

        graph = blockedGraphInit(numVertices, perc, blockSize);
    }
    else
        graph = graphInit(numVertices, perc);


    //! ------------ PARALLEL FLOYD WARSHALL ON GPU -----

    short* w_GPU = nullptr;
    if (algorithm)
        w_GPU = blocked_parallel_FW(graph, numCol, blockSize);
    else
        w_GPU = simple_parallel_FW(graph, numCol, blockSize);

    //! ---------------------------------------------

    //! ------------ VERIFY --------------
    if(toVerify){
        bool cpuExec = numVertices < CPU_VERT_LIMIT;
        
        std::ifstream in;
        short* resultsCached = nullptr;
        std::string graphFilename = "cachedResults/results_" + std::to_string(numVertices) + "_" + std::to_string(perc) + ".txt";

        in.open(graphFilename, std::ifstream::in);
        if(in.is_open()){
            resultsCached = new short[numVertices * numVertices];
            for(int i = 0; i < numVertices; i++)
                for(int j = 0; j < numVertices; j++)
                    in >> resultsCached[i * numVertices + j];
            in.close();
        }
        else if (cpuExec)
            resultsCached = FloydWarshallCPU(graph, numVertices, numCol);
        else
            resultsCached = simple_parallel_FW(graph, numCol, blockSize, true);

        if(saveToCache)
            writeToFile(resultsCached, numVertices, numVertices, graphFilename);

        verify(resultsCached, numVertices, w_GPU, numCol);

        if(cpuExec)
            delete[] resultsCached;
        else
            cuda(cudaFreeHost(resultsCached));
    }

    //! -----------------------------------------------------------

    delete[] graph;
    cuda(cudaFreeHost(w_GPU));
    exit(0);
}
