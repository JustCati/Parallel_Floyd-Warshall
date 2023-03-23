#include <iostream>
#include <string>
#include <fstream>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"


/*
    -c : Enable CPU execution if results are not found in cache
    -b : Set block size for GPU execution on Blocked Floyd-Warshall
    -p : Set percentage for Erdos-Renyi graph generation
*/
int main(int argc, char **argv){
    if(argc < 2 || argc > 7)
        err("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage (0 < percentage < 100) [-b] BlockSize [-c]");

    bool cpu = false;
    int blockSize = 0, p = 50;
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-p") == 0){
            p = atoi(argv[i + 1]);
            if(p <= 0 || p >= 100)
                err("Inserire percentuale compreso tra 0 e 100 (estremi esclusi)");
        }
        if(strcmp(argv[i], "-b") == 0)
            blockSize = atoi(argv[i + 1]);
        if(strcmp(argv[i], "-c") == 0)
            cpu = true;
    }

    Graph* g = new Graph(atoi(argv[1 + cpu]), p, blockSize);
    std::string graphFilename = "cachedResults/results_" + std::to_string(g->getNumVertices()) + "_" + std::to_string(p) + ".txt";


    //! ------------ CPU EXECUTION AND CACHING RESULTS --------------
    
    int *w_CPU;
    std::ifstream in(graphFilename, std::ifstream::in);

    if(in.is_open()){
        w_CPU = new int[g->getNumVertices() * g->getNumVertices()];
        for(int i = 0; i < g->getNumVertices(); i++)
            for(int j = 0; j < g->getNumVertices(); j++)
                in >> w_CPU[i * g->getNumVertices() + j];
        in.close();
    }
    else if(cpu){
        w_CPU = FloydWarshallCPU(*g);
        writeToFile(w_CPU, g->getNumVertices(), graphFilename);
    }

    //! -----------------------------------------------------------


    //! ------------ SIMPLE FLOYD WARSHALL GPU -----

    int* w_GPU = simple_parallel_FW(*g);

    //! ---------------------------------------------

    //? -------------- VERIFY ------------------

    if(cpu)
        verify(w_CPU, w_GPU, g->getNumVertices());

    //? -----------------------------------------


    delete g;
    delete[] w_CPU;
    cuda(cudaFreeHost(w_GPU));
    
    exit(0);
}
