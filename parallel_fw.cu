#include <iostream>
#include <string>
#include <fstream>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"

/*
    -b <block size>: Set block size for GPU execution on Blocked Floyd-Warshall
    -p <percentage>: Set percentage for Erdos-Renyi graph generation
*/
int main(int argc, char **argv){
    int perc = 50, blockSize = 0;

    if(argc < 2 || argc > 6)
        err("Utilizzo comando: ./parallel_fw num_vertices [-p] percentage [-b] BlockSize");
    
    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-p") == 0){
            perc = atoi(argv[i + 1]);
            if(perc <= 0 || perc >= 100)
                err("Inserire percentuale compreso tra 0 e 100 (estremi esclusi)");
        }
        if(strcmp(argv[i], "-b") == 0)
            blockSize = atoi(argv[i + 1]);
    }

    Graph* g = new Graph(atoi(argv[1]), perc, (blockSize != 0));
    std::string graphFilename = "cachedResults/results_" + std::to_string(g->getNumVertices()) + "_" + std::to_string(perc) + ".txt";

    //! ------------ SIMPLE FLOYD WARSHALL GPU -----

    int* w_GPU = simple_parallel_FW(*g);

    //! ---------------------------------------------

    //! ------------ CACHED RESULTS READING/WRITING --------------
    
    int *resultsCached = nullptr;
    std::ifstream in(graphFilename, std::ifstream::in);

    if(in.is_open()){
        resultsCached = new int[g->getNumVertices() * g->getNumVertices()];
        for(int i = 0; i < g->getNumVertices(); i++)
            for(int j = 0; j < g->getNumVertices(); j++)
                in >> resultsCached[i * g->getNumVertices() + j];
        in.close();
    }
    else
        resultsCached = simple_parallel_FW(*g, true);

    writeToFile(resultsCached, g->getNumVertices(), graphFilename);

    //! -----------------------------------------------------------

    //? -------------- VERIFY ------------------
        
    verify(resultsCached, w_GPU, g->getNumVertices());

    //? -----------------------------------------

    delete g;
    delete[] resultsCached;
    cuda(cudaFreeHost(w_GPU));
    
    exit(0);
}
