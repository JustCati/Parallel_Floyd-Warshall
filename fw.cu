#include <iostream>
#include <string>
#include <fstream>

#include "utils.hpp"
#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"



int main(int argc, char **argv){
    
    if(argc < 2)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");
    if(argc != 3 || atoi(argv[2]) <= 0 || atoi(argv[2]) >= 100)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");

    int p = atoi(argv[2]);
    Graph* g = new Graph(atoi(argv[1]), p);
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
    else{
        w_CPU = FloydWarshallCPU(*g);
        writeToFile(w_CPU, g->getNumVertices(), graphFilename);
    }

    //! -----------------------------------------------------------


    //! ------------ SIMPLE FLOYD WARSHALL GPU -----

    int* w_GPU = simple_parallel_FW(*g);

    //! ---------------------------------------------

    //? -------------- VERIFY ------------------

        verify(w_CPU, w_GPU, g->getNumVertices());

    //? -----------------------------------------


    delete g;
    delete[] w_CPU;
    cuda(cudaFreeHost(w_GPU));
    
    exit(0);
}
