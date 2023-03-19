#include "GraphCPU/graphCPU.hpp"
#include "GraphCUDA/graphCuda.cuh"
#include <iostream>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

int main(int argc, char **argv){
    if(argc < 2)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");
    if(argc != 3 || atoi(argv[2]) <= 0 || atoi(argv[2]) >= 100)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");

    int p = atoi(argv[2]);
    GraphCPU *g = new GraphCPU(atoi(argv[1]), p);

    for (int i = 0; i < g->getNumVertices(); i++){
        for (int j = 0; j < g->getNumVertices(); j++)
            if(g->adjMatrix[i * g->getNumVertices() + j] == INT_MAX >> 1)
                std::cout << "INF\t";
            else
                std::cout << g->adjMatrix[i * g->getNumVertices() + j] << "\t";
        std::cout << std::endl;
    }

    //! ------------ TEST FLOYD WARSHALL CPU--------------
    int *d = FloydWarshallCPU(*g);
    delete[] d;
    //! --------------------------------------------------

    delete g;
    exit(0);
}
