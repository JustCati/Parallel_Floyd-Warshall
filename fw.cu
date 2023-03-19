#include <iostream>
#include <string>
#include <fstream>

#include "Graph/graph.hpp"
#include "Cuda/CudaFunctions.cuh"


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

void writeToFile(const int* matrix, int num, std::string filename){
    std::ofstream out(filename);

    for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++)
            matrix[i * num + j] == INT_MAX >> 1 ? out << "INF" << " " : out << matrix[i * num + j] << "\t";
        out << std::endl;
    }

    out.close();
}

int main(int argc, char **argv){
    if(argc < 2)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");
    if(argc != 3 || atoi(argv[2]) <= 0 || atoi(argv[2]) >= 100)
        err("Utilizzo comando: ./fw num_vertices percentage (0 < percentage < 100)");

    int p = atoi(argv[2]);
    Graph* g = new Graph(atoi(argv[1]), p);

    writeToFile(g->getAdjMatrix(), g->getNumVertices(), "graph.txt");

    //! ------------ FLOYD WARSHALL CPU--------------

    int *d_CPU = FloydWarshallCPU(*g);
    writeToFile(d_CPU, g->getNumVertices(), "fw_cpu.txt");
    delete[] d_CPU;

    //! --------------------------------------------------


    //! ------------ SIMPLE FLOYD WARSHALL GPU --------------

    int* d_GPU = simple_parallel_FW(*g);
    writeToFile(d_GPU, g->getNumVertices(), "fw_gpu_simple.txt");
    cuda(cudaFreeHost(d_GPU));

    //! ----------------------------------------------------------

    delete g;
    exit(0);
}
