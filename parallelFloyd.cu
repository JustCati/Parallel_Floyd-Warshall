#include "Graph/graphCPU.hpp"
#include <iostream>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

int main(int argc, char **argv){
    if(argc < 2)
        err("Utilizzo comando: ./parallelFloyd num_vertices percentage (0 < percentage < 100)");
    if(argc != 4 || atoi(argv[3]) <= 0 || atoi(argv[3]) >= 100)
        err("Utilizzo comando: ./parallelFloyd -n num_vertices percentage (0 < percentage < 100)");

    int p = atoi(argv[3]);

    GraphCPU *g = new GraphCPU(atoi(argv[2]));
    ErdosRenyiCPU(*g, p);

    //! ------------ TEST FLOYD WARSHALL CPU--------------
    int *d = FloydWarshallCPU(*g);
    delete[] d;
    //! --------------------------------------------------

    delete g;
    exit(0);
}
