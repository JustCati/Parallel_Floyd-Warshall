#include "Graph/graph.cuh"
#include <iostream>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

int main(int argc, char **argv){
    if(argc < 2)
        err("Utilizzo comando: ./parallelFloyd ([-f] file_name | [-g] num_vertices percentage (0 < percentage < 100))");

    Graph *g;

    if (std::string(argv[1]) == "-f"){
        if(argc != 3)
            err("Utilizzo comando: ./parallelFloyd -f file_name");
        g = new Graph(argv[2]);
    }
    else if (std::string(argv[1]) == "-g"){
        if(argc != 4 || atoi(argv[3]) <= 0 || atoi(argv[3]) >= 100)
            err("Utilizzo comando: ./parallelFloyd -n num_vertices percentage (0 < percentage < 100)");
    
        g = new Graph(atoi(argv[2]));
        int p = atoi(argv[3]);

        ErdosRenyi(*g, p);
    }
    else
        err("Utilizzo comando: ./parallelFloyd ([-f] file_name | [-g] num_vertices percentage (0 < percentage < 100))");


    //! ------------ TEST FLOYD WARSHALL CPU--------------
    int *d = FloydWarshallCPU(*g);
    delete[] d;
    //! --------------------------------------------------

    delete g;
    exit(0);
}
