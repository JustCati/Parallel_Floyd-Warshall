#include "Graph/graph.cuh"
#include <iostream>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

int main(int argc, char **argv){
    if(argc < 2)
        err("Utilizzo comando: ./parallelFloyd ([-f] file_name | [-n] num_vertices)");

    Graph *g;

    if (std::string(argv[1]) == "-f"){
        if(argc != 3)
            err("Utilizzo comando: ./parallelFloyd -f file_name");
        g = new Graph(argv[2]);
    }
    else if (std::string(argv[1]) == "-n"){
        if(argc != 3)
            err("Utilizzo comando: ./parallelFloyd -n num_vertices");
        g = new Graph(atoi(argv[2]));
    }
    else
        err("Utilizzo comando: ./parallelFloyd ([-f] file_name | [-n] num_vertices)");



    //! ----------------------------
    double p = 0.5;
    ErdosRenyi(*g, p);
    g->printGraph();

    delete g;
    exit(0);
}
