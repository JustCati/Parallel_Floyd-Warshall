#include <cstring>
#include <iostream>
#include <limits.h>

#include "graph.hpp"


int* graphInit(int numVertices, int p, int seed){
    int *g = new int[numVertices * numVertices];

    srand(seed);
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++){
            if(i == j){
                g[i * numVertices + j] = 0;
                continue;
            }
            int perc = rand() / (RAND_MAX / 100) + 1;
            if(perc >= p)
                g[i * numVertices + j] = rand() / (RAND_MAX >> 4) + 1;
            else
                g[i * numVertices + j] = INT_MAX >> 1;
        }
    }
    return g;
}


int* blockedGraphInit(int numVertices, int p, int blockSize, int seed){
    int num;
    int remainder = numVertices - blockSize * (numVertices / blockSize);

    if (remainder != 0)
        num = numVertices + blockSize - remainder;
    else 
        num = numVertices;
    int *g = new int[num * num];

    srand(seed);
    for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++){
            if(i < numVertices && j < numVertices){
                if (i == j){
                    g[i * num + j] = 0;
                    continue;
                }
                int perc = rand() / (RAND_MAX / 100) + 1;
                if(perc >= p){
                    g[i * num + j] = rand() / (RAND_MAX >> 4) + 1;
                }
                else
                    g[i * num + j] = INT_MAX >> 1;
            }
            else
                g[i * num + j] = INT_MAX >> 1;
        }
    }
    return g;
}


int* FloydWarshallCPU(const int* g, int numVertices){
    int *W = new int[numVertices * numVertices];
    size_t memsize = numVertices * numVertices * sizeof(int);

    std::memcpy(W, g, memsize);

    for (int k = 0; k < numVertices; k++)
        for (int i = 0; i < numVertices; i++)
            for (int j = 0; j < numVertices; j++)
                W[i * numVertices + j] = std::min(W[i * numVertices + j], W[i * numVertices + k] + W[k * numVertices + j]);
    return W;
}
