#include <iostream>
#include <limits.h>

#include "graph.hpp"


short* graphInit(int numVertices, int p, int seed){
    short* g = new short[numVertices * numVertices];

    srand(seed);
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++){
            if(i == j){
                g[i * numVertices + j] = 0;
                continue;
            }
            short perc = rand() / (RAND_MAX / 100) + 1;
            if(perc >= p)
                g[i * numVertices + j] = rand() / (RAND_MAX >> 4) + 1;
            else
                g[i * numVertices + j] = SHRT_MAX >> 1;
        }
    }
    return g;
}


short* blockedGraphInit(int numVertices, int p, int blockSize, int seed){
    int numCol;
    int remainder = numVertices - blockSize * (numVertices / blockSize);

    if (remainder != 0)
        numCol = numVertices + blockSize - remainder;
    else 
        numCol = numVertices;
    short* g = new short[numCol * numCol];

    srand(seed);
    for(int i = 0; i < numCol; i++){
        for(int j = 0; j < numCol; j++){
            if(i < numVertices && j < numVertices){
                if (i == j){
                    g[i * numCol + j] = 0;
                    continue;
                }
                short perc = rand() / (RAND_MAX / 100) + 1;
                if(perc >= p)
                    g[i * numCol + j] = rand() / (RAND_MAX >> 4) + 1;
                else
                    g[i * numCol + j] = SHRT_MAX >> 1;
            }
            else
                g[i * numCol + j] = SHRT_MAX >> 1;
        }
    }
    return g;
}


short* FloydWarshallCPU(const short* g, int numVertices, int numCol){
    short* W = new short[numVertices * numVertices];

    for(int i = 0; i < numVertices; i++)
        for(int j = 0; j < numVertices; j++)
            W[i * numVertices + j] = g[i * numCol + j];

    for (int k = 0; k < numVertices; k++)
        for (int i = 0; i < numVertices; i++)
            for (int j = 0; j < numVertices; j++)
                W[i * numVertices + j] = std::min(W[i * numVertices + j], (short)(W[i * numVertices + k] + W[k * numVertices + j]));
    
    return W;
}
