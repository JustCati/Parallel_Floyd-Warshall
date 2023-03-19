#include "graphCPU.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>


GraphCPU::GraphCPU(int numVertices) {
    this->numVertices = numVertices;

    this->adjMatrix = new int[numVertices*numVertices];
    for(int i = 0; i < numVertices*numVertices; i++)
        this->adjMatrix[i] = INT32_MAX;
}

GraphCPU::~GraphCPU() {
    delete[] this->adjMatrix;
}

void GraphCPU::addEdge(int src, int dest, int weight) {
    this->adjMatrix[src * this->numVertices + dest] = weight;
}
 
int GraphCPU::getNumVertices() const {
    return this->numVertices;
}

inline const int* GraphCPU::getAdjMatrix() const {
    return this->adjMatrix;
}

void ErdosRenyiCPU(GraphCPU& g, int p) {
    int percentage = p % 100;
    for (int i = 0; i < g.numVertices; i++)
        for (int j = 0; j < g.numVertices; j++){
            if (i == j)
                continue;
            int random = rand() % 100;
            if (random >= percentage)
                g.addEdge(i, j, (rand() % 15) + 1);
        }
}

int* FloydWarshallCPU(const GraphCPU& g){
    int numVertices = g.getNumVertices();
    int *W = new int[g.getNumVertices() * g.getNumVertices()];
    std::memcpy(W, g.getAdjMatrix(), numVertices * numVertices * sizeof(int));

    for (int k = 0; k < numVertices; k++)
        for (int i = 0; i < numVertices; i++)
            for (int j = 0; j < numVertices; j++){
                if (i == j)
                    continue;
                int min = W[i * numVertices + j];
                if(W[i * numVertices + k] != INT32_MAX && W[k * numVertices + j] != INT32_MAX)
                    min = std::min(min, W[i * numVertices + k] + W[k * numVertices + j]);
                W[i * numVertices + j] = min;
            }
    return W;
}
