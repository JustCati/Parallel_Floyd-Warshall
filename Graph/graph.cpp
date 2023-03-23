#include <cstring>
#include <iostream>
#include <limits.h>

#include "graph.hpp"



Graph::Graph(int numVertices, int p, int blockSize, int seed) {

    int numOversize = numVertices;
    if(blockSize){
        int remainder = numVertices % blockSize;
        if(remainder)
            numOversize = numVertices + blockSize - remainder;
    }

    this->numVertices = numVertices;
    this->numOversized = numOversize;
    this->blockSize = blockSize;

    this->adjMatrix = new int[this->numOversized * this->numOversized];
    this->memsize = this->numOversized * this->numOversized * sizeof(int);

    // Initialize the matrix using the Erdos-Renyi algorithm
    for(int i = 0; i < this->numVertices * this->numVertices; i++){
        if (i % (this->numVertices + 1) == 0)
            this->adjMatrix[i] = 0;
        else{
            int random = rand() % 100;
            if (random >= p)
                this->adjMatrix[i] = (rand() % 15) + 1;
            else
                // (INT_MAX / 2) to avoid overflow when summing two "INF"
                this->adjMatrix[i] = INT_MAX >> 1;
        }
    }
}

Graph::~Graph() {
    delete[] this->adjMatrix;
}
 
int Graph::getNumVertices() const {
    return this->numVertices;
}

size_t Graph::getMatrixMemSize() const {
    return this->memsize;
}
 
const int* Graph::getAdjMatrix() const {
    return this->adjMatrix;
}
 
int Graph::getBlockSize() const {
    return this->blockSize;
}


int* FloydWarshallCPU(const Graph& g){
    int numVertices = g.getNumVertices();
    int *W = new int[g.getNumVertices() * g.getNumVertices()];

    std::memcpy(W, g.getAdjMatrix(), g.getMatrixMemSize());

    for (int k = 0; k < numVertices; k++)
        for (int i = 0; i < numVertices; i++)
            for (int j = 0; j < numVertices; j++)
                W[i * numVertices + j] = std::min(W[i * numVertices + j], W[i * numVertices + k] + W[k * numVertices + j]);
    return W;
}
