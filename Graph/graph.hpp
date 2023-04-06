#pragma once
#include <stdio.h>

#define DEFAULT_BLOCK_SIZE 32

class Graph {
private:
    int numVertices;
    int* adjMatrix;
    size_t memsize;
    int blockSize, numOversized;

public:

    Graph(int numVertices, int p = 50, bool gpu = false, int blockSize = DEFAULT_BLOCK_SIZE, int seed = 1234);

    ~Graph();
     
    int getNumVertices() const;

    size_t getMatrixMemSize() const;

    int getBlockSize() const;

    const int* getAdjMatrix() const;
};

int* FloydWarshallCPU(const Graph& g);
