#pragma once
#include <stdio.h>


class Graph {
private:
    int numVertices;
    int* adjMatrix;
    size_t memsize;
    int blockSize, numOversized;

public:

    Graph(int numVertices, int p = 50, int blockSize = 0, int seed = 1234);

    ~Graph();
     
    int getNumVertices() const;

    size_t getMatrixMemSize() const;

    int getBlockSize() const;

    const int* getAdjMatrix() const;
};

int* FloydWarshallCPU(const Graph& g);
