#pragma once
#include <stdio.h>


class Graph {
private:
    int numVertices;
    int* adjMatrix;
    size_t memsize;

public:

    Graph(int numVertices, int p, int seed = 1234);

    ~Graph();
     
    int getNumVertices() const;

    size_t getMatrixSize() const;

    const int* getAdjMatrix() const;
};

int* FloydWarshallCPU(const Graph& g);
