#pragma once
#include <vector>
#include <string>

class Graph {
private:
    int numVertices;
    int* adjMatrix;
    size_t memsize;

public:

    Graph(int numVertices, int p);

    ~Graph();
     
    int getNumVertices() const;

    size_t getMatrixSize() const;

    const int* getAdjMatrix() const;
};

int* FloydWarshallCPU(const Graph& g);
