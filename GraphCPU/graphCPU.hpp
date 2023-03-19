#pragma once
#include <vector>
#include <string>

class GraphCPU {
private:
    int numVertices;
    
    size_t memsize;

public:

int* adjMatrix;
    GraphCPU(int numVertices, int p);

    ~GraphCPU();

    void addEdge(int src, int dest, int weight);
     
    int getNumVertices() const;

    size_t getMatrixSize() const;

    const int* getAdjMatrix() const;
};

int* FloydWarshallCPU(const GraphCPU& g);
