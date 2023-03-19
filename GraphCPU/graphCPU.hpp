#pragma once
#include <vector>
#include <string>

class GraphCPU {
private:
    int numVertices;
    int* adjMatrix;

public:
    GraphCPU(int numVertices);

    ~GraphCPU();

    void addEdge(int src, int dest, int weight);
     
    int getNumVertices() const;

    const int* getAdjMatrix() const;

    friend void ErdosRenyiCPU(GraphCPU& g, int p);
};


void ErdosRenyiCPU(GraphCPU& g, int p);
int* FloydWarshallCPU(const GraphCPU& g);
