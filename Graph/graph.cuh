#pragma once
#include <vector>
#include <string>

class Graph {
private:
    int numVertices;
    int* adjMatrix;

public:
    Graph(int numVertices);

    Graph(std::string filename);

    ~Graph();

    void addEdge(int src, int dest, int weight);

    int getWeight(int src, int dest) const;
     
    int getNumVertices() const;

    const int* getAdjMatrix() const;

    void printGraph() const;

    void printGraphToFile(std::string filename) const;

    friend void ErdosRenyi(Graph& g, double p);
};


void ErdosRenyi(Graph& g, double p);
int* FloydWarshall(const Graph& g);
