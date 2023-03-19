#include "graph.cuh"
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>


Graph::Graph(int numVertices) {
    this->numVertices = numVertices;

    this->adjMatrix = new int[numVertices*numVertices];
    for(int i = 0; i < numVertices*numVertices; i++)
        this->adjMatrix[i] = INT32_MAX;
}

Graph::Graph(std::string filename){
    std::ifstream in(filename);
    
    if (!in.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    in >> this->numVertices;
    this->adjMatrix = new int[this->numVertices * this->numVertices];

    while(!in.eof()){
        int src, dest, weight;
        in >> src >> dest >> weight;
        this->addEdge(src, dest, weight);
    }

    in.close();
}

Graph::~Graph() {
    delete[] this->adjMatrix;
}

void Graph::addEdge(int src, int dest, int weight) {
    this->adjMatrix[src * this->numVertices + dest] = weight;
}

int Graph::getWeight(int src, int dest) const {
    return this->adjMatrix[src * this->numVertices + dest];
}
 
int Graph::getNumVertices() const {
    return this->numVertices;
}

inline const int* Graph::getAdjMatrix() const {
    return this->adjMatrix;
}

void Graph::printGraph() const {
    for (int i = 0; i < this->numVertices; i++) {
        for (int j = 0; j < this->numVertices; j++){
            if (i == j)
                std::cout << "0\t";
            else if(this->adjMatrix[i * this->numVertices + j] == INT32_MAX)
                std::cout << "INF\t";
            else
                std::cout << this->adjMatrix[i * this->numVertices + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void Graph::printGraphToFile(std::string filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cout << "Error opening file" << std::endl;
        exit(1);
    }

    out << this->numVertices << std::endl;
    for (int i = 0; i < this->numVertices; i++)
        for (int j = 0; j < this->numVertices; j++)
            if (this->adjMatrix[i * this->numVertices + j] != 0)
                out << i << " " << j << " " << this->adjMatrix[i * this->numVertices + j] << std::endl;

    out.close();
}


void ErdosRenyi(Graph& g, int p) {
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

int* FloydWarshallCPU(const Graph& g){
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
