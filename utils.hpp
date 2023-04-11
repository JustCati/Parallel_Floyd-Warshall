#pragma once
#include <iostream>
#include <fstream>
#include <limits.h>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

void printMatrix(const int* matrix, int numVertices, int numCol){
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++){
            if(matrix[i * numCol + j] == INT_MAX >> 1)
                std::cout << "INF" << "\t";
            else
                std::cout << matrix[i * numCol + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void verify(const int* w_CPU, int numColCPU, const int* w_GPU, int numColGPU){
    for(int i = 0; i < numColCPU; i++)
        for(int j = 0; j < numColCPU; j++)
            if(w_CPU[i * numColCPU + j] != w_GPU[i * numColGPU + j])
                std::cerr << "Errore all'indice '" << i * numColCPU + j << "' : " << \
                w_CPU[i * numColCPU + j] << " != " << w_GPU[i * numColGPU + j] << std::endl;
}


void writeToFile(const int* matrix, int numVertices, int numCol, std::string filename){
    std::ofstream out(filename, std::ofstream::binary);
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++)
            out << matrix[i * numCol + j] << " ";
        out << std::endl;
    }
    out.close();
}
