#pragma once
#include <iostream>
#include <fstream>
#include <limits.h>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

void printMatrix(const int* matrix, int num, int numVertices){
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++){
            if(matrix[i * num + j] == INT_MAX >> 1)
                std::cout << "INF" << "\t";
            else
                std::cout << matrix[i * num + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void verify(const int* w_CPU, int numCPU, const int* w_GPU, int numGPU){
    for(int i = 0; i < numCPU; i++)
        for(int j = 0; j < numCPU; j++)
            if(w_CPU[i * numCPU + j] != w_GPU[i * numGPU + j])
                std::cerr << "Errore all'indice '" << i*numCPU + j << "' : " << w_CPU[i * numCPU + j] << " != " << w_GPU[i * numGPU+ j] << std::endl;
}


void writeToFile(const int* matrix, int num, std::string filename){
    std::ofstream out(filename, std::ofstream::binary);
    for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++)
            out << matrix[i * num + j] << " ";
        out << std::endl;
    }
    out.close();
}

