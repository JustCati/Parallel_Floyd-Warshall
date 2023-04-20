#pragma once
#include <iostream>
#include <fstream>
#include <limits.h>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

void printMatrix(const short* matrix, int numVertices, int numCol){
    std::cout << std::endl;
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++){
            if(matrix[i * numCol + j] == SHRT_MAX >> 1)
                std::cout << "-" << "\t";
            else
                std::cout << matrix[i * numCol + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void verify(const short* w_CPU, int numColCPU, const short* w_GPU, int numColGPU){
    std::cout << std::endl <<  "Verifica..." << std::endl;
    for(int i = 0; i < numColCPU; i++)
        for(int j = 0; j < numColCPU; j++)
            if(w_CPU[i * numColCPU + j] != w_GPU[i * numColGPU + j])
                std::cerr << "Errore all'indice '" << i * numColCPU + j << "' : " << \
                w_CPU[i * numColCPU + j] << " != " << w_GPU[i * numColGPU + j] << std::endl;
    std::cout << "Verifica completata!" << std::endl;
}


void writeToFile(const short* matrix, int numVertices, int numCol, std::string filename){
    std::ofstream out(filename, std::ofstream::binary);
    for(int i = 0; i < numVertices; i++){
        for(int j = 0; j < numVertices; j++)
            out << matrix[i * numCol + j] << " ";
        out << std::endl;
    }
    out.close();
}
