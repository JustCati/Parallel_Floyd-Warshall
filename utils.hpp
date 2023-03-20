#pragma once
#include <iostream>
#include <fstream>
#include <limits.h>


void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
}

void printMatrix(const int* matrix, int num){
    for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++){
            if(matrix[i * num + j] == INT_MAX >> 1)
                std::cout << "INF" << "\t";
            else
                std::cout << matrix[i * num + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void verify(const int* w_CPU, const int* w_GPU, int num){
    for(int i = 0; i < num; i++)
        for(int j = 0; j < num; j++)
            if(w_CPU[i * num + j] != w_GPU[i * num + j]){
                std::cerr << "Errore: " << w_CPU[i * num + j] << " != " << w_GPU[i * num + j] << std::endl;
                exit(1);
            }
    std::cout << "Verifica completata con successo" << std::endl;
}


void writeToFile(const int* matrix, int num, std::string filename){
    std::ofstream out(filename, std::ofstream::out);
    for(int i = 0; i < num; i++){
        for(int j = 0; j < num; j++)
            out << matrix[i * num + j] << " ";
        out << std::endl;
    }
    out.close();
}

