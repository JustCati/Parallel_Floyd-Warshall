#pragma once
#include <iostream>
#include <fstream>



void err(const char *msg){
    std::cout << msg << std::endl;
    exit(1);
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

