#pragma once
#include <stdio.h>

#define DEFAULT_BLOCK_SIZE 32


int* graphInit(int numVertices, int p, int seed = 1234);

int* blockedGraphInit(int numVertices, int p, int blockSize = DEFAULT_BLOCK_SIZE, int seed = 1234);

int* FloydWarshallCPU(const int* g, int numVertices, int numCol);
