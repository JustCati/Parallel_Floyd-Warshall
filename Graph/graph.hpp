#pragma once
#include <stdio.h>

#define DEFAULT_BLOCK_SIZE 16
#define ll long long


short* graphInit(int numVertices, int p, int seed = 1234);

short* blockedGraphInit(int numVertices, int p, int blockSize = DEFAULT_BLOCK_SIZE, int seed = 1234);

short* FloydWarshallCPU(const short* g, ll numVertices, ll numCol);
