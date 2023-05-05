#pragma once
#include <stdio.h>

#define ll long long
#define DEFAULT_SEED 1234
#define DEFAULT_BLOCK_SIZE 16


short* graphInit(int numVertices, int p, int seed = DEFAULT_SEED);

short* blockedGraphInit(int numVertices, int p, int blockSize = DEFAULT_BLOCK_SIZE, int seed = DEFAULT_SEED);

short* FloydWarshallCPU(const short* g, ll numVertices, ll numCol);
