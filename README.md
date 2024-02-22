# Parallel Floyd Warshall Algorithm



## Description
This project implements a parallelized solution of the Floyd-Warshall algorithm using CUDA technology for execution on GPU. The Floyd-Warshall algorithm is widely employed for finding the shortest path between all nodes in a weighted graph, and it is of particular interest in the realm of routing problems and network optimization.

The proposed solutions consist of three approaches:

1. Simple parallelization of the outermost loop of the algorithm and its vectorized version with type short4.
2. Optimized version of the previous implementation [1] through the utilization of sub-blocks to increase the responsibility of individual threads and reduce the number of accesses to global memory.
3. Blocked Floyd-Warshall: an optimized version of the Floyd-Warshall algorithm that leverages blocking technique to decrease the number of accesses to global memory and efficiently utilize the GPU cache (shared memory). Additionally, a vectorized version with type short4 is implemented.
All results have been tested on randomly generated graphs generated on-the-fly using the Erdos-Renyi algorithm.


## Screenshots
<img src="./images/final_results.png" width=75% height=75%>


## License
This project is licensed under the [GNU General Public License v3.0](LICENSE)

