CXXFLAGS = -g -O3 
CUFLAGS = -arch=native
OBJSDIR = obj

parallelFloyd: parallelFloyd.cu graph.o
	nvcc $(CXXFLAGS) $(CUFLAGS) -o parallelFloyd parallelFloyd.cu ${OBJSDIR}/graph.o

graph.o: Graph/graph.cu Graph/graph.cuh
	nvcc $(CXXFLAGS) $(CUFLAGS) -c Graph/graph.cu -o ${OBJSDIR}/graph.o
