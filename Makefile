CXXFLAGS = -g -O3 
CUFLAGS = -arch=native
OBJSDIR = obj

parallelFloyd: parallelFloyd.cu graphCPU.o
	nvcc $(CXXFLAGS) $(CUFLAGS) -o parallelFloyd parallelFloyd.cu ${OBJSDIR}/graphCPU.o

graphCPU.o: Graph/graphCPU.cpp Graph/graphCPU.hpp
	nvcc $(CXXFLAGS) $(CUFLAGS) -c Graph/graphCPU.cpp -o ${OBJSDIR}/graphCPU.o
