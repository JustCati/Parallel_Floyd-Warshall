CXXFLAGS = -g -O3 
CUFLAGS = -arch=native
OBJSDIR = obj

fw: fw.cu graphCPU.o graphCUDA.o
	nvcc $(CXXFLAGS) $(CUFLAGS) -o fw fw.cu ${OBJSDIR}/graphCPU.o ${OBJSDIR}/graphCUDA.o

graphCPU.o: GraphCPU/graphCPU.cpp GraphCPU/graphCPU.hpp
	nvcc $(CXXFLAGS) $(CUFLAGS) -c GraphCPU/graphCPU.cpp -o ${OBJSDIR}/graphCPU.o

graphCUDA.o: GraphCUDA/graphCuda.cu GraphCUDA/graphCuda.cuh
	nvcc $(CXXFLAGS) $(CUFLAGS) -c GraphCUDA/graphCuda.cu -o ${OBJSDIR}/graphCUDA.o
