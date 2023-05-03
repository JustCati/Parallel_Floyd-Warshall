CXXFLAGS = -g -O3
CUFLAGS = -arch=native

OBJSDIR = obj
CUDADIR = Cuda
GRAPHDIR = Graph

parallel_fw: parallel_fw.cu graph.o cuda.o
	@nvcc $(CXXFLAGS) $(CUFLAGS) -o parallel_fw parallel_fw.cu ${OBJSDIR}/*.o

graph.o: ${GRAPHDIR}/graph.cpp ${GRAPHDIR}/graph.hpp
	@nvcc $(CXXFLAGS) $(CUFLAGS) -c ${GRAPHDIR}/graph.cpp -o ${OBJSDIR}/graph.o

cuda.o: ${CUDADIR}/CudaFunctions.cu ${CUDADIR}/CudaFunctions.cuh
	@nvcc $(CXXFLAGS) $(CUFLAGS) -c ${CUDADIR}/CudaFunctions.cu -o ${OBJSDIR}/cuda.o

clean:
	rm -f fw ${OBJSDIR}/*.o