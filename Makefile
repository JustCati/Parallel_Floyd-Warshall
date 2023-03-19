CXXFLAGS = -g -O3 
CUFLAGS = -arch=native

OBJSDIR = obj
CUDADIR = Cuda
GRAPHDIR = Graph

fw: fw.cu graph.o cuda.o kernels.o
	nvcc $(CXXFLAGS) $(CUFLAGS) -o fw fw.cu ${OBJSDIR}/*.o

graph.o: ${GRAPHDIR}/graph.cpp ${GRAPHDIR}/graph.hpp
	nvcc $(CXXFLAGS) $(CUFLAGS) -c ${GRAPHDIR}/graph.cpp -o ${OBJSDIR}/graph.o

cuda.o: ${CUDADIR}/CudaFunctions.cu ${CUDADIR}/CudaFunctions.cuh
	nvcc $(CXXFLAGS) $(CUFLAGS) -c ${CUDADIR}/CudaFunctions.cu -o ${OBJSDIR}/cuda.o

kernels.o: ${CUDADIR}/kernels.cu ${CUDADIR}/kernels.cuh
	nvcc $(CXXFLAGS) $(CUFLAGS) -c ${CUDADIR}/kernels.cu -o ${OBJSDIR}/kernels.o

clean:
	rm -f fw ${OBJSDIR}/*.o
	rm -f *.txt