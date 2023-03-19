CXXFLAGS = -g -O3 
CUFLAGS = -arch=native
OBJSDIR = obj

fw: fw.cu graph.o
	nvcc $(CXXFLAGS) $(CUFLAGS) -o fw fw.cu ${OBJSDIR}/graph.o

graph.o: Graph/graph.cpp Graph/graph.hpp
	nvcc $(CXXFLAGS) $(CUFLAGS) -c Graph/graph.cpp -o ${OBJSDIR}/graph.o
