.SUFFIXES: .cxx .cu .o

CXX = mpicxx -Wall -O3 -fopenmp
NVCC = nvcc -O3 -use_fast_math -arch=sm_60 -Xcompiler "-Wall -fopenmp -O3"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o	:
	$(NVCC) -c $? -o $@

device: device.o
	$(CXX) $? -lcudart

gemmEx: gemmEx.o
	$(NVCC) $? -lcublas

mpi_direct: mpi_direct.o
	$(CXX) $? -lcudart

clean:
	$(RM) *.o *.out
