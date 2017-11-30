.SUFFIXES: .cxx .cu .o

CXX = mpicxx -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I../include -I/usr/include/vtk-5.8
NVCC = nvcc -O3 -use_fast_math -arch=sm_60 -Xcompiler "-Wall -fopenmp -O3 -ffast-math -funroll-loops"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o	:
	$(NVCC) -c $? -o $@

device: device.o
	$(CXX) $? -lcudart
	./a.out

mpi_direct: mpi_direct.o
	$(CXX) $? -lcudart
	mpirun -np 2 ./a.out
