.SUFFIXES: .cxx .cu .o

CXX = mpicxx -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I../include -I/usr/include/vtk-5.8
NVCC = nvcc -O3 -use_fast_math -arch=sm_35 -Xcompiler "-Wall -fopenmp -O3 -ffast-math -funroll-loops"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o	:
	$(NVCC) -c $? -o $@

cilk: cilk.cxx
	icpc $? -lcilkrts
	./a.out 1
	./a.out 4
device: device.o
	$(CXX) $? -lcudart
	./a.out
hello: hello.asm
	nasm $? -f elf
	ld -m elf_i386 -s hello.o
	./a.out
nitadori: nitadori.o
	$(CXX) $?
	./a.out
opencl: opencl.o
	$(CXX) $? -lOpenCL
	./a.out
pthreads: pthreads.o
	$(CXX) $?
	./a.out
template: template.o
	$(CXX) $?
	./a.out
tsp: tsp.o
	$(CXX) $?
	mpirun -np 2 ./a.out
vtk: vtk.o
	$(CXX) $? -lvtkCharts -lvtkRendering -lvtkGraphics -lvtkFiltering -lvtkViews -lvtkCommon -lvtkWidgets -lvtkIO
	./a.out
