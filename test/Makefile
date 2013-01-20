.SUFFIXES: .cxx .cu .o

#CXX = mpicxx -Wall -O3 -mavx -fopenmp -ffast-math -funroll-loops -I/usr/include/vtk-5.8
CXX = icc -Wall -mmic -O3 -openmp
NVCC = nvcc -O3 -use_fast_math -arch=sm_21 -lcuda --ptxas-options=-v -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"
VFLAGS = -lvtkCommon -lvtkCharts

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

sort: sort.o
	$(CXX) $?
	./a.out
radixsort: radixsort.o
	$(CXX) $?
	./a.out
radixbody: radixbody.o
	$(CXX) $?
	./a.out
tree: tree.o
	$(CXX) $?
	./a.out
neighbor: neighbor.o
	$(CXX) $?
	./a.out
binary: binary.o
	$(CXX) $? $(VFLAGS)
	./a.out
template: template.o
	$(CXX) $?
	./a.out
pthreads: pthreads.o
	$(CXX) $?
	./a.out
getopt: getopt.o
	$(CXX) $?
	./a.out
avx: avx.o direct_sse.o direct_avx.o
	$(CXX) $? -lpapi
	./a.out
mic: mic.o direct.o direct_mic.o
	$(CXX) $?
hello: hello.asm
	nasm $? -f elf
	ld -m elf_i386 -s hello.o
	./a.out
mpi: mpi.o
	$(CXX) $?
	mpirun -np 2 ./a.out
sse: sse.o direct_sse.o direct_asm.o
	$(NVCC) $? -lpapi
	./a.out
tau: tau.cxx
	tau_cxx.sh tau.cxx
	mpirun -np 8 ./a.out
p2p: p2p.o
	$(NVCC) $? -lpapi
	./a.out
cpufmm: cpufmm.o
	$(NVCC) $?
	./a.out
gpufmm: gpufmm.o thrust.o
	$(NVCC) $?
	./a.out
12steps: 12steps.o
	$(NVCC) $?
	./a.out
openacc: openacc.c
	pgcc -O3 -mp -acc -ta=nvidia $?
	./a.out
cleanall:
	@make -C .. cleanall
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
