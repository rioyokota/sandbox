.SUFFIXES: .cxx .cu .o

CC = gcc -Wall -fopenmp -O3 -ffast-math -funroll-loops
CXX = g++ -Wall -fopenmp -O3 -ffast-math -funroll-loops
ICC = icc -Wall -openmp -O3 -fimf-domain-exclusion=15
NVCC = nvcc -O3 -use_fast_math -arch=sm_30 -lcuda --ptxas-options=-v -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"

sse: sse.c
	$(CC) $< -lpapi
	./a.out
avx: avx.c
	$(CC) -mavx $< -lpapi
	./a.out
mic: mic.c
	$(ICC) -mmic $< -L/home/rio/.papi/mic/lib -lpapi -lpfm
	ssh mic0 LD_LIBRARY_PATH=~/.lib sandbox/direct/a.out
mic2: mic2.o
	$(CXX) $?
cuda: cuda.cu direct_sse.o direct_asm.o
	$(NVCC) $? -lpapi
	./a.out
p2p: p2p.o
	$(NVCC) $? -lpapi
	./a.out
openacc: openacc.c
	pgcc -fast -mp -acc -ta=nvidia $?
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
