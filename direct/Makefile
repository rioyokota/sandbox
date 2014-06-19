.SUFFIXES: .c .cu .o

CC = gcc -Wall -fopenmp -O3 -ffast-math -funroll-loops
CXX = g++ -Wall -fopenmp -O3 -ffast-math -funroll-loops
ICC = icc -Wall -openmp -O3 -fimf-domain-exclusion=15
NVCC = nvcc -O3 -use_fast_math -arch=sm_35 -Xcompiler "-Wall -fopenmp -O3 -ffast-math -funroll-loops"

sse: sse.c
	$(CC) $< -lpapi
	./a.out
avx: avx.c
	$(CC) -mavx $< -lpapi
	./a.out
mic: mic.c
	$(ICC) -mmic $< -L/home/rio/.papi/mic/lib -lpapi -lpfm
	ssh mic1 LD_LIBRARY_PATH=~/.lib sandbox/direct/a.out
mic2: mic2.c
	$(ICC) -mmic $<
	ssh mic1 LD_LIBRARY_PATH=~/.lib KMP_AFFINITY=compact sandbox/direct/a.out
cuda: cuda.cu
	$(NVCC) $< -lpapi
	./a.out
openacc: openacc.c
	pgcc -fast -mp -acc -ta=nvidia $< -lpapi
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
