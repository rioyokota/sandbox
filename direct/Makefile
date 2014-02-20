.SUFFIXES: .cxx .cu .o

CC = gcc -Wall -fopenmp -O3 -ffast-math -funroll-loops
CXX = g++ -Wall -fopenmp -O3 -ffast-math -funroll-loops
#CXX = icc -Wall -openmp -fast
NVCC = nvcc -O3 -use_fast_math -arch=sm_30 -lcuda --ptxas-options=-v -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"

.c.o	:
	$(CC) -c $? -o $@
.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

sse: sse.o
	$(CC) $? -lpapi -lm
	./a.out
avx: avx.o
	$(CC) $? -lpapi -lm
	./a.out
mic: mic.o direct.o direct_mic.o
	$(CXX) $? -L/home/rio/.papi/mic/lib -lpapi -lpfm
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
