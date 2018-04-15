.SUFFIXES: .c .cu .o

CC = gcc -Wall -fopenmp -O3 -ffast-math -funroll-loops
CXX = g++ -Wall -fopenmp -O3 -ffast-math -funroll-loops
ICC = icc -Wall -qopenmp -O3 -fimf-domain-exclusion=15
NVCC = nvcc -O3 -use_fast_math -arch=sm_61 -Xcompiler "-Wall -fopenmp -O3 -ffast-math -funroll-loops"

sse: sse.c
	$(CXX) $<
	./a.out
avx: avx.c
	$(CXX) -mavx $<
	./a.out
knc: knc.c
	$(ICC) -mmic $<
	ssh $(HOSTNAME)-mic0 LD_LIBRARY_PATH=/opt/intel/compiler/latest/lib/mic direct/a.out
knl: knl.c
	$(ICC) -xMIC-AVX512 $<
	./a.out
mpi_avx: mpi_avx.cxx
	mpiicpc $? -qopenmp -xCORE-AVX2 -o mpi_avx
mpi_knl: mpi_knl.cxx
	mpiicpc $? -qopenmp -xMIC-AVX512 -o mpi_knl
cuda: cuda.cu
	$(NVCC) $<
	./a.out
openacc: openacc.c
	pgcc -fast -mp -acc -ta=nvidia $<
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
