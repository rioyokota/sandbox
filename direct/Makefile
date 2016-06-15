.SUFFIXES: .c .cu .o

CC = gcc -Wall -fopenmp -O3 -ffast-math -funroll-loops
CXX = g++ -Wall -fopenmp -O3 -ffast-math -funroll-loops
ICC = icc -Wall -qopenmp -O3 -fimf-domain-exclusion=15
NVCC = nvcc -O3 -use_fast_math -Xcompiler "-Wall -fopenmp -O3 -ffast-math -funroll-loops"

sse: sse.c
	$(ICC) $<
	./a.out
avx: avx.c
	$(ICC) -mavx $<
	./a.out
knc: knc.c
	$(ICC) -mmic $<
	ssh $(HOSTNAME)-mic0 LD_LIBRARY_PATH=/opt/intel/compiler/latest/lib/mic direct/a.out
knl: knl.c
	$(ICC) -xMIC-AVX512 $<
	./a.out
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
