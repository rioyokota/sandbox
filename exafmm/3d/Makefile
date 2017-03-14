.SUFFIXES: .cxx .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -mavx -ffast-math -funroll-loops -DNDEBUG -DEXAFMM_LAPLACE -DEXAFMM_PMAX=10

.cxx.o  :
	$(CXX) -c $? -o $@

all:
	@make run_kernel
	@make run_fmm

kernel: kernel.o
	$(CXX) $? -o $@

run_kernel: kernel
	./kernel -P 10
	./kernel -P 20
	./kernel -P 30

fmm: fmm.o
	$(CXX) $? -o $@

run_fmm: fmm
	./fmm -aDv -c 256 -r 1 -n 10000

clean:
	$(RM) ./*.o ./kernel ./fmm
