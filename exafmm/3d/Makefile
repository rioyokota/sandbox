.SUFFIXES: .cxx .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -mavx -ffast-math -funroll-loops -DEXAFMM_LAPLACE

.cxx.o  :
	$(CXX) -c $? -o $@

all:
	@make run_kernel
	@make run_fmm

kernel: kernel.o
	$(CXX) $? -o $@

run_kernel: kernel
	./kernel 10
	./kernel 20
	./kernel 30

fmm: fmm.o
	$(CXX) $? -o $@

run_fmm: fmm
	./fmm

clean:
	$(RM) ./*.o ./kernel ./fmm
