.SUFFIXES: .cxx .cu .o

CXX = mpicxx -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops
NVCC = nvcc -O3 -use_fast_math -arch=sm_35 -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

cpu1d: cpu1d.o
	$(CXX) $?
	./a.out
cpu2d: cpu2d.o
	$(CXX) $?
	./a.out
cpu3d: cpu3d.o
	$(CXX) $?
	./a.out
kmeans: kmeans.o
	$(CXX) $?
	./a.out
matern: matern.o
	$(CXX) $?
	./a.out
12steps: 12steps.o
	$(NVCC) $?
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
