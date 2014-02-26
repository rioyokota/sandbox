.SUFFIXES: .cxx .cu .o

CXX = mpicxx -Wall -O3 -fopenmp -ffast-math -funroll-loops
NVCC = nvcc -O3 -use_fast_math -arch=sm_35 -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

host: host.o
	$(CXX) $?
	mpirun -np 2 ./a.out
async: async.o
	$(CXX) $?
	mpirun -np 4 ./a.out
tau: tau.cxx
	tau_cxx.sh tau.cxx
	mpirun -np 8 ./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
