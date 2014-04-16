.SUFFIXES: .cxx .cu .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops
NVCC = nvcc -O3 -use_fast_math -arch=sm_35 -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

binary: binary.o
	$(CXX) $? $(VFLAGS)
	./a.out
neighbor: neighbor.o
	$(CXX) $?
	./a.out
radixbody: radixbody.o
	$(CXX) $?
	./a.out
radixsort: radixsort.o
	$(CXX) $?
	./a.out
sort: sort.o
	$(CXX) $?
	./a.out
tree: tree.o
	$(CXX) $?
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
