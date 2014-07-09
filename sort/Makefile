.SUFFIXES: .cxx .o

#CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I../include
CXX = icpc -Wall -xHOST -O3 -openmp -funroll-loops -finline-functions -ansi-alias -I../include

.cxx.o	:
	$(CXX) -c $? -o $@

binary: binary.o
	$(CXX) $? $(VFLAGS)
	./a.out
neighbor: neighbor.o
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
