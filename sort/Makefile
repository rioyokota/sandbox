.SUFFIXES: .cxx .c .o

CC = gcc -ggdb3 -Wall -O3
CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I../include
#CXX = icpc -Wall -xHOST -O3 -openmp -funroll-loops -finline-functions -ansi-alias -I../include

.c.o	:
	$(CC) -c $? -o $@
.cxx.o	:
	$(CXX) -c $? -o $@
avlsort: avlsort.o
	$(CC) $?
	./a.out
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
	OMP_NUM_THREADS=16 ./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
