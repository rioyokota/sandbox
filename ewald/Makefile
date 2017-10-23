.SUFFIXES: .cxx .o

CXX = mpicxx -Wall -O3 -fopenmp -ffast-math -funroll-loops

.cxx.o	:
	$(CXX) -c $? -o $@

ewald: fmm.o
	$(CXX) $?
	./a.out
