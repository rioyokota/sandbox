.SUFFIXES: .cxx .cu .o

CC = gcc -Wall -O3 -fopenmp -mavx -ffast-math -funroll-loops
CXX = g++ -Wall -O3 -fopenmp -mavx -ffast-math -funroll-loops

.c.o	:
	$(CC) -c $? -o $@
.cxx.o	:
	$(CXX) -c $? -o $@

platform: platform.o
	$(CXX) $? -lOpenCL
	./a.out
vadd: vadd.o
	$(CXX) $? -lOpenCL
	./a.out
vmul: vmul.o
	$(CXX) $? -lOpenCL
	./a.out

clean:
	@find . -name "*.o" -o -name "*.out*" -o -name "*.mod" | xargs rm -rf
