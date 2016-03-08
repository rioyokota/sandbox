.SUFFIXES: .cpp

CXX = mpic++ -O3 -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
#CXX = mpicxx -O3 -msse4 -std=c++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -llapack -lblas

%.o: %.cpp
	$(CXX) -c $< -o $@

float: main.cpp
	$(CXX) -c $< -o main.o -DFLOAT

double: main.cpp
	$(CXX) -c $< -o main.o

link: main.o
	$(CXX) $? $(LDFLAGS)

clean:
	rm -f *.o *.out

tags:
	find . -name "*.cpp" -o -name "*.hpp" | xargs etags -f TAGS

m4:
	./a.out -omp 4 -N 100000 -m 4 -M 64

m16:
	./a.out -omp 4 -N 100000 -m 16 -M 320

y4:
	./a.out -omp 16 -N 1000000 -m 4 -M 64

y16:
	./a.out -omp 16 -N 1000000 -m 16 -M 320

r4:
	./a.out -omp 72 -N 1000000 -m 4 -M 64

r16:
	./a.out -omp 72 -N 1000000 -m 16 -M 320
