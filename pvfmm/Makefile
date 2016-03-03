.SUFFIXES: .cpp

#CXX = mpic++ -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
CXX = mpicxx -msse4 -std=c++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -llapack -lblas

%.o: %.cpp
	$(CXX) -c $< -o $@

float: src/fmm_pts.cpp
	$(CXX) -c $< -o src/fmm_pts.o -DFLOAT

double: src/fmm_pts.cpp
	$(CXX) -c $< -o src/fmm_pts.o

link: src/mortonid.o src/profile.o src/fmm_pts.o
	$(CXX) $? $(LDFLAGS) -o ./fmm_pts

clean:
	rm -f src/*.o ./fmm_pts

tags:
	find . -name "*.cpp" -o -name "*.hpp" -o -name "*.txx" | xargs etags -f TAGS

m4:
	./fmm_pts -omp 4 -N 10000 -m 4 -M 64

m16:
	./fmm_pts -omp 4 -N 10000 -m 16 -M 320

y4:
	./fmm_pts -omp 16 -N 1000000 -m 4 -M 64

y16:
	./fmm_pts -omp 16 -N 1000000 -m 16 -M 320

r4:
	./fmm_pts -omp 72 -N 1000000 -m 4 -M 64

r16:
	./fmm_pts -omp 72 -N 1000000 -m 16 -M 320
