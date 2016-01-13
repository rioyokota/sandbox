.SUFFIXES: .cpp

CXX = mpic++ -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

%.o: %.cpp
	$(CXX) -c $< -o $@

build: src/mat_utils.o src/mem_mgr.o src/mortonid.o src/profile.o src/tree_node.o src/fmm_pts.o
	$(CXX) $? $(LDFLAGS) -o ./fmm_pts

clean:
	rm -f src/*.o ./fmm_pts

tags:
	find . -name "*.cpp" -o -name "*.hpp" -o -name "*.txx" | xargs etags -f TAGS

y4:
	./fmm_pts -omp 16 -N 1000000 -m 4 -M 64 -sp 1

y16:
	./fmm_pts -omp 16 -N 1000000 -m 16 -M 320

r4:
	./fmm_pts -omp 72 -N 1000000 -m 4 -M 32 -sp 1

r16:
	./fmm_pts -omp 72 -N 1000000 -m 16 -M 320
