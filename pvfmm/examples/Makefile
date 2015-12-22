all:
	mpic++ -mavx -fabi-version=6 -std=gnu++11 -fopenmp -DALLTOALLV_FIX -I/home/yokotar/fmm_codes/pvfmm/include/pvfmm -I./include -c src/fmm_pts.cpp -o obj/fmm_pts.o
	mpic++ -mavx -fabi-version=6 -std=gnu++11 -fopenmp -DALLTOALLV_FIX -I/home/yokotar/fmm_codes/pvfmm/include/pvfmm obj/fmm_pts.o -L/home/yokotar/fmm_codes/pvfmm/lib/pvfmm -lpvfmm -lfftw3  -lfftw3f -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread  -ldl -lstdc++ -limf -lgfortran -lm -lquadmath   -ldl -lstdc++ -limf -lm  -o bin/fmm_pts
