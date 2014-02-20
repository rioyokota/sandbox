.SUFFIXES: .cxx .cu .o

#CXX = mpicxx -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I/usr/include/vtk-5.8
#CXX = icc -gcc-name=gcc-4.6 -mavx -O3 -openmp -lstdc++
CXX = icpc -Wall -mmic -O3 -openmp
NVCC = nvcc -O3 -use_fast_math -arch=sm_30 -lcuda --ptxas-options=-v -Xcompiler "-Wall -fopenmp -ffast-math -funroll-loops"
VFLAGS = -lvtkCharts -lvtkRendering -lvtkGraphics -lvtkFiltering -lvtkViews -lvtkCommon -lvtkWidgets -lvtkIO

.cxx.o	:
	$(CXX) -c $? -o $@
.cu.o   :
	$(NVCC) -c $? -o $@

avx: avx.o direct_sse.o direct_avx.o
	$(CXX) $? -lpapi
	./a.out
mic: mic.o direct.o direct_mic.o
	$(CXX) $? -L/home/rio/.papi/mic/lib -lpapi -lpfm
mic2: mic2.o
	$(CXX) $?
sse: sse.cu direct_sse.o direct_asm.o
	$(NVCC) $? -lpapi
	./a.out
p2p: p2p.o
	$(NVCC) $? -lpapi
	./a.out
openacc: openacc.c
	pgcc -fast -mp -acc -ta=nvidia $?
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
