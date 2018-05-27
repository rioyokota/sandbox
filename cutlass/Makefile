sgemm : gemm.cu
	nvcc -DTEST_SGEMM -arch=sm_60 gemm.cu -O3 -Xptxas -v -std=c++11 -I. -lcublas
	./a.out

clean :
	rm -f *.o *.out
