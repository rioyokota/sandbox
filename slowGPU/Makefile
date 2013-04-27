NVCC	= nvcc --compiler-bindir=/usr/bin/g++-4.4 -arch sm_21 -use_fast_math --ptxas-options=-v -Xcompiler "-fopenmp -O3 -g"

all:	main.cu
	$(NVCC) $?
	./a.out

clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
