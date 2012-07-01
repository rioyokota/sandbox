NVCC	= nvcc -arch sm_21 -use_fast_math -Iinclude -Xcompiler "-fopenmp -O3"

all:	main.cu
	$(NVCC) $?
	./a.out

cleanall:
	@make -C .. cleanall
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
