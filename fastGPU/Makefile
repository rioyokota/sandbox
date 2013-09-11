.SUFFIXES: .cu

NVCC = nvcc -g -O3 -arch=sm_35 -rdc=true -ftz=true
LFLAGS = -lcudadevrt

all: main.o
	$(NVCC) $^ $(LFLAGS)
	./a.out

%.o: %.cu
	$(NVCC) -c $<

clean:
	/bin/rm -rf *.o *.out

$(OBJ): Treecode.h cuda_primitives.h cudamem.h plummer.h buildtree.h upwardpass.h grouptargets.h traversal.h