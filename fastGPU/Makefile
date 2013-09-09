.SUFFIXES: .cu

NVCC = nvcc -g -O3 -arch=sm_35 -rdc=true -ftz=true
LFLAGS = -lcudadevrt

SRC = treecode.cu buildTree.cu computeMultipoles.cu groupTargets.cu computeForces.cu

OBJ = $(SRC:%.cu=%.o)

all: $(OBJ)
	$(NVCC) $^ $(LFLAGS)
	./a.out

%.o: %.cu
	$(NVCC) -c $<

clean:
	/bin/rm -rf *.o *.out

$(OBJ): Treecode.h cuda_primitives.h cudamem.h