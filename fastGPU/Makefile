.SUFFIXES: .o .cpp .ptx .cu

CXX = g++ -O3 -g -Wall -Wstrict-aliasing=2 
NVCC = nvcc -g -O3 -arch=sm_35
LFLAGS = -lcudadevrt 

SRC = treecode.cu buildTree.cu computeMultipoles.cu makeGroups.cu computeForces.cu moveParticles.cu computeEnergies.cu

OBJ = $(SRC:%.cu=%.o)

all: $(OBJ)
	$(NVCC) $^ $(LFLAGS)
	./a.out -n 16777216 --plummer

%.o: %.cu
	$(NVCC) -rdc=true -c $<

computeForces.o: computeForces.cu
	$(NVCC) -c $< -ftz=true -rdc=true

clean:
	/bin/rm -rf *.o *.out *.a *.cu_o 

$(CUOBJ): Treecode.h Particle4.h cuda_primitives.h cudamem.h