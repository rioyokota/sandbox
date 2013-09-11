.SUFFIXES: .cu

NVCC = nvcc -g -O3 -arch=sm_35 -rdc=true -ftz=true
LFLAGS = -lcudadevrt

SRC = thrust.cu main.cu

OBJ = $(SRC:%.cu=%.o)

all: $(OBJ)
	$(NVCC) $^ $(LFLAGS)
	./a.out

%.o: %.cu
	$(NVCC) -c $<

clean:
	/bin/rm -rf *.o *.out