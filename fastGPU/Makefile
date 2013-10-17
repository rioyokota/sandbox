.SUFFIXES: .cu

NVCC = nvcc -g -O3 -arch=sm_35 -rdc=true -ftz=true
LFLAGS = -lcudadevrt

SRC = thrust.cu serial.cu

OBJ = $(SRC:%.cu=%.o)

serial: $(OBJ)
	$(NVCC) $^ $(LFLAGS)
	./a.out

%.o: %.cu
	$(NVCC) -c $<

clean:
	rm -f serial.o

cleanall:
	rm -f *.o *.out