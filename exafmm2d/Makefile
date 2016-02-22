.SUFFIXES: .cxx .cu .o

### choose C++ compiler
CXX	= mpicxx -ggdb3 -msse4a -O3 -fopenmp -ffast-math -funroll-loops # GCC

LFLAGS	+= -DEXPANSION=6 # Specifcy expansion order
LFLAGS	+= -DUSE_SIMD # Use SIMD intrinsics

### Debugging flags
LFLAGS	+= -DASSERT # Turns on asserttions (otherwise define an empty macro function)

### Intel TBB flags : TBB is available from http://threadingbuildingblocks.org/download.php
LFLAGS	+= -std=c++0x -DTBB -ltbb

### MassiveThreads flags : MassiveThreads is available from http://code.google.com/p/massivethreads/
#LFLAGS	+= -std=c++0x -DMTHREAD -lmyth

### Trace flags
#LFLAGS	+= -DTRACE

MAKEFLAGS += --no-print-directory

.cxx.o  :
	$(CXX) -c $^ -o $@ $(LFLAGS)
.cu.o   :
	$(NVCC) -c $^ -o $@ $(LFLAGS)

test: test.o
	$(CXX) $? $(LFLAGS)
	./a.out

help:
	@make -C .. help
clean:
	@make -C .. clean
cleandat:
	@make -C .. cleandat
cleanlib:
	@make -C .. cleanlib
cleanall:
	@make -C .. cleanall
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
