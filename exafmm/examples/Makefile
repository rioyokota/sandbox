include ../Makefile.include

LFLAGS	+= -DMASS # Use all positive sources
#LFLAGS	+= -DIneJ # Different target and source points
LFLAGS	+= -DDIRECT # Compare with direct summation
#LFLAGS	+= -DWRITE_TIME # Write timings to file

# Test for serial FMM
serial: serial.o $(OBJECTS)
	$(CXX) $? $(LFLAGS)
	./a.out
