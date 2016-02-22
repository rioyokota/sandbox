.SUFFIXES: .cxx .o

CXX	= g++ -O3 -fopenmp -ffast-math

LFLAGS	+= -DEXPANSION=6 # Specifcy expansion order

### Intel TBB flags : TBB is available from http://threadingbuildingblocks.org/download.php
LFLAGS	+= -std=c++0x -DTBB -ltbb

### MassiveThreads flags : MassiveThreads is available from http://code.google.com/p/massivethreads/
#LFLAGS	+= -std=c++0x -DMTHREAD -lmyth

.cxx.o  :
	$(CXX) -c $^ -o $@ $(LFLAGS)

test: test.o
	$(CXX) $? $(LFLAGS)
	./a.out

clean:
	$(RM) *.o *.out
