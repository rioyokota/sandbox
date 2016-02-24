.SUFFIXES: .cxx .o

CXX	= g++ -O3 -fopenmp -ffast-math

.cxx.o  :
	$(CXX) -c $^ -o $@ $(LFLAGS)

test: test.o
	$(CXX) $? $(LFLAGS)
	./a.out

clean:
	$(RM) *.o *.out
