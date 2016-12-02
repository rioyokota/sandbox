.SUFFIXES: .cxx .o

CXX = g++ -O3 -fopenmp -ffast-math

.cxx.o  :
	$(CXX) -c $^ -o $@

test: test.o
	$(CXX) $?
	./a.out

clean:
	$(RM) *.o *.out
