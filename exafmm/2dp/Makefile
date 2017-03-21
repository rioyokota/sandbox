.SUFFIXES: .cxx .o

CXX = g++ -g -Wall -O3 -fopenmp

.cxx.o  :
	$(CXX) -c $^ -o $@

fmm: fmm.o
	$(CXX) $? -o $@
	./fmm

clean:
	$(RM) ./*.o ./fmm
