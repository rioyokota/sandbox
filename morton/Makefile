.SUFFIXES: .cxx .o

CXX = g++ -ggdb3 -Wall -O3

.cxx.o	:
	$(CXX) -c $? -o $@

morton2: morton2.o
	$(CXX) $? -o $@
	./morton2 1000000000 100000000

morton3: morton3.o
	$(CXX) $? -o $@
	./morton3 1000000 100000 10000

hilbert2: hilbert2.o
	$(CXX) $? -o $@
	./hilbert2 1000000000 100000000

hilbert3: hilbert3.o
	$(CXX) $? -o $@
	./hilbert3 1000000 100000 10000

clean:
	rm -f *.o morton2 morton3 hilbert2 hilbert3
