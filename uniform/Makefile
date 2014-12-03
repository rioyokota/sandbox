CXX	= mpicxx -ggdb3 -Wall -Wextra -O3 -msse4a -ffast-math -funroll-loops -fopenmp
MAKEFLAGS += --no-print-directory

all	:
	make clean
	make compile
	make run

compile	: main.cxx
	$(CXX) $? -DSerial

run	:
	./a.out

clean:
	find . -name "*.o" -o -name "*.out*" -o -name "*.mod" | xargs rm -rf
save:
	make clean
	cd .. && tar zcvf uniform.tgz uniform
