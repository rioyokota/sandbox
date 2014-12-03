CXX	= mpicxx -ggdb3 -Wall -Wextra -O3 -msse4a -ffast-math -funroll-loops -fopenmp

serial	: main.cxx
	$(CXX) $? -I../include -DSerial
	./a.out -n 10000 -c 32 -i 3

parallel: main.cxx
	$(CXX) $? -I../include
	mpirun -np 8 ./a.out -n 62500 -c 32 -i 3
#	$(CXX) $? -DIJHPCA
#	mpirun -np 8 ./a.out -n 62500 -c 32 -i 1

clean:
	find . -name "*.o" -o -name "*.out*" -o -name "*.mod" | xargs rm -rf
save:
	make clean
	cd .. && tar zcvf uniform.tgz uniform
