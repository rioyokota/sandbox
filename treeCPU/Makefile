CXX = mpicxx -ggdb3 -Wall -Wextra -Wshadow -Wuninitialized -O3 -fopenmp -ffast-math -funroll-loops -fforce-addr

a	: main.cxx
	$(CXX) $?
	./a.out

clean:
	find . -name "*.o" -o -name "*.out*" | xargs rm -rf
save	:
	make clean
	cd .. && tar zcvf tree.tgz tree
