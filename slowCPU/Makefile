CXX     = g++ -ggdb3 -Wall -Wextra -Wshadow -Wuninitialized -O3 -fopenmp -ffast-math -funroll-loops -fforce-addr

a	: main.cxx
	$(CXX) $?
	./a.out

cleanall:
	find . -name "*.o" -o -name "*.out*" -o -name "*.dat" | xargs rm -rf
save    :
	make cleanall
	cd .. && tar zcvf slow.tgz slow
