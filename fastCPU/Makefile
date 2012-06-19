CXX	= g++ -ggdb3 -Wall -Wextra -Wshadow -Wuninitialized -O3 -fopenmp -ffast-math -funroll-loops -fforce-addr

LFLAGS	+= -DMUTUAL
LFLAGS	+= -DCOMkernel
LFLAGS	+= -DERROR_OPT
LFLAGS	+= -DUSE_RMAX
LFLAGS	+= -DUSE_BMAX
LFLAGS	+= -DCOMcenter
LFLAGS	+= -DDUAL

a	: main.cxx
	$(CXX) $? $(LFLAGS) -lpapi
	./a.out

cleanall:
	find . -name "*.o" -o -name "*.out*" -o -name "*.dat" | xargs rm -rf
save	:
	make cleanall
	cd .. && tar zcvf fastCPU.tgz fastCPU
