CXX	= g++ -ggdb3 -Wall -Wextra -Wshadow -Wuninitialized -O3 -msse4.2 -fopenmp -ffast-math -funroll-loops -fforce-addr

LFLAGS	+= -DCOMkernel
#LFLAGS	+= -DERROR_OPT
LFLAGS	+= -DUSE_RMAX
LFLAGS	+= -DUSE_BMAX
LFLAGS	+= -DCOMcenter
LFLAGS	+= -DDUAL

a	: main.cxx
	$(CXX) $? $(LFLAGS) -lpapi
	./a.out

cleanall:
	@make -C .. cleanall
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
