CXX = mpicxx -ggdb3 -Wall -Wextra -Wshadow -Wuninitialized -O3 -fopenmp -ffast-math -funroll-loops -fforce-addr

all:	main.cxx
	$(CXX) $?
	./a.out

cleanall:
	@make -C .. cleanall
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
