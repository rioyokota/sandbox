.SUFFIXES: .cxx .f90 .o

CXX = mpicxx -ggdb3 -Wall -O3 -fopenmp -ffast-math -funroll-loops -I../include
FC = mpif90 -ggdb3 -Wall -O3 -fopenmp -ffast-math -funroll-loops -I../include

.cxx.o	:
	$(CXX) -c $? -o $@
.f90.o	:
	$(FC) -c $? -o $@

host: host.o
	$(CXX) $?
	mpirun -np 2 ./a.out
hostf: hostf.o
	$(FC) $?
	mpirun -np 2 ./a.out
async: async.o
	$(CXX) $?
	mpirun -np 4 ./a.out
split: split.o
	$(CXX) $?
	mpirun -np 7 ./a.out
bisection: bisection.o
	$(CXX) $?  -lvtkCharts -lvtkRendering -lvtkGraphics -lvtkFiltering -lvtkViews -lvtkCommon -lvtkWidgets -lvtkIO
	mpirun -np 43 ./a.out
tau: tau.cxx
	tau_cxx.sh tau.cxx
	mpirun -np 8 ./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
