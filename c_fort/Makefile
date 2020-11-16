.SUFFIXES: .cpp .f90 .o
  
CXX = icc
F90 = ifort

.cpp.o  :
	$(CXX) -c $? -o $@

.f90.o  :
	$(F90) -c $? -o $@

from_c: c_main.o f_sub.o
	$(CXX) $?
	./a.out
from_f: f_main.o c_sub.o
	$(F90) $? -lstdc++
	./a.out
clean:
	$(RM) *.mod *.o *.out
