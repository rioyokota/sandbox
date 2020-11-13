.SUFFIXES: .cpp .f90 .o
  
CXX = icc
F90 = ifort

.cpp.o  :
	$(CXX) -c $? -o $@

.f90.o  :
	$(F90) -c $? -o $@

main: main.o sub.o
	$(CXX) $?
	./a.out
clean:
	$(RM) *.o *.out
