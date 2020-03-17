.SUFFIXES: .cpp .o

.cpp.o:
	g++ -c $? -o $@

scope: scope.cpp bar.o
	g++ $?

clean:
	$(RM) *.o *.out
