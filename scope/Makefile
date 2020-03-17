.SUFFIXES: .cpp .o

.cpp.o:
	g++ -c $? -o $@

main: main.cpp scope.o
	g++ $?

clean:
	$(RM) *.o *.out
