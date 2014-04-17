.SUFFIXES: .cxx .o

CXX = mpicxx -Wall -O3 -fopenmp -msse4a -ffast-math -funroll-loops -I../include -I/usr/include/vtk-5.8

.cxx.o	:
	$(CXX) -c $? -o $@

device: device.o
	$(CXX) $? -lcudart
	./a.out
hello: hello.asm
	nasm $? -f elf
	ld -m elf_i386 -s hello.o
	./a.out
nitadori: nitadori.o
	$(CXX) $?
	./a.out
pthreads: pthreads.o
	$(CXX) $?
	./a.out
template: template.o
	$(CXX) $?
	./a.out
vtk: vtk.o
	$(CXX) $? -lvtkCharts -lvtkRendering -lvtkGraphics -lvtkFiltering -lvtkViews -lvtkCommon -lvtkWidgets -lvtkIO
	./a.out
clean:
	@make -C .. clean
commit:
	@make -C .. commit
save:
	@make -C .. save
revert:
	@make -C .. revert
