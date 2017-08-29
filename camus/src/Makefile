# Makefile
FC = gfortran
CC = gcc
#for debug
FFLAGS = -O2 -fopenmp -cpp
CFLAGS = -O2

TARGET = ../bin/camus
OBJS = v5.o sub.o tea.o read_file.o
.SUFFIXES:.c .o
.SUFFIXES:.f90 .o

$(TARGET): $(OBJS)
	$(FC) -o $(TARGET) $(OBJS) $(FFLAGS)

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.f90.o:
	$(FC) -c $(FFLAGS) $< -o $@

sub.o: v5.o tea.o
sub.o: v5.o tea.o
read_file.o: v5.o

test:
	make clean ; make  ; cd ../input_gen/ ; ./$(TARGET) zz_test.inp_5k

clean:
	rm *.o *.mod $(TARGET)
	$(RM) $(TARGET) $(OBJS)  $(OBJS:.o=.mod)
	$(RM) core gmon.out *~ #*#
