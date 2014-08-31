FC=ifort -O3 -openmp -xHost -WB
#FC=ifort -check -traceback
MAKEFLAGS += --no-print-directory

.f.o:
	$(FC) -c $<

.SUFFIXES: .o .f

SRC = main.f fmm.f kernels.f translate.f h3dterms.f bessel.f d3tstrcr.f legeexps.f

all: $(SRC:.f=.o)
	$(FC) $?
	./a.out

clean:
	@rm -f *.o *.out
