FC=ifort -O3 -openmp -xHost -WB
#FC=ifort -check -traceback
MAKEFLAGS += --no-print-directory

.f.o:
	$(FC) -c $<

.SUFFIXES: .o .f

SRC = main.f fmm.f kernels.f hfmm3drouts.f \
     h3dterms.f cdjseval3d.f \
     rotproj.f dfft.f d3tstrcr.f yrecursion.f\
     legeexps.f rotviarecur3.f

all: $(SRC:.f=.o)
	@$(FC) $?
	./a.out

clean:
	@rm -f *.o *.out