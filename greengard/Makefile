FC=ifort -O3 -fopenmp
MAKEFLAGS += --no-print-directory

.f.o:
	$(FC) -c $<

.SUFFIXES: .o .f

SRC = main.f fmm.f hfmm3drouts.f \
     h3dterms.f helmrouts3d.f h3dtrans.f projections.f cdjseval3d.f \
     rotproj.f dfft.f d3tstrcr.f d3mtreeplot.f prini.f prinm.f yrecursion.f\
     legeexps.f rotviarecur3.f

all: $(SRC:.f=.o)
	@$(FC) $?
	./a.out

clean:
	@rm -f *.o *.out