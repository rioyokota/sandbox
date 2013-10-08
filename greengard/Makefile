FC=ifort -O3 -fopenmp
MAKEFLAGS += --no-print-directory

.f.o:
	@$(FC) -c $<

.SUFFIXES: .o .f

OBJ = d3tstrcr.f d3mtreeplot.f prini.f prinm.f yrecursion.f legeexps.f rotviarecur3.f hkrand.f dlaran.f

LP = lfmm3dpart_driver.f lfmm3dpart.f lfmm3drouts.f \
     l3dterms.f laprouts3d.f l3dtrans.f $(OBJ)

LT = lfmm3dtria_driver.f lfmm3dtria.f lfmm3drouts.f \
     l3dterms.f laprouts3d.f l3dtrans.f \
     triahquad.f triagauc.f triasymq.f l3dtrirouts.f \
     triquadflatlib.f trilib.f $(OBJ)

HP = hfmm3dpart_driver.f hfmm3dpart.f hfmm3drouts.f \
     h3dterms.f helmrouts3d.f h3dtrans.f projections.f cdjseval3d.f \
     rotproj.f dfft.f $(OBJ)

HT = hfmm3dtria_driver.f hfmm3dtria.f hfmm3drouts.f \
     h3dterms.f helmrouts3d.f h3dtrans.f projections.f cdjseval3d.f \
     rotproj.f dfft.f \
     triahquad.f triagauc.f triasymq.f h3dtrirouts.f \
     triquadflatlib.f trilib.f \
     lfmm3dtria.f lfmm3drouts.f l3dterms.f laprouts3d.f l3dtrans.f \
     l3dtrirouts.f $(OBJ)

all:
	make clean; make lp
	make clean; make lt
	make clean; make hp
	make clean; make ht

lp: $(LP:.f=.o)
	@$(FC) $?
	./a.out

lt: $(LT:.f=.o)
	@$(FC) $?
	./a.out

hp: $(HP:.f=.o)
	@$(FC) $?
	./a.out

ht: $(HT:.f=.o)
	@$(FC) $?
	./a.out

clean:
	@rm -f *.o *.out