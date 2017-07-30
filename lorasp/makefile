include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
OBJS = lorasp.o tree.o edge.o redNode.o blackNode.o superNode.o node.o params.o
EIGEN_DIR = $(HOME)/eigen
SCOTCH_DIR = $(HOME)/scotch_6.0.4
CINCLUDES = -I$(EIGEN_DIR) -I$(SCOTCH_DIR)/include
LFLAGS = -L$(SCOTCH_DIR)/lib
CFLAGS = -O3 -std=c++0x
CPP = g++
LIBS = -lscotch -lscotcherr -lm -lpthread

lorasp: $(OBJS)
	$(CPP) $(CFLAGS) $(CINCLUDES) -o lorasp.out $(OBJS) $(LFLAGS) $(LIBS)
	./lorasp.out params.in

lorasp.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c lorasp.cpp

tree.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c tree.cpp

edge.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c edge.cpp

superNode.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c superNode.cpp

blackNode.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c blackNode.cpp

redNode.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c redNode.cpp

node.o:
	$(CPP) $(CFLAGS) $(CINCLUDES) -c node.cpp

params.o:
	$(CPP) $(CFLAGS) -c params.cpp

petsc: petsc.o chkopts
	${CLINKER} -o petsc.out petsc.o ${PETSC_KSP_LIB}
	./petsc.out -ksp_monitor

clean::
	rm -rf *.o *.out *.mtx *.txt
