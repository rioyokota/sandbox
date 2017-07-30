OBJS = LoRaSp.o tree.o edge.o redNode.o blackNode.o superNode.o node.o params.o
EIGEN_DIR = "../eigen"
SCOTCH_DIR = "../scotch_6.0.4"
CINCLUDES = -I$(EIGEN_DIR) -I$(SCOTCH_DIR)/include
LFLAGS = -L$(SCOTCH_DIR)/lib
CFLAGS = -O3 -std=c++0x
CPP = icpc
LIBS = -lscotch -lscotcherr -lm -lpthread

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

default: LoRaSp

LoRaSp: $(OBJS)
	$(CPP) $(CFLAGS) $(CINCLUDES) -o LoRaSp $(OBJS) $(LFLAGS) $(LIBS)

LoRaSp.o: LoRaSp.cpp gmres.h eyePC.h diagPC.h tree.cpp tree.h params.cpp params.h node.cpp node.h
	$(CPP) $(CFLAGS) $(CINCLUDES) -c LoRaSp.cpp

tree.o: tree.h tree.cpp node.h node.cpp params.h params.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c tree.cpp

edge.o: edge.cpp edge.h node.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c edge.cpp

superNode.o: tree.h tree.cpp node.h node.cpp superNode.h superNode.cpp params.h params.cpp edge.h edge.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c superNode.cpp

blackNode.o: tree.h tree.cpp node.h node.cpp blackNode.h blackNode.cpp params.h params.cpp edge.h edge.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c blackNode.cpp

redNode.o: tree.h tree.cpp node.h node.cpp redNode.h redNode.cpp params.h params.cpp edge.h edge.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c redNode.cpp

node.o: tree.h tree.cpp node.h node.cpp params.h params.cpp edge.h edge.cpp
	$(CPP) $(CFLAGS) $(CINCLUDES) -c node.cpp

params.o: params.h params.cpp
	$(CPP) $(CFLAGS) -c params.cpp

petsc: petsc.o chkopts
	-${CLINKER} petsc.o  ${PETSC_KSP_LIB}

cleanall:
	rm -rf *.o LoRaSp
