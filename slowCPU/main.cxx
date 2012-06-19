#include "serialfmm.h"

int main() {
  double tic, toc;
  SerialFMM FMM;
  srand48(0);
  FMM.numBodies = 1000000;
  FMM.maxLevel = int(log(FMM.numBodies / 30)/M_LN2/3) + 1;
  FMM.allocate();
  if( FMM.printNow ) {
    printf("N       : %d\n",FMM.numBodies);
    printf("Levels  : %d\n",FMM.maxLevel);
    printf("------------------\n");
  }

  for( int it=0; it<1; it++ ) {
    FMM.R0 = .5;
    for_3d FMM.X0[d] = FMM.R0;
    srand48(0);
    for( int i=0; i<FMM.numBodies; i++ ) {
      FMM.Jbodies[i][0] = drand48();
      FMM.Jbodies[i][1] = drand48();
      FMM.Jbodies[i][2] = drand48();
      FMM.Jbodies[i][3] = 1. / FMM.numBodies;
    }
    double tic2 = FMM.getTime();
    tic = FMM.getTime();
    FMM.sortBodies();
    toc = FMM.getTime();
    if( FMM.printNow ) printf("Sort    : %lf\n",toc-tic);
  
    tic = FMM.getTime();
    FMM.buildTree();
    toc = FMM.getTime();
    if( FMM.printNow ) printf("Tree    : %lf\n",toc-tic);
  
    FMM.upwardPass();
  
    FMM.downwardPass();
    double toc2 = FMM.getTime();
    if( FMM.printNow ) printf("FMM     : %lf\n\n",toc2-tic2);
  
    tic = FMM.getTime();
    FMM.direct();
    toc = FMM.getTime();
    if( FMM.printNow ) printf("Direct  : %lf\n",toc-tic);
  }
  FMM.deallocate();

}
