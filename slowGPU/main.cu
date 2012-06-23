#include "serialfmm.h"

int main() {
  double tic, toc;
  int numBodies = 1000;
  THETA = 0.6;
  Bodies bodies, bodies2;
  Cells cells;
  SerialFMM FMM;
  for( int it=0; it<25; ++it ) {
    numBodies = int(pow(10,(it+24)/8.0));
    std::cout << "N                    : " << numBodies << std::endl;
    bodies.resize(numBodies);
    srand48(0);
    for( Body *B=&*bodies.begin(); B<&*bodies.end(); ++B ) {      // Loop over bodies
      for( int d=0; d<3; ++d ) {
       B->X[d] = drand48();
      }
      B->SRC = 1. / bodies.size();
      B->TRG = 0;
    }

    tic = FMM.getTime();
    FMM.bottomup(bodies,cells);
    FMM.evaluate(cells,cells);
    toc = FMM.getTime();
    if( FMM.printNow ) printf("FMM                  : %lf\n",toc-tic);

    bodies2 = bodies;
    bodies2.resize(100);
    tic = FMM.getTime();
    FMM.direct(bodies2,bodies);
    toc = FMM.getTime();
    if( FMM.printNow ) printf("Direct               : %lf\n",toc-tic);
  }
}
