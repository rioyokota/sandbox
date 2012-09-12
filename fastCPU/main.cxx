#include <papi.h>
#include "serialfmm.h"

int main() {
  double tic, toc;
  int numBodies = 1000;
  THETA = 0.6;
  Bodies bodies, bodies2;
  Cells cells;
  SerialFMM FMM;
  numBodies = 1000000;
  std::cout << "N                    : " << numBodies << std::endl;
  bodies.resize(numBodies);
  srand48(0);
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {      // Loop over bodies
    for( int d=0; d!=3; ++d ) {
      B->X[d] = drand48() * 2 * M_PI - M_PI;
    }
    B->SRC = 1. / bodies.size();
    B->TRG = 0;
  }

  int Events[3] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM };
  int EventSet = PAPI_NULL;
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  PAPI_start(EventSet);

  tic = FMM.getTime();
  FMM.bottomup(bodies,cells);
  FMM.evaluate(cells,cells);
  toc = FMM.getTime();
  if( FMM.printNow ) printf("FMM                  : %lf\n",toc-tic);

  long long values[4];
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;

  bodies2 = bodies;
  bodies2.resize(100);
  tic = FMM.getTime();
  FMM.direct(bodies2,bodies);
  toc = FMM.getTime();
  if( FMM.printNow ) printf("Direct               : %lf\n",toc-tic);
}
