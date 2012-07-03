#include "serialfmm.h"

int main() {
  double tic, toc;
  int numBodies = 1000;
  SerialFMM FMM(1000000);
  for( int it=0; it<25; ++it ) {
    numBodies = int(pow(10,(it+24)/8.0));
    std::cout << "N                    : " << numBodies << std::endl;
    FMM.dataset(numBodies);

    tic = FMM.getTime();
    FMM.bottomup();
    FMM.evaluate();
    toc = FMM.getTime();
    if( FMM.printNow ) printf("FMM                  : %lf\n",toc-tic);

    tic = FMM.getTime();
    FMM.direct();
    toc = FMM.getTime();
    if( FMM.printNow ) printf("Direct               : %lf\n",toc-tic);
  }
}
