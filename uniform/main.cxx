#include "args.h"
#include "dataset.h"
#include "traversal.h"
#include "verify.h"
#include "serialfmm.h"

int main(int argc, char ** argv) {
  const real_t eps2 = 0.0;
  const real cycle = 10 * M_PI;

  Args args(argc, argv);
  Dataset data;
  Traversal traversal(args.nspawn, args.images, eps2);
  SerialFMM FMM;

  const int numBodies = args.numBodies;
  const int ncrit = 100;
  const int maxLevel = numBodies >= ncrit ? 1 + int(log(numBodies / ncrit)/M_LN2/3) : 0;
  const int gatherLevel = 1;
  const int numImages = args.images;

  FMM.allocate(numBodies, maxLevel, numImages);
  args.verbose &= FMM.MPIRANK == 0;
  logger::verbose = args.verbose;
  logger::printTitle("FMM Parameters");
  args.print(logger::stringLength, PP);

  logger::printTitle("FMM Profiling");
  logger::startTimer("Total FMM");
  logger::startTimer("Partition");
  FMM.partitioner(gatherLevel);
  logger::stopTimer("Partition");

  for( int it=0; it<1; it++ ) {
    int ix[3] = {0, 0, 0};
    FMM.R0 = 0.5 * cycle / FMM.numPartition[FMM.maxGlobLevel][0];
    for_3d FMM.RGlob[d] = FMM.R0 * FMM.numPartition[FMM.maxGlobLevel][d];
    FMM.getGlobIndex(ix,FMM.MPIRANK,FMM.maxGlobLevel);
    for_3d FMM.X0[d] = 2 * FMM.R0 * (ix[d] + .5);
    srand48(FMM.MPIRANK);
    real average = 0;
    for( int i=0; i<numBodies; i++ ) {
      FMM.Jbodies[i][0] = 2 * FMM.R0 * (drand48() + ix[0]);
      FMM.Jbodies[i][1] = 2 * FMM.R0 * (drand48() + ix[1]);
      FMM.Jbodies[i][2] = 2 * FMM.R0 * (drand48() + ix[2]);
      FMM.Jbodies[i][3] = (drand48() - .5) / numBodies;
      average += FMM.Jbodies[i][3];
    }
    average /= numBodies;
    for( int i=0; i<numBodies; i++ ) {
      FMM.Jbodies[i][3] -= average;
    }
  
    logger::startTimer("Grow tree");
    FMM.sortBodies();
    FMM.buildTree();
    logger::stopTimer("Grow tree");
  
    logger::startTimer("Upward pass");
    FMM.upwardPass();
    logger::stopTimer("Upward pass");
  
    FMM.periodicM2L();

    FMM.downwardPass();
    logger::stopTimer("Total FMM", 0);

    Bodies bodies(numBodies);
    B_iter B = bodies.begin();
    for (int b=0; b<numBodies; b++, B++) {
      for_3d B->X[d] = FMM.Jbodies[b][d];
      B->SRC = FMM.Jbodies[b][3];
      for_4d B->TRG[d] = FMM.Ibodies[b][d];
    }
    Bodies jbodies = bodies;
    logger::startTimer("Total Direct");
    const int numTargets = 100;
    data.sampleBodies(bodies, numTargets);
    Bodies bodies2 = bodies;
    data.initTarget(bodies);
    for (int i=0; i<FMM.MPISIZE; i++) {
      if (args.verbose) std::cout << "Direct loop          : " << i+1 << "/" << FMM.MPISIZE << std::endl;
      traversal.direct(bodies, jbodies, cycle);
    }
    traversal.normalize(bodies);
    logger::printTitle("Total runtime");
    logger::printTime("Total FMM");
    logger::stopTimer("Total Direct");
    logger::resetTimer("Total Direct");
    Verify verify;
    double potDif = verify.getDifScalar(bodies, bodies2);
    double potNrm = verify.getNrmScalar(bodies);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);
    logger::printTitle("FMM vs. direct");
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
  }
  FMM.deallocate();
}
