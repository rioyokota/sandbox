#include <cassert>

#include "boundbox.h"
#include "buildtree.h"
#include "logger.h"
#include "traversal.h"
#include "updownpass.h"

int main(int argc, char ** argv) {
  Logger logger;

  const int numBodies = 1000000;
  const int images = 0;
  const int ncrit = 8;
  const int nspawn = 1000;
  const real_t theta = 0.4;
  const real_t eps2 = 0.0;
  const real_t cycle = 2 * M_PI;
  BoundBox boundbox(nspawn);
  BuildTree tree(ncrit,nspawn);
  UpDownPass pass(theta,eps2);
  Traversal traversal(nspawn,images,eps2);
  logger.printTitle("FMM Profiling");
  logger.startTimer("Total FMM");
//! Initialize dsitribution, source & target value of bodies
  srand48(0);                                                 // Set seed for random number generator
  Bodies bodies(numBodies);                                   // Initialize bodies
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
    for (int d=0; d<2; d++) {                                 //  Loop over dimension
      B->X[d] = drand48() * 2 * M_PI - M_PI;                  //   Initialize positions
    }                                                         //  End loop over dimension
    B->SRC = drand48() - .5;                                  //   Initialize charge
    B->TRG = 0;                                               //  Clear target values
  }                                                           // End loop over bodies
  Bounds bounds = boundbox.getBounds(bodies);
  Cells cells = tree.buildTree(bodies, bounds);
  pass.upwardPass(cells);
  traversal.dualTreeTraversal(cells, cells, cycle);
  Bodies jbodies = bodies;
  pass.downwardPass(cells);
  logger.printTitle("Total runtime");
  logger.stopTimer("Total FMM");
//! Downsize target bodies by even sampling 
  int numTargets = 10;                                          // Number of target bodies
  int stride = bodies.size() / numTargets;                      // Stride of sampling
  for (int i=0; i<numTargets; i++) {                            // Loop over target samples
    bodies[i] = bodies[i*stride];                               //  Sample targets
  }                                                             // End loop over target samples
  bodies.resize(numTargets);                                    // Resize bodies to target size
  Bodies bodies2 = bodies;                                      // Save bodies in bodies2
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over bodies
    B->TRG = 0;                                                 //  Clear target values
  }                                                             // End loop over bodies
  logger.startTimer("Total Direct");
  traversal.direct(bodies, jbodies, cycle);
  traversal.normalize(bodies);
  logger.stopTimer("Total Direct");
  double diff1 = 0, norm1 = 0;
//! Evaluate relaitve L2 norm error
  B_iter B2 = bodies2.begin();                                // Set iterator for bodies2
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
    double dp = (B->TRG - B2->TRG) * (B->TRG - B2->TRG);      // Difference of potential
    double  p = B2->TRG * B2->TRG;                            //  Value of potential
    diff1 += dp;                                              //  Accumulate difference of potential
    norm1 += p;                                               //  Accumulate value of potential
  }                                                           // End loop over bodies & bodies2
  logger.printTitle("FMM vs. direct");
  logger.printError(diff1, norm1);
  return 0;
}
