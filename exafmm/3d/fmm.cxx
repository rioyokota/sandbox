#include "build_tree.h"
#include "kernel.h"
#include "timer.h"
#include "traversal.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 1000;
  P = 10;
  ncrit = 64;
  theta = 0.4;

  printf("--- %-16s ------------\n", "FMM Profiling");
  //! Initialize bodies
  start("Initialize bodies");
  Bodies bodies(numBodies);
  real_t average = 0;
  srand48(0);
  for (int b=0; b<int(bodies.size()); b++) {
    for (int d=0; d<3; d++) {
      bodies[b].X[d] = drand48() * 2 * M_PI - M_PI;
    }
  }
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].q = drand48() - .5;
    average += bodies[b].q;
  }
  average /= bodies.size();
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].q -= average;
    bodies[b].p = 0;
    for (int d=0; d<3; d++) bodies[b].F[d] = 0;
  }
  stop("Initialize bodies");

  //! Build tree
  start("Build tree");
  Cell * cells = buildTree(bodies);
  stop("Build tree");

  //! FMM evaluation
  start("Upward pass");
  initKernel();
  upwardPass(cells);
  stop("Upward pass");
  start("Traversal");
  traversal(cells, cells);
  stop("Traversal");
  start("Downward pass");
  downwardPass(cells);
  stop("Downward pass");

  //! Direct N-Body
  start("Direct N-Body");                                       // Start timer
  const int numTargets = 10;                                    // Number of targets for checking answer
  Bodies jbodies = bodies;                                      // Save bodies in jbodies
  int stride = bodies.size() / numTargets;                      // Stride of sampling
  for (int b=0; b<numTargets; b++) {                            // Loop over target samples
    bodies[b] = bodies[b*stride];                               //  Sample targets
  }                                                             // End loop over target samples
  bodies.resize(numTargets);                                    // Resize bodies
  Bodies bodies2 = bodies;                                      // Backup bodies
  for (int b=0; b<int(bodies.size()); b++) {                    // Loop over bodies
    bodies[b].p = 0;                                            //  Clear potential
    for (int d=0; d<3; d++) bodies[b].F[d] = 0;                 //  Clear force
  }                                                             // End loop over bodies
  direct(bodies, jbodies);                                      // Direct N-Body
  stop("Direct N-Body");                                        // Stop timer

  //! Verify result
  real_t pSum = 0, pSum2 = 0, FDif = 0, FNrm = 0;
  for (int b=0; b<int(bodies.size()); b++) {
    pSum += bodies[b].p * bodies[b].q;
    pSum2 += bodies2[b].p * bodies2[b].q;
    FDif += (bodies[b].F[0] - bodies2[b].F[0]) * (bodies[b].F[0] - bodies2[b].F[0]) +
      (bodies[b].F[1] - bodies2[b].F[1]) * (bodies[b].F[1] - bodies2[b].F[1]) +
      (bodies[b].F[2] - bodies2[b].F[2]) * (bodies[b].F[2] - bodies2[b].F[2]);
    FNrm += bodies[b].F[0] * bodies[b].F[0] + bodies[b].F[1] * bodies[b].F[1] +
      bodies[b].F[2] * bodies[b].F[2];
  }
  real_t pDif = (pSum - pSum2) * (pSum - pSum2);
  real_t pNrm = pSum * pSum;
  real_t pRel = std::sqrt(pDif/pNrm);
  real_t FRel = std::sqrt(FDif/FNrm);
  printf("--- %-16s ------------\n", "FMM vs. Ewald");
  printf("%-20s : %8.5e s\n","Rel. L2 Error (p)", pRel);
  printf("%-20s : %8.5e s\n","Rel. L2 Error (F)", FRel);
  return 0;
}
