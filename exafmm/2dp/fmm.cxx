#include "build_tree.h"
#include "kernel.h"
#include "timer.h"
#include "traversal.h"
using namespace exafmm;

int main(int argc, char ** argv) {                              // Main function
  const int numBodies = 10000;                                  // Number of bodies
  const real_t cycle = 2 * M_PI;                                // Cycle of periodic boundary condition
  P = 10;                                                       // Order of expansions
  ncrit = 8;                                                    // Number of bodies per leaf cell
  theta = 0.4;                                                  // Multipole acceptance criterion
  images = 3;                                                   // 3^images * 3^images * 3^images periodic images

  //! Initialize dsitribution, source & target value of bodies
  printf("--- FMM Profiling ----------------\n");               // Start profiling
  start("Initialize bodies");                                   // Start timer
  srand48(0);                                                   // Set seed for random number generator
  Bodies bodies(numBodies);                                     // Initialize bodies
  real_t average = 0;                                           // Average charge
  for (int b=0; b<int(bodies.size()); b++) {                    // Loop over bodies
    for (int d=0; d<2; d++) {                                   //  Loop over dimension
      bodies[b].X[d] = drand48() * 2 * M_PI - M_PI;             //   Initialize positions
    }                                                           //  End loop over dimension
    bodies[b].q = drand48() - .5;                               //  Initialize charge
    average += bodies[b].q;                                     //  Accumulate charge
    bodies[b].p = 0;                                            //  Clear potential
    for (int d=0; d<2; d++) bodies[b].F[d] = 0;                 //  Clear force
  }                                                             // End loop over bodies
  average /= bodies.size();                                     // Average charge
  for (int b=0; b<int(bodies.size()); b++) {                    // Loop over bodies
    bodies[b].q -= average;                                     // Charge neutral
  }                                                             // End loop over bodies
  stop("Initialize bodies");                                    // Stop timer

  //! Build tree
  start("Build tree");                                          // Start timer
  Cell * cells = buildTree(bodies);                             // Build tree
  stop("Build tree");                                           // Stop timer

  //! FMM evaluation
  start("Upward pass");                                         // Start timer
  upwardPass(cells);                                            // Upward pass for P2M, M2M
  stop("Upward pass");                                          // Stop timer
  start("Traversal");                                           // Start timer
  traversal(cells, cells, cycle);                               // Traversal for M2L, P2P
  stop("Traversal");                                            // Stop timer
  start("Downward pass");                                       // Start timer
  downwardPass(cells);                                          // Downward pass for L2L, L2P
  stop("Downward pass");                                        // Stop timer

  // Direct N-Body
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
    for (int d=0; d<2; d++) bodies[b].F[d] = 0;                 //  Clear force
  }                                                             // End loop over bodies
  direct(bodies, jbodies, cycle);                               // Direct N-Body
  stop("Direct N-Body");                                        // Stop timer

  //! Evaluate relaitve L2 norm error
  double pDif = 0, pNrm = 0, FDif = 0, FNrm = 0;
  for (int b=0; b<int(bodies.size()); b++) {                    // Loop over bodies & bodies2
    pDif += (bodies[b].p - bodies2[b].p) * (bodies[b].p - bodies2[b].p);// Difference of potential
    pNrm += bodies2[b].p * bodies2[b].p;                        //  Value of potential
    FDif += (bodies[b].F[0] - bodies2[b].F[0]) * (bodies[b].F[0] - bodies2[b].F[0])// Difference of force
      + (bodies[b].F[0] - bodies2[b].F[0]) * (bodies[b].F[0] - bodies2[b].F[0]);// Difference of force
    FNrm += bodies2[b].F[0] * bodies2[b].F[0] + bodies2[b].F[1] * bodies2[b].F[1];//  Value of force
  }                                                             // End loop over bodies & bodies2
  printf("--- FMM vs. direct ---------------\n");               // Print message
  printf("Rel. L2 Error (p)  : %e\n",sqrtf(pDif/pNrm));         // Print potential error
  printf("Rel. L2 Error (F)  : %e\n",sqrtf(FDif/FNrm));         // Print force error
  return 0;
}
