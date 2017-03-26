#include "build_tree.h"
#include "kernel.h"
#include "timer.h"
#include "traversal.h"
using namespace exafmm;

int main(int argc, char ** argv) {                              // Main function
  const int numBodies = 10000;                                  // Number of bodies
  const int numTargets = 10;                                    // Number of targets for checking answer
  const int ncrit = 8;                                          // Number of bodies per leaf cell
  P = 10;                                                       // Order of expansions
  theta = 0.4;                                                  // Multipole acceptance criterion

  //! Initialize distribution, source & target value of bodies
  printf("--- FMM Profiling ----------------\n");               // Start profiling
  start("Initialize bodies");                                   // Start timer
  srand48(0);                                                   // Set seed for random number generator
  Bodies bodies(numBodies);                                     // Initialize bodies
  real_t average = 0;                                           // Average charge
  for (int b=0; b<int(bodies.size()); b++) {                             // Loop over bodies
    for (int d=0; d<2; d++) {                                   //  Loop over dimension
      bodies[b].X[d] = drand48() * 2 * M_PI - M_PI;             //   Initialize positions
    }                                                           //  End loop over dimension
    bodies[b].q = drand48() - .5;                               //  Initialize charge
    average += bodies[b].q;                                     //  Accumulate charge
    bodies[b].p = 0;                                            //  Clear potential
    for (int d=0; d<2; d++) bodies[b].F[d] = 0;                 //  Clear force
  }                                                             // End loop over bodies
  average /= bodies.size();                                         // Average charge
  for (int b=0; b<int(bodies.size()); b++) {                             // Loop over bodies
    bodies[b].q -= average;                                     // Charge neutral
  }                                                             // End loop over bodies
  stop("Initialize bodies");                                    // Stop timer

  // ! Get Xmin and Xmax of domain
  start("Build tree");                                          // Start timer
  real_t R0;                                                    // Radius of root cell
  real_t Xmin[2], Xmax[2], X0[2];                               // Min, max of domain, and center of root cell
  for (int d=0; d<2; d++) Xmin[d] = Xmax[d] = bodies[0].X[d];   // Initialize Xmin, Xmax
  for (int b=0; b<int(bodies.size()); b++) {                             // Loop over range of bodies
    for (int d=0; d<2; d++) Xmin[d] = fmin(bodies[b].X[d], Xmin[d]);//  Update Xmin
    for (int d=0; d<2; d++) Xmax[d] = fmax(bodies[b].X[d], Xmax[d]);//  Update Xmax
  }                                                             // End loop over range of bodies
  for (int d=0; d<2; d++) X0[d] = (Xmax[d] + Xmin[d]) / 2;      // Calculate center of domain
  R0 = 0;                                                       // Initialize localRadius
  for (int d=0; d<2; d++) {                                     // Loop over dimensions
    R0 = fmax(X0[d] - Xmin[d], R0);                             //  Calculate min distance from center
    R0 = fmax(Xmax[d] - X0[d], R0);                             //  Calculate max distance from center
  }                                                             // End loop over dimensions
  R0 *= 1.00001;                                                // Add some leeway to radius

  //! Build tree structure
  Bodies buffer = bodies;                                       // Copy bodies to buffer
  Cell * cells = new Cell();                                    // Allocate root cell
  buildTree(&bodies[0], &buffer[0], 0, bodies.size(), cells, X0, R0, ncrit); // Build tree recursively
  stop("Build tree");                                           // Stop timer

  //! FMM evaluation
  start("Upward pass");                                         // Start timer
  upwardPass(cells);                                            // Upward pass for P2M, M2M
  stop("Upward pass");                                          // Stop timer
  start("Traversal");                                           // Start timer
  traversal(cells, cells);                                      // Traversal for M2L, P2P
  stop("Traversal");                                            // Stop timer
  start("Downward pass");                                       // Start timer
  downwardPass(cells);                                          // Downward pass for L2L, L2P
  stop("Downward pass");                                        // Stop timer

  //! Direct N-Body
  start("Direct N-Body");                                       // Start timer
  Bodies jbodies = bodies;                                      // Save bodies in jbodies
  int stride = bodies.size() / numTargets;                          // Stride of sampling
  for (int b=0; b<numTargets; b++) {                            // Loop over target samples
    bodies[b] = bodies[b*stride];                               //  Sample targets
  }                                                             // End loop over target samples
  bodies.resize(numTargets);                                    // Resize bodies
  Bodies bodies2 = bodies;                                      // Backup bodies
  for (int b=0; b<int(bodies.size()); b++) {                    // Loop over bodies
    bodies[b].p = 0;                                            //  Clear potential
    for (int d=0; d<2; d++) bodies[b].F[d] = 0;                 //  Clear force
  }                                                             // End loop over bodies
  direct(bodies, jbodies);                                      // Direct N-Body
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
