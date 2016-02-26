#include <cassert>
#include <cstdlib>

#include "buildtree.h"
#include "logger.h"
#include "traversal.h"
#include "updownpass.h"

int main(int argc, char ** argv) {
  const int numBodies = 10000;
  const int numTargets = 10;
  const int images = 3;
  const int ncrit = 8;
  const real_t theta = 0.4;
  const real_t eps2 = 0.0;
  const real_t cycle = 2 * M_PI;
  Traversal traversal(images,theta);
  printf("--- FMM Profiling ----------------\n");
  startTimer("Total FMM");

//! Initialize dsitribution, source & target value of bodies
  startTimer("Init bodies");                                    // Start timer
  srand48(0);                                                   // Set seed for random number generator
  Bodies bodies(numBodies);                                     // Initialize bodies
  real_t average = 0;                                           // Average charge
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over bodies
    for (int d=0; d<2; d++) {                                   //  Loop over dimension
      B->X[d] = drand48() * 2 * M_PI - M_PI;                    //   Initialize positions
    }                                                           //  End loop over dimension
    B->SRC = drand48() - .5;                                    //  Initialize charge
    average += B->SRC;                                          //  Accumulate charge
    B->TRG = 0;                                                 //  Clear target values
  }                                                             // End loop over bodies
  average /= bodies.size();                                     // Average charge
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over bodies
    B->SRC -= average;                                          // Charge neutral
  }                                                             // End loop over bodies
  stopTimer("Init bodies");                                     // Stop timer

// ! Get Xmin and Xmax of domain
  startTimer("Get bounds");                                     // Start timer
  real_t R0;                                                    // Radius of root cell
  vec2 Xmin, Xmax, X0;                                          // min, max of domain, and center of root cell
  Xmin = Xmax = bodies.front().X;                               // Initialize Xmin, Xmax
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over range of bodies
    Xmin = min(B->X, Xmin);                                     //  Update Xmin
    Xmax = max(B->X, Xmax);                                     //  Update Xmax
  }                                                             // End loop over range of bodies
  for (int d=0; d<2; d++) X0[d] = (Xmax[d] + Xmin[d]) / 2;      // Calculate center of domain
  R0 = 0;                                                       // Initialize localRadius
  for (int d=0; d<2; d++) {                                     // Loop over dimensions
    R0 = std::max(X0[d] - Xmin[d], R0);                         //  Calculate min distance from center
    R0 = std::max(Xmax[d] - X0[d], R0);                         //  Calculate max distance from center
  }                                                             // End loop over dimensions
  R0 *= 1.00001;                                                // Add some leeway to radius
  stopTimer("Get bounds");                                      // Stop timer
  Bodies buffer = bodies;                                       // Copy bodies to buffer
  startTimer("Grow tree");                                      // Start timer
  B_iter B0 = bodies.begin();                                   // Iterator of first body
  Cell * C0 = buildTree(bodies, buffer, B0, 0, bodies.size(), X0, R0, ncrit);// Build tree recursively
  stopTimer("Grow tree");                                       // Stop timer

  startTimer("Upward pass");                                    // Start timer
  upwardPass(C0);                                               // Upward pass for P2M, M2M
  stopTimer("Upward pass");                                     // Stop timer
  traversal.dualTreeTraversal(C0, C0, cycle);                   // Traversal for M2L, P2P
  Bodies jbodies = bodies;
  startTimer("Downward pass");                                  // Start timer
  downwardPass(C0);                                             // Downward pass for L2L, L2P
  stopTimer("Downward pass");                                   // Stop timer
  printf("--- Total runtime ----------------\n");
  stopTimer("Total FMM");
//! Downsize target bodies by even sampling 
  int stride = bodies.size() / numTargets;                      // Stride of sampling
  for (int i=0; i<numTargets; i++) {                            // Loop over target samples
    bodies[i] = bodies[i*stride];                               //  Sample targets
  }                                                             // End loop over target samples
  bodies.resize(numTargets);                                    // Resize bodies to target size
  Bodies bodies2 = bodies;                                      // Save bodies in bodies2
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over bodies
    B->TRG = 0;                                                 //  Clear target values
  }                                                             // End loop over bodies
  startTimer("Total Direct");
  traversal.direct(bodies, jbodies, cycle);
  traversal.normalize(bodies);
  stopTimer("Total Direct");
  double diff1 = 0, norm1 = 0;
//! Evaluate relaitve L2 norm error
  B_iter B2 = bodies2.begin();                                // Set iterator for bodies2
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
    double dp = (B->TRG - B2->TRG) * (B->TRG - B2->TRG);      // Difference of potential
    double  p = B2->TRG * B2->TRG;                            //  Value of potential
    diff1 += dp;                                              //  Accumulate difference of potential
    norm1 += p;                                               //  Accumulate value of potential
  }                                                           // End loop over bodies & bodies2
  printf("--- FMM vs. direct ---------------\n");
  printf("Rel. L2 Error (pot)  : %e\n",sqrtf(diff1/norm1));  // Print potential error
  return 0;
}
