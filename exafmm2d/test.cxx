#include <cassert>
#include <cstdlib>

#include "buildtree.h"
#include "logger.h"
#include "traversal.h"
#include "updownpass.h"

int main(int argc, char ** argv) {
  const int numBodies = 100000;
  const int images = 0;
  const int ncrit = 8;
  const real_t theta = 0.4;
  const real_t eps2 = 0.0;
  const real_t cycle = 2 * M_PI;
  BuildTree tree(ncrit);
  UpDownPass pass(theta);
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
  Bounds bounds;                                                // Bounds : Contains Xmin, Xmax
  bounds.Xmin = bounds.Xmax = bodies.front().X;                 // Initialize Xmin, Xmax
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {         // Loop over range of bodies
    bounds.Xmin = min(B->X, bounds.Xmin);                       //  Update Xmin
    bounds.Xmax = max(B->X, bounds.Xmax);                       //  Update Xmax
  }                                                             // End loop over range of bodies
  stopTimer("Get bounds");                                      // Stop timer

  Cells cells = tree.buildTree(bodies, bounds);
  pass.upwardPass(cells);
  traversal.dualTreeTraversal(cells, cells, cycle);
  Bodies jbodies = bodies;
  pass.downwardPass(cells);
  printf("--- Total runtime ----------------\n");
  stopTimer("Total FMM");
//! Downsize target bodies by even sampling 
  int numTargets = 100;                                         // Number of target bodies
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
  std::cout << std::setw(20) << std::left << std::scientific  //  Set format
	    << "Rel. L2 Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;// Print potential error
  return 0;
}
