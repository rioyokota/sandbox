#ifndef dataset_h
#define dataset_h
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "types.h"

//! Initialize dsitribution, source & target value of bodies
Bodies initBodies(int numBodies) {
  srand48(0);                                                 // Set seed for random number generator
  Bodies bodies(numBodies);                                   // Initialize bodies
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
    for (int d=0; d<2; d++) {                                 //  Loop over dimension
      B->X[d] = drand48() * 2 * M_PI - M_PI;                  //   Initialize positions
    }                                                         //  End loop over dimension
    B->SRC = drand48() - .5;                                  //   Initialize charge
    B->TRG = 0;                                               //  Clear target values
  }                                                           // End loop over bodies
  return bodies;                                              // Return bodies
}

//! Downsize target bodies by even sampling 
void sampleBodies(Bodies &bodies, int numTargets) {
  if (numTargets < int(bodies.size())) {                      // If target size is smaller than current
    int stride = bodies.size() / numTargets;                  //  Stride of sampling
    for (int i=0; i<numTargets; i++) {                        //  Loop over target samples
      bodies[i] = bodies[i*stride];                           //   Sample targets
    }                                                         //  End loop over target samples
    bodies.resize(numTargets);                                //  Resize bodies to target size
  }                                                           // End if for target size
}

//! Evaluate relaitve L2 norm error
void evalError(Bodies &bodies, Bodies &bodies2, double &diff1, double &norm1) {
  B_iter B2 = bodies2.begin();                                // Set iterator for bodies2
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) { // Loop over bodies & bodies2
    double dp = (B->TRG - B2->TRG) * (B->TRG - B2->TRG);      // Difference of potential
    double  p = B2->TRG * B2->TRG;                            //  Value of potential
    diff1 += dp;                                              //  Accumulate difference of potential
    norm1 += p;                                               //  Accumulate value of potential
  }                                                           // End loop over bodies & bodies2
}
#endif
