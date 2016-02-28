#include <cmath>
#include <cstdlib>
#include <sys/time.h>

#include "buildtree.h"
#include "kernel.h"
#include "traversal.h"
#include "types.h"

//! Timer function
double getTime() {
  struct timeval tv;                                            // Time value
  gettimeofday(&tv, NULL);                                      // Get time of day in seconds and microseconds
  return double(tv.tv_sec+tv.tv_usec*1e-6);                     // Combine seconds and microseconds and return
}

int main(int argc, char ** argv) {
  const int numBodies = 10000;
  const int numTargets = 10;
  const int ncrit = 8;
  const real_t eps2 = 0.0;
  const real_t cycle = 2 * M_PI;
  images = 3;
  theta = 0.4;
  printf("--- FMM Profiling ----------------\n");

  //! Initialize dsitribution, source & target value of bodies
  double time = getTime();                                      // Start timer
  srand48(0);                                                   // Set seed for random number generator
  Body * bodies = new Body [numBodies];                         // Initialize bodies
  real_t average = 0;                                           // Average charge
  for (int b=0; b<numBodies; b++) {                             // Loop over bodies
    for (int d=0; d<2; d++) {                                   //  Loop over dimension
      bodies[b].X[d] = drand48() * 2 * M_PI - M_PI;             //   Initialize positions
    }                                                           //  End loop over dimension
    bodies[b].SRC = drand48() - .5;                             //  Initialize charge
    average += bodies[b].SRC;                                   //  Accumulate charge
    bodies[b].TRG = 0;                                          //  Clear target values
  }                                                             // End loop over bodies
  average /= numBodies;                                         // Average charge
  for (int b=0; b<numBodies; b++) {                             // Loop over bodies
    bodies[b].SRC -= average;                                   // Charge neutral
  }                                                             // End loop over bodies
  printf("%-20s : %lf s\n","Init bodies",getTime()-time);       // Stop timer

  // ! Get Xmin and Xmax of domain
  time = getTime();                                             // Start timer 
  real_t R0;                                                    // Radius of root cell
  real_t Xmin[2], Xmax[2], X0[2];                               // Min, max of domain, and center of root cell
  for (int d=0; d<2; d++) Xmin[d] = Xmax[d] = bodies[0].X[d];   // Initialize Xmin, Xmax
  for (int b=0; b<numBodies; b++) {                             // Loop over range of bodies
    for (int d=0; d<2; d++) Xmin[d] = fmin(bodies[b].X[d], Xmin[d]);//  Update Xmin
    for (int d=0; d<2; d++) Xmax[d] = fmax(bodies[b].X[d], Xmax[d]);//  Update Xmax
  }                                                             // End loop over range of bodies
  for (int d=0; d<2; d++) X0[d] = (Xmax[d] + Xmin[d]) / 2;      // Calculate center of domain
  R0 = 0;                                                       // Initialize localRadius
  for (int d=0; d<2; d++) {                                     // Loop over dimensions
    R0 = std::max(X0[d] - Xmin[d], R0);                         //  Calculate min distance from center
    R0 = std::max(Xmax[d] - X0[d], R0);                         //  Calculate max distance from center
  }                                                             // End loop over dimensions
  R0 *= 1.00001;                                                // Add some leeway to radius
  printf("%-20s : %lf s\n","Get bounds",getTime()-time);        // Stop timer 

  //! Build tree structure
  time = getTime();                                             // Start timer 
  Body * buffer = new Body [numBodies];                         // Buffer for bodies
  for (int b=0; b<numBodies; b++) buffer[b] = bodies[b];        // Copy bodies to buffer
  Cell * C0 = buildTree(bodies, buffer, 0, numBodies, X0, R0, ncrit);// Build tree recursively
  printf("%-20s : %lf s\n","Grow tree",getTime()-time);         // Stop timer 

  //! FMM evaluation
  time = getTime();                                             // Start timer 
  upwardPass(C0);                                               // Upward pass for P2M, M2M
  printf("%-20s : %lf s\n","Upward pass",getTime()-time);       // Stop timer 
  time = getTime();                                             // Start timer 
  traversal(C0, C0, cycle);                                     // Traversal for M2L, P2P
  printf("%-20s : %lf s\n","Traverse",getTime()-time);          // Stop timer 
  time = getTime();                                             // Start timer 
  downwardPass(C0);                                             // Downward pass for L2L, L2P
  printf("%-20s : %lf s\n","Downward pass",getTime()-time);     // Stop timer 

  //! Downsize target bodies by even sampling 
  Body * jbodies = new Body [numBodies];                        // Source bodies
  for (int b=0; b<numBodies; b++) jbodies[b] = bodies[b];       // Save bodies in jbodies
  int stride = numBodies / numTargets;                          // Stride of sampling
  for (int b=0; b<numTargets; b++) {                            // Loop over target samples
    bodies[b] = bodies[b*stride];                               //  Sample targets
  }                                                             // End loop over target samples
  Body * bodies2 = new Body [numTargets];                       // Backup bodies
  for (int b=0; b<numTargets; b++) {                            // Loop over bodies
    bodies2[b].TRG = bodies[b].TRG;                             //  Save bodies target in bodies2
  }                                                             // End loop over bodies
  for (int b=0; b<numTargets; b++) {                            // Loop over bodies
    bodies[b].TRG = 0;                                          //  Clear target values
  }                                                             // End loop over bodies
  time = getTime();                                             // Start timer 
  direct(numTargets, bodies, numBodies, jbodies, cycle);        // Direc N-body
  printf("%-20s : %lf s\n","Direct N-Body",getTime()-time);     // Stop timer 

    //! Evaluate relaitve L2 norm error
  double diff1 = 0, norm1 = 0;
  for (int b=0; b<numTargets; b++) {                            // Loop over bodies & bodies2
    double dp = (bodies[b].TRG - bodies2[b].TRG) * (bodies[b].TRG - bodies2[b].TRG);// Difference of potential
    double  p = bodies2[b].TRG * bodies2[b].TRG;                //  Value of potential
    diff1 += dp;                                                //  Accumulate difference of potential
    norm1 += p;                                                 //  Accumulate value of potential
  }                                                             // End loop over bodies & bodies2
  printf("--- FMM vs. direct ---------------\n");               // Print message
  printf("Rel. L2 Error (pot)  : %e\n",sqrtf(diff1/norm1));     // Print potential error
  return 0;
}
