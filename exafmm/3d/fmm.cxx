#include "build_tree.h"
#include "kernel.h"
#include "ewald.h"
#include "timer.h"
#include "traversal.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 1000;
  const real_t cycle = 2 * M_PI;
  P = 10;
  ncrit = 64;
  theta = 0.4;
  images = 4;

  ksize = 11;
  alpha = ksize / cycle;
  sigma = .25 / M_PI;
  cutoff = cycle / 2;

  printf("--- %-16s ------------\n", "FMM Profiling");
  // Initialize bodies
  start("Initialize bodies");
  Bodies bodies(numBodies);
  real_t average = 0;
  srand48(0);
  for (int b=0; b<int(bodies.size()); b++) {
    for (int d=0; d<3; d++) {
      bodies[b].X[d] = drand48() * cycle - cycle * .5;
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
    bodies[b].F = 0;
  }
  stop("Initialize bodies");

  // Build tree
  start("Build tree");
  Bodies buffer(bodies.size());
  Cells cells = buildTree(bodies, buffer);
  stop("Build tree");

  // FMM evaluation
  start("Upward pass");
  initKernel();
  upwardPass(&cells[0]);
  stop("Upward pass");
  start("Traversal");
  traversal(&cells[0], &cells[0], cycle);
  stop("Traversal");
  start("Downward pass");
  downwardPass(&cells[0]);
  stop("Downward pass");

  // Dipole correction
  start("Dipole correction");
  buffer = bodies;
  real_t dipole[3] = {0, 0, 0};
  for (int b=0; b<int(bodies.size()); b++) {
    for (int d=0; d<3; d++) dipole[d] += bodies[b].X[d] * bodies[b].q;
  }
  real_t coef = 4 * M_PI / (3 * cycle * cycle * cycle);
  for (int b=0; b<int(bodies.size()); b++) {
    real_t dnorm = dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2];
    bodies[b].p -= coef * dnorm / bodies.size() / bodies[b].q;
    for (int d=0; d!=3; d++) bodies[b].F[d] -= coef * dipole[d];
  }
  stop("Dipole correction");

  printf("--- %-16s ------------\n", "Ewald Profiling");
  // Ewald summation
  start("Build tree");
  Bodies bodies2 = bodies;
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].p = 0;
    bodies[b].F = 0;
  }
  Bodies jbodies = bodies;
  Cells jcells = buildTree(jbodies, buffer);
  stop("Build tree");
  start("Wave part");
  wavePart(bodies, jbodies, cycle);
  stop("Wave part");
  start("Real part");
  realPart(cells, jcells, cycle);
  selfTerm(bodies);
  stop("Real part");

  // Verify result
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
