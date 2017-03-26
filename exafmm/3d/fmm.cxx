#include "build_tree.h"
#include "kernel.h"
#include "ewald.h"
#include "timer.h"
#include "traversal.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 1000;
  ncrit = 64;
  cycle = 2 * M_PI;
  P = 10;
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
    bodies[b].SRC = drand48() - .5;
    average += bodies[b].SRC;
  }
  average /= bodies.size();
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].SRC -= average;
    bodies[b].TRG = 0;
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
  upwardPass(cells.begin(), cells.begin());
  stop("Upward pass");
  start("Traversal");
  traversal(cells, cells, cycle);
  stop("Traversal");
  start("Downward pass");
  downwardPass(cells.begin(), cells.begin());
  stop("Downward pass");

  // Dipole correction
  start("Dipole correction");
  buffer = bodies;
  vec3 dipole = 0;
  for (int b=0; b<int(bodies.size()); b++) {
    dipole += bodies[b].X * bodies[b].SRC;
  }
  real_t coef = 4 * M_PI / (3 * cycle * cycle * cycle);
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].TRG[0] -= coef * norm(dipole) / bodies.size() / bodies[b].SRC;
    for (int d=0; d!=3; d++) {
      bodies[b].TRG[d+1] -= coef * dipole[d];
    }
  }
  stop("Dipole correction");

  printf("--- %-16s ------------\n", "Ewald Profiling");
  // Ewald summation
  start("Build tree");
  Bodies bodies2 = bodies;
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].TRG = 0;
  }
  Bodies jbodies = bodies;
  Cells jcells = buildTree(jbodies, buffer);
  stop("Build tree");
  start("Wave part");
  wavePart(bodies, jbodies);
  stop("Wave part");
  start("Real part");
  realPart(cells, jcells);
  selfTerm(bodies);
  stop("Real part");

  // Verify result
  real_t pSum = 0, pSum2 = 0, FDif = 0, FNrm = 0;
  for (int b=0; b<int(bodies.size()); b++) {
    pSum += bodies[b].TRG[0] * bodies[b].SRC;
    pSum2 += bodies2[b].TRG[0] * bodies2[b].SRC;
    FDif += (bodies[b].TRG[1] - bodies2[b].TRG[1]) * (bodies[b].TRG[1] - bodies2[b].TRG[1]) +
      (bodies[b].TRG[2] - bodies2[b].TRG[2]) * (bodies[b].TRG[2] - bodies2[b].TRG[2]) +
      (bodies[b].TRG[3] - bodies2[b].TRG[3]) * (bodies[b].TRG[3] - bodies2[b].TRG[3]);
    FNrm += bodies[b].TRG[1] * bodies[b].TRG[1] + bodies[b].TRG[2] * bodies[b].TRG[2] +
      bodies[b].TRG[3] * bodies[b].TRG[3];
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
