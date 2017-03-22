#include "build_tree.h"
#include "ewald.h"
#include "kernel.h"
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
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    for (int d=0; d<3; d++) {
      B->X[d] = drand48() * cycle - cycle * .5;
    }
  }
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->SRC = drand48() - .5;
    average += B->SRC;
  }
  average /= numBodies;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->SRC -= average;
    B->TRG = 0;
  }
  stop("Initialize bodies");

  // Build tree
  start("Build tree");
  Bodies buffer(numBodies);
  Cells cells = buildTree(bodies, buffer);
  stop("Build tree");

  // FMM evaluation
  start("Upward pass");
  initKernel();
  upwardPass(cells.begin(), cells.begin());
  stop("Upward pass");
  start("Traversal");
  traverse(cells, cells, cycle);
  stop("Traversal");
  start("Downward pass");
  downwardPass(cells.begin(), cells.begin());
  stop("Downward pass");

  // Dipole correction
  start("Dipole correction");
  buffer = bodies;
  vec3 dipole = 0;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    dipole += B->X * B->SRC;
  }
  real_t coef = 4 * M_PI / (3 * cycle * cycle * cycle);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->TRG[0] -= coef * norm(dipole) / numBodies / B->SRC;
    for (int d=0; d!=3; d++) {
      B->TRG[d+1] -= coef * dipole[d];
    }
  }
  stop("Dipole correction");

  printf("--- %-16s ------------\n", "Ewald Profiling");
  // Ewald summation
  start("Build tree");
  Bodies bodies2 = bodies;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->TRG = 0;
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
  B_iter B2 = bodies2.begin();
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) {
    pSum += B->TRG[0] * B->SRC;
    pSum2 += B2->TRG[0] * B2->SRC;
    FDif += (B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]) +
      (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]) +
      (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]);
    FNrm += B->TRG[1] * B->TRG[1] + B->TRG[2] * B->TRG[2] + B->TRG[3] * B->TRG[3];
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
