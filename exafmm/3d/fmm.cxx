#include "build_tree.h"
#include "ewald.h"
#include "kernel.h"
#include "timer.h"
#include "traversal.h"
#include "up_down_pass.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  const int numBodies = 1000;
  const int P = 10;
  const int ncrit = 64;
  const int images = 4;
  const int ksize = 11;
  const real_t cycle = 2 * M_PI;
  const real_t alpha = ksize / cycle;
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = cycle / 2;
  const real_t theta = 0.4;

  // Initialize bodies
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

  // Build tree
  timer::printTitle("FMM Profiling");
  Bodies buffer(numBodies);
  BuildTree buildTree(ncrit);
  Cells cells = buildTree.buildTree(bodies, buffer);

  // FMM evaluation
  Kernel kernel(P);
  UpDownPass upDownPass(kernel);
  upDownPass.upwardPass(cells);
  Traversal traversal(kernel, theta, images);
  traversal.traverse(cells, cells, cycle);
  upDownPass.downwardPass(cells);

  // Dipole correction
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

  // Ewald summation
  timer::printTitle("Ewald Profiling");
  Bodies bodies2 = bodies;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->TRG = 0;
  }
  Bodies jbodies = bodies;
  Cells jcells = buildTree.buildTree(jbodies, buffer);
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  ewald.wavePart(bodies, jbodies);
  ewald.realPart(cells, jcells);
  ewald.selfTerm(bodies);

  // Verify result
  real_t potSum = 0, potSum2 = 0, accDif = 0, accNrm = 0;
  B_iter B2 = bodies2.begin();
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) {
    potSum += B->TRG[0] * B->SRC;
    potSum2 += B2->TRG[0] * B2->SRC;
    accDif += (B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]) +
      (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]) +
      (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]);
    accNrm += B->TRG[1] * B->TRG[1] + B->TRG[2] * B->TRG[2] + B->TRG[3] * B->TRG[3];
  }
  double potDif = (potSum - potSum2) * (potSum - potSum2);
  double potNrm = potSum * potSum;
  double potRel = std::sqrt(potDif/potNrm);
  double accRel = std::sqrt(accDif/accNrm);
  timer::printTitle("FMM vs. Ewald");
  printf("%-20s : %8.5e s\n","Rel. L2 Error (pot)", potRel);
  printf("%-20s : %8.5e s\n","Rel. L2 Error (acc)", accRel);
  return 0;
}
