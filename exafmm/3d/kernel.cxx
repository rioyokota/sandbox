#include <cmath>
#include <cstdlib>
#include <iostream>
#include "namespace.h"
#include "kernel.h"
#include <stdint.h>
#include <vector>
#include "verify.h"
using namespace EXAFMM_NAMESPACE;

int main(int argc, char ** argv) {
  const int P = atoi(argv[1]);
  const real_t eps2 = 0.0;
  const complex_t wavek = complex_t(1.,.1) / real_t(2 * M_PI);
  Bodies bodies(1), bodies2(1), jbodies(1);
  Kernel kernel(P, eps2, wavek);
  logger::verbose = true;

  Cells cells(4);
  Verify verify;
  jbodies[0].X = 2;
  jbodies[0].SRC = 1;
  C_iter Cj = cells.begin();
  Cj->X = 1;
  Cj->X[0] = 3;
  Cj->R = 1;
  Cj->BODY = jbodies.begin();
  Cj->NBODY = jbodies.size();
  Cj->M.resize(kernel.NTERM, 0.0);
  kernel.P2M(Cj);

  C_iter CJ = cells.begin()+1;
  CJ->ICHILD = Cj-cells.begin();
  CJ->NCHILD = 1;
  CJ->X = 0;
  CJ->X[0] = 4;
  CJ->R = 2;
  CJ->M.resize(kernel.NTERM, 0.0);
  kernel.M2M(CJ, cells.begin());

  C_iter CI = cells.begin()+2;
  CI->X = 0;
  CI->X[0] = -4;
  CI->R = 2;
  CI->L.resize(kernel.NTERM, 0.0);
  kernel.M2L(CI, CJ);

  C_iter Ci = cells.begin()+3;
  Ci->X = 1;
  Ci->X[0] = -3;
  Ci->R = 1;
  Ci->IPARENT = 2;
  Ci->L.resize(kernel.NTERM, 0.0);
  kernel.L2L(Ci, cells.begin());

  bodies[0].X = 2;
  bodies[0].X[0] = -2;
  bodies[0].SRC = 1;
  bodies[0].TRG = 0;
  Ci->BODY = bodies.begin();
  Ci->NBODY = bodies.size();
  kernel.L2P(Ci);

  for (B_iter B=bodies2.begin(); B!=bodies2.end(); B++) {
    *B = bodies[B-bodies2.begin()];
    B->TRG = 0;
  }
  Cj->NBODY = jbodies.size();
  Ci->NBODY = bodies2.size();
  Ci->BODY = bodies2.begin();
  kernel.P2P(Ci, Cj);
  for (B_iter B=bodies2.begin(); B!=bodies2.end(); B++) {
    B->TRG /= B->SRC;
  }

  std::fstream file;
  double potDif = verify.getDifScalar(bodies, bodies2);
  double potNrm = verify.getNrmScalar(bodies);
  double accDif = verify.getDifVector(bodies, bodies2);
  double accNrm = verify.getNrmVector(bodies);
  std::cout << P << " " << std::sqrt(potDif/potNrm) << "  " << std::sqrt(accDif/accNrm) << std::endl;
  double potRel = std::sqrt(potDif/potNrm);
  double accRel = std::sqrt(accDif/accNrm);
  verify.print("Rel. L2 Error (pot)",potRel);
  verify.print("Rel. L2 Error (acc)",accRel);
  return 0;
}
