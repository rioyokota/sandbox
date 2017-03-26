#include "kernel.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  P = atoi(argv[1]);
  initKernel();

  // P2M
  Bodies jbodies(1);
  jbodies[0].X = 2;
  jbodies[0].SRC = 1;
  Cells cells(4);
  C_iter Cj = cells.begin();
  Cj->X = 1;
  Cj->X[0] = 3;
  Cj->R = 1;
  Cj->BODY = jbodies.begin();
  Cj->NBODY = jbodies.size();
  Cj->M.resize(NTERM, 0.0);
  P2M(Cj);

  // M2M
  C_iter CJ = cells.begin()+1;
  CJ->ICHILD = Cj-cells.begin();
  CJ->NCHILD = 1;
  CJ->X = 0;
  CJ->X[0] = 4;
  CJ->R = 2;
  CJ->M.resize(NTERM, 0.0);
  M2M(CJ, cells.begin());

  // M2L
  C_iter CI = cells.begin()+2;
  CI->X = 0;
  CI->X[0] = -4;
  CI->R = 2;
  CI->L.resize(NTERM, 0.0);
  M2L(CI, CJ);

  // L2L
  C_iter Ci = cells.begin()+3;
  CI->ICHILD = Ci-cells.begin();
  CI->NCHILD = 1;
  Ci->X = 1;
  Ci->X[0] = -3;
  Ci->R = 1;
  Ci->L.resize(NTERM, 0.0);
  L2L(CI, cells.begin());

  // L2P
  Bodies bodies(1);
  bodies[0].X = 2;
  bodies[0].X[0] = -2;
  bodies[0].SRC = 1;
  bodies[0].TRG = 0;
  Ci->BODY = bodies.begin();
  Ci->NBODY = bodies.size();
  L2P(Ci);

  // P2P
  Bodies bodies2(1);
  for (B_iter B=bodies2.begin(); B!=bodies2.end(); B++) {
    *B = bodies[B-bodies2.begin()];
    B->TRG = 0;
  }
  Cj->NBODY = jbodies.size();
  Ci->NBODY = bodies2.size();
  Ci->BODY = bodies2.begin();
  P2P(Ci, Cj);
  for (B_iter B=bodies2.begin(); B!=bodies2.end(); B++) {
    B->TRG /= B->SRC;
  }

  // Verify results
  real_t potDif = 0, potNrm = 0, accDif = 0, accNrm = 0;
  B_iter B2 = bodies2.begin();
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, B2++) {
    potDif += (B->TRG[0] - B2->TRG[0]) * (B->TRG[0] - B2->TRG[0]);
    potNrm += B->TRG[0] * B->TRG[0];
    accDif += (B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]) +
      (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]) +
      (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]);
    accNrm += B->TRG[1] * B->TRG[1] + B->TRG[2] * B->TRG[2] + B->TRG[3] * B->TRG[3];
  }
  real_t potRel = std::sqrt(potDif/potNrm);
  real_t accRel = std::sqrt(accDif/accNrm);
  printf("%-20s : %8.5e s\n","Rel. L2 Error (pot)", potRel);
  printf("%-20s : %8.5e s\n","Rel. L2 Error (acc)", accRel);
  return 0;
}
