void getCoef(real *C, const real *dist, real &invR2, const real &invR) {
  C[0] = invR;
  invR2 = -invR2;
  real x = dist[0];
  real invR3 = invR * invR2;
  C[1] = x * invR3;
  real invR5 = 3 * invR3 * invR2;
  real t = x * invR5;
  C[4] = x * t + invR3;
  real invR7 = 5 * invR5 * invR2;
  t = x * x * invR7;
  C[10] = x * (t + 3 * invR5);
  real invR9 = 7 * invR7 * invR2;
  t = x * x * invR9;
  C[20] = x * x * (t + 6 * invR7) + 3 * invR5;
  real invR11 = 9 * invR9 * invR2;
  t = x * x * invR11;
  C[35] = x * x * x * (t + 10 * invR9) + 15 * x * invR7;
  real invR13 = 11 * invR11 * invR2;
  t = x * x * invR13;
  C[56] = x * x * x * x * (t + 15 * invR11) + 45 * x * x * invR9 + 15 * invR7;
}

void M2LSum(real *L, const real *C, const real*M) {
  L[0] += M[0]*C[0];

  L[0] += M[1]*C[1];
  L[1] += M[1]*C[4];
  L[1] += M[0]*C[1];

  L[0] += M[4]*C[4];
  L[1] += M[4]*C[10];
  L[4] += M[1]*C[10];
  L[4] += M[0]*C[4];

  L[0] += M[10]*C[10];
  L[1] += M[10]*C[20];
  L[4] += M[4]*C[20];
  L[10] += M[1]*C[20];
  L[10] += M[0]*C[10];

  L[0] += M[20]*C[20];
  L[1] += M[20]*C[35];
  L[4] += M[10]*C[35];
  L[10] += M[4]*C[35];
  L[20] += M[1]*C[35];
  L[20] += M[0]*C[20];

  L[0] += M[35]*C[35];
  L[1] += M[35]*C[56];
  L[4] += M[20]*C[56];
  L[10] += M[10]*C[56];
  L[20] += M[4]*C[56];
  L[35] += M[1]*C[56];
  L[35] += M[0]*C[35];
}

void powerM(real *C, const real *dist) {
  C[1] = C[0] * dist[0];
  C[4] = C[1] * dist[0] / 2;
  C[10] = C[4] * dist[0] / 3;
  C[20] = C[10] * dist[0] / 4;
  C[35] = C[20] * dist[0] / 5;
}

void powerL(real *C, const real *dist) {
  C[1] = C[0] * dist[0];
  C[4] = C[1] * dist[0] / 2;
  C[10] = C[4] * dist[0] / 3;
  C[20] = C[10] * dist[0] / 4;
  C[35] = C[20] * dist[0] / 5;
  C[56] = C[35] * dist[0] / 6;
}

void M2MSum(real *MI, const real *C, const real *MJ) {
  for (int i=1; i<MTERM; i++) MI[i] += MJ[i];
  MI[4] += C[1]*MJ[1];
  MI[10] += C[1]*MJ[4]+C[4]*MJ[1];
  MI[20] += C[1]*MJ[10]+C[4]*MJ[4]+C[10]*MJ[1];
  MI[35] += C[1]*MJ[20]+C[4]*MJ[10]+C[10]*MJ[4]+C[20]*MJ[1];
}

void L2LSum(real *LI, const real *C, const real *LJ) {
  LI[1] += C[1]*LJ[4];

  LI[1] += C[4]*LJ[10];
  LI[4] += C[1]*LJ[10];

  LI[1] += C[10]*LJ[20];
  LI[4] += C[4]*LJ[20];
  LI[10] += C[1]*LJ[20];

  LI[1] += C[20]*LJ[35];
  LI[4] += C[10]*LJ[35];
  LI[10] += C[4]*LJ[35];
  LI[20] += C[1]*LJ[35];

  LI[1] += C[35]*LJ[56];
  LI[4] += C[20]*LJ[56];
  LI[10] += C[10]*LJ[56];
  LI[20] += C[4]*LJ[56];
  LI[35] += C[1]*LJ[56];
}

void L2PSum(real *TRG, const real *C, const real *L) {
  TRG[1] += C[1]*L[4];
  TRG[1] += C[4]*L[10];
  TRG[1] += C[10]*L[20];
  TRG[1] += C[20]*L[35];
  TRG[1] += C[35]*L[56];
}
