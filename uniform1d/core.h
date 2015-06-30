void getCoef(real *C, const real *dist, real &invR2, const real &invR) {
  C[0] = invR;
  invR2 = -invR2;
  real x = dist[0];
  real invR3 = invR * invR2;
  C[1] = x * invR3;
  real invR5 = 3 * invR3 * invR2;
  real t = x * invR5;
  C[2] = x * t + invR3;
  real invR7 = 5 * invR5 * invR2;
  t = x * x * invR7;
  C[3] = x * (t + 3 * invR5);
  real invR9 = 7 * invR7 * invR2;
  t = x * x * invR9;
  C[4] = x * x * (t + 6 * invR7) + 3 * invR5;
  real invR11 = 9 * invR9 * invR2;
  t = x * x * invR11;
  C[5] = x * x * x * (t + 10 * invR9) + 15 * x * invR7;
  real invR13 = 11 * invR11 * invR2;
  t = x * x * invR13;
  C[6] = x * x * x * x * (t + 15 * invR11) + 45 * x * x * invR9 + 15 * invR7;
}

void M2LSum(real *L, const real *C, const real*M) {
  L[0] += M[0]*C[0];

  L[0] += M[1]*C[1];
  L[1] += M[1]*C[2];
  L[1] += M[0]*C[1];

  L[0] += M[2]*C[2];
  L[1] += M[2]*C[3];
  L[2] += M[1]*C[3];
  L[2] += M[0]*C[2];

  L[0] += M[3]*C[3];
  L[1] += M[3]*C[4];
  L[2] += M[2]*C[4];
  L[3] += M[1]*C[4];
  L[3] += M[0]*C[3];

  L[0] += M[4]*C[4];
  L[1] += M[4]*C[5];
  L[2] += M[3]*C[5];
  L[3] += M[2]*C[5];
  L[4] += M[1]*C[5];
  L[4] += M[0]*C[4];

  L[0] += M[5]*C[5];
  L[1] += M[5]*C[6];
  L[2] += M[4]*C[6];
  L[3] += M[3]*C[6];
  L[4] += M[2]*C[6];
  L[5] += M[1]*C[6];
  L[5] += M[0]*C[5];
}

void powerM(real *C, const real *dist) {
  C[1] = C[0] * dist[0];
  C[2] = C[1] * dist[0] / 2;
  C[3] = C[2] * dist[0] / 3;
  C[4] = C[3] * dist[0] / 4;
  C[5] = C[4] * dist[0] / 5;
}

void powerL(real *C, const real *dist) {
  C[1] = C[0] * dist[0];
  C[2] = C[1] * dist[0] / 2;
  C[3] = C[2] * dist[0] / 3;
  C[4] = C[3] * dist[0] / 4;
  C[5] = C[4] * dist[0] / 5;
  C[6] = C[5] * dist[0] / 6;
}

void M2MSum(real *MI, const real *C, const real *MJ) {
  for (int i=1; i<PP; i++) MI[i] += MJ[i];
  MI[2] += C[1]*MJ[1];
  MI[3] += C[1]*MJ[2]+C[2]*MJ[1];
  MI[4] += C[1]*MJ[3]+C[2]*MJ[2]+C[3]*MJ[1];
  MI[5] += C[1]*MJ[4]+C[2]*MJ[3]+C[3]*MJ[2]+C[4]*MJ[1];
}

void L2LSum(real *LI, const real *C, const real *LJ) {
  LI[1] += C[1]*LJ[2];

  LI[1] += C[2]*LJ[3];
  LI[2] += C[1]*LJ[3];

  LI[1] += C[3]*LJ[4];
  LI[2] += C[2]*LJ[4];
  LI[3] += C[1]*LJ[4];

  LI[1] += C[4]*LJ[5];
  LI[2] += C[3]*LJ[5];
  LI[3] += C[2]*LJ[5];
  LI[4] += C[1]*LJ[5];

  LI[1] += C[5]*LJ[6];
  LI[2] += C[4]*LJ[6];
  LI[3] += C[3]*LJ[6];
  LI[4] += C[2]*LJ[6];
  LI[5] += C[1]*LJ[6];
}

void L2PSum(real *TRG, const real *C, const real *L) {
  TRG[1] += C[1]*L[2];
  TRG[1] += C[2]*L[3];
  TRG[1] += C[3]*L[4];
  TRG[1] += C[4]*L[5];
  TRG[1] += C[5]*L[6];
}
