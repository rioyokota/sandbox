void getCoef(real *C, const real &dx, real &invR2, const real &invR) {
  C[0] = invR;
  invR2 = -invR2;
  real invR3 = invR * invR2;
  C[1] = dx * invR3;
  real invR5 = 3 * invR3 * invR2;
  real t = dx * invR5;
  C[2] = dx * t + invR3;
  real invR7 = 5 * invR5 * invR2;
  t = dx * dx * invR7;
  C[3] = dx * (t + 3 * invR5);
  real invR9 = 7 * invR7 * invR2;
  t = dx * dx * invR9;
  C[4] = dx * dx * (t + 6 * invR7) + 3 * invR5;
  real invR11 = 9 * invR9 * invR2;
  t = dx * dx * invR11;
  C[5] = dx * dx * dx * (t + 10 * invR9) + 15 * dx * invR7;
  real invR13 = 11 * invR11 * invR2;
  t = dx * dx * invR13;
  C[6] = dx * dx * dx * dx * (t + 15 * invR11) + 45 * dx * dx * invR9 + 15 * invR7;
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

void M2MSum(real *MI, const real *C, const real *MJ) {
  for (int i=1; i<PP; i++) MI[i] += MJ[i];
  MI[2] += C[1]*MJ[1];
  MI[3] += C[1]*MJ[2]+C[2]*MJ[1];
  MI[4] += C[1]*MJ[3]+C[2]*MJ[2]+C[3]*MJ[1];
  MI[5] += C[1]*MJ[4]+C[2]*MJ[3]+C[3]*MJ[2]+C[4]*MJ[1];
}
