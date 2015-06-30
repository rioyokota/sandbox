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
