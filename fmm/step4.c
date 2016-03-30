// Step 4. Single-level at level 3

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int getIndex(int iX[2], int level){
  int d, l, index = 0;
  int jX[2] = {iX[0], iX[1]};
  for (l=0; l<level; l++) {
    index += (jX[1] & 1) << (2*l);
    jX[1] >>= 1;
    index += (jX[0] & 1) << (2*l+1);
    jX[0] >>= 1;
  }
  return index;
}

void getIX(int iX[2], int index) {
  int l = 0;
  iX[0] = iX[1] = 0;
  while (index > 0) {
    iX[1] += (index & 1) << l;
    index >>= 1;
    iX[0] += (index & 1) << l;
    index >>= 1;
    l++;
  }
}

int main() {
  int i, j, N = 100;
  int level = 3;
  double x[N], y[N], u[N], q[N];
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
  int nx = 1 << level;
  int Ncell = nx * nx;
  double M[Ncell], L[Ncell];
  for (i=0; i<Ncell; i++) {
    M[i] = L[i] = 0;
  }
  // P2M
  int iX[2];
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    M[j] += q[i];
  }
  // M2L
  int jX[2];
  for (i=0; i<Ncell; i++) {
    getIX(iX, i);
    for (j=0; j<Ncell; j++) {
      getIX(jX, j);
      if (abs(iX[0]-jX[0]) > 1 || abs(iX[1]-jX[1]) > 1) {
	double dx = (double)(iX[0] - jX[0]) / nx;
	double dy = (double)(iX[1] - jX[1]) / nx;
	double r = sqrt(dx * dx + dy * dy);
	i = getIndex(iX, level);
	j = getIndex(jX, level);
	L[i] += M[j] / r;
      }
    }
  }
  // L2P
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    j = getIndex(iX, level);
    u[i] += L[j];
  }
  // P2P
  for (i=0; i<N; i++) {
    iX[0] = x[i] * nx;
    iX[1] = y[i] * nx;
    for (j=0; j<N; j++) {
      jX[0] = x[j] * nx;
      jX[1] = y[j] * nx;
      if (abs(iX[0]-jX[0]) <= 1 && abs(iX[1]-jX[1]) <= 1) {
	double dx = x[i] - x[j];
	double dy = y[i] - y[j];
	double r = sqrt(dx * dx + dy * dy);
	if (r!=0) u[i] += q[j] / r;
      }
    }
  }
  // Check answer
  for (i=0; i<N; i++) {
    double ui = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx * dx + dy * dy);
      if (r != 0) ui += q[j] / r;
    }
    printf("%d %lf %lf\n", i, u[i], ui);
  }
}
