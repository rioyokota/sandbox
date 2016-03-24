// Step 1. Direct summation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
  int i, j, N = 10000;
  double x[N], y[N], u[N], q[N];
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
#pragma omp parallel for private(j)
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx * dx + dy * dy);
      u[i] += q[j] / r;
    }
  }
}
