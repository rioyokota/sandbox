#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void solve(double * A, double * b, double * x, int n, int max_it, double tol) {
  int it, i, j;
  double * r = (double*) malloc(n * sizeof(double));
  double * p = (double*) malloc(n * sizeof(double));
  double * Ap = (double*) malloc(n * sizeof(double));
  for (i=0; i<n; i++) {
    double Ax = 0;
    for (j=0; j<n; j++) {
      Ax += A[n*i+j] * x[j];
    }
    r[i] = b[i] - Ax;
  }
  for (i=0; i<n; i++) p[i] = r[i];
  double rr = 0;
  for (i=0; i<n; i++) rr += r[i] * r[i];
  for (it=0; it<max_it; it++) {
    for (i=0; i<n; i++) {
      double Api = 0;
      for (j=0; j<n; j++) {
        Api += A[n*i+j] * p[j];
      }
      Ap[i] = Api;
    }
    double pAp = 0;
    for (i=0; i<n; i++) pAp += p[i] * Ap[i];
    double alpha = rr / pAp;
    for (i=0; i<n; i++) x[i] += alpha * p[i];
    for (i=0; i<n; i++) r[i] -= alpha * Ap[i];
    double r2 = 0;
    for (i=0; i<n; i++) r2 += r[i] * r[i];
    double res = sqrt(r2);
    printf("%d %lf\n",it,res);
    if (res < tol) break;
    double beta = r2 / rr;
    for (i=0; i<n; i++) p[i] = r[i] + beta * p[i];
    rr = r2;
  }
  free(r);
  free(p);
  free(Ap);
}
