// Step 1. Far-field approximation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}
int main() {
  int i, j, N = 10000;
  double xi[N], yi[N], ui[N];
  double xj[N], yj[N], qj[N];
  double r = 2;
  double tic = get_time();
#pragma omp parallel for
  for (i=0; i<N; i++) {
    xi[i] = r+drand48();
    yi[i] = drand48();
    ui[i] = 0;
    xj[i] = drand48();
    yj[i] = drand48();
    qj[i] = 1;
  }
  // P2M
  double toc = get_time();
  printf("%f\n",toc-tic);
  double M = 0;
#pragma omp parallel for
  for (i=0; i<N; i++) {
    M += qj[i];
  }
  tic = get_time();
  printf("%f\n",tic-toc);
  // M2L
  double L = M / r;
  // L2P
#pragma omp parallel for
  for (i=0; i<N; i++) {
    ui[i] = L;
  }
  toc = get_time();
  printf("%f\n",toc-tic);
  // Check answer
#pragma omp parallel for private(j,r)
  for (i=0; i<N; i++) {
    double uid = 0;
    for (j=0; j<N; j++) {
      double dx = xi[i] - xj[j];
      double dy = yi[i] - yj[j];
      r = sqrt(dx * dx + dy * dy);
      uid += qj[j] / r;
    }
    //printf("%d %lf %lf\n", i, ui[i], uid);
  }
  tic = get_time();
  printf("%f\n",tic-toc);
}
