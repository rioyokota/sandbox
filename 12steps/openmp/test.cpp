#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

int main() {
  int nx = 400;
  int ny = 400;
  double dx = 0.1;
  double dy = 0.1;
  double ** a = new double * [ny];
  double ** b = new double * [ny];
  for (int j=0; j<ny; j++) {
    a[j] = new double [nx];
    b[j] = new double [nx];
    for (int i=0; i<nx; i++) {
      a[j][i] = drand48();
      b[j][i] = drand48();
    }
  }
  double tic = get_time();
  for (int it=0; it<500; it++) {
#pragma omp parallel for
    for (int i=1; i<nx-1; i++) {
      for (int j=1; j<ny-1; j++) {
        a[j][i] = ((b[j][i+1] + b[j][i-1]) * dy * dy + (b[j+1][i] + b[j-1][i]) * dx * dx) /
          (2 * (dx * dx + dy * dy)) - dx * dx * dy * dy / (2 * (dx * dx + dy * dy));
      }
    }
  }
  double toc = get_time();
  printf("%lf\n",toc-tic);
  for (int j=0; j<ny; j++) {
    delete[] a[j];
    delete[] b[j];
  }
  delete[] a;
  delete[] b;
}
