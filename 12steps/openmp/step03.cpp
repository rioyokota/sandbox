#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
int main(int argc, char ** argv) {
  struct timeval tic, toc;
  int n = atoi(argv[1]);
  double * a = new double [n];
  double * b = new double [n];
  double * y = new double [n];
  double * z = new double [n];
  gettimeofday(&tic, NULL);
#pragma omp parallel
  {
#pragma omp for nowait
    for (int i=1; i<n; i++)
      b[i] = (a[i] + a[i-1]) / 2.0;
#pragma omp for nowait
    for (int i=0; i<n; i++)
      y[i] = sqrt(z[i]);
  }
  gettimeofday(&toc, NULL);
  printf("%lf s\n",toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6);
  delete[] a;
  delete[] b;
  delete[] y;
  delete[] z;
}
