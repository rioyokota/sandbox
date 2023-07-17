#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  int N = 1 << 16;
  float OPS = 20. * N * N * 1e-9;
  float * x = (float*) malloc(N * sizeof(float));
  float * y = (float*) malloc(N * sizeof(float));
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
  }
  printf("N      : %d\n",N);

  double tic = get_time();
  double toc = get_time();
  printf("No SIMD: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

  free(x);
  free(y);
  return 0;
}
