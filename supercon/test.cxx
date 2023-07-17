#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  uint64_t N = 1 << 20;
  uint64_t *I = new uint64_t [N]; 
  float *X = new float [N];
  float *Y = new float [N];
  srand48(1);
  double tic = get_time();
  for (uint64_t i=0; i<N; i++) {
    I[i] = i;
    X[i] = drand48();
    Y[i] = drand48();
  }
  printf("N      : %llu\n",N);

  double toc = get_time();
  printf("%e s\n",toc-tic);

  delete I;
  delete X;
  delete Y;
  return 0;
}
