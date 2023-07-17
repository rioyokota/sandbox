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
  const uint64_t N = 1 << 10;
  const int level = 2;
  const int Nx = 1 << level;
  const int range = Nx * Nx;
  printf("N          : %llu\n",N);
  double tic = get_time();
  float *X = new float [N];
  float *Y = new float [N];
  uint64_t *I = new uint64_t [N]; 
  uint64_t *key = new uint64_t [N]; 
  uint64_t *bucket = new uint64_t [range];
  uint64_t *permutation = new uint64_t [N]; 
  double toc = get_time();
  printf("Alloc      : %e s\n",toc-tic);
  srand48(1);
  for (uint64_t i=0; i<N; i++) {
    X[i] = drand48();
    Y[i] = drand48();
    I[i] = i;
  }
  tic = get_time();
  printf("Init       : %e s\n",tic-toc);
  for (uint64_t i=0; i<N; i++) {
    uint64_t jx = X[i] * Nx;
    uint64_t jy = Y[i] * Nx;
    uint64_t j = 0;
    for (int l=0; l<level; l++) {
       j |= (jy & (uint64_t)1 << l) <<  l;
       j |= (jx & (uint64_t)1 << l) << (l + 1);
    }
    key[i] = j;
  }
  toc = get_time();
  printf("Index      : %e s\n",toc-tic);
  for (uint64_t i=0; i<range; i++)
    bucket[i] = 0;
  for (uint64_t i=0; i<N; i++)
    bucket[key[i]]++;
  for (uint64_t i=1; i<range; i++)
    bucket[i] += bucket[i-1];
  for (int64_t i=N-1; i>=0; i--) {
    bucket[key[i]]--;
    uint64_t inew = bucket[key[i]];
    permutation[inew] = i;
  }
  tic = get_time();
  printf("Sort       : %e s\n",tic-toc);
  for (uint64_t i=0; i<N; i++) {
    printf("%llu %llu %llu\n",i,permutation[i],key[permutation[i]]);
  }
  delete X;
  delete Y;
  delete I;
  delete key;
  delete bucket;
  delete permutation;
  return 0;
}
