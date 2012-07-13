#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>

#define OMP_NUM_THREADS 12

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void radixsort(int *a, int *b, int *p, int *q, int n) {
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int mask = stride - 1;
  int (*bucket2D)[stride] = new int [OMP_NUM_THREADS][stride]();
  int amax = 0;
  for( int i=0; i<n; i++ )
    if( a[i] > amax ) amax = a[i];
  while( amax > 0 ) {
    int bucket[stride] = {0};
    for( int t=0; t<OMP_NUM_THREADS; t++ )
      for( int i=0; i<stride; i++ )
        bucket2D[t][i] = 0;
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      bucket2D[omp_get_thread_num()][a[i] & mask]++;
    for( int t=0; t<OMP_NUM_THREADS; t++ )
      for( int i=0; i<stride; i++ )
        bucket[i] += bucket2D[t][i];
    for( int i=1; i<stride; i++ )
      bucket[i] += bucket[i-1];
    for( int i=n-1; i>=0; i-- )
      q[i] = --bucket[a[i] & mask];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      b[q[i]] = p[i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      p[i] = b[i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      b[q[i]] = a[i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      a[i] = b[i] >> bitStride;
    amax >>= bitStride;
  }
  delete[] bucket2D;
}
 
 
int main()
{
  const int N = 10000000;
  int *a = new int [N];
  int *b = new int [N];
  int *c = new int [N];
  int *p = new int [N];
  int *q = new int [N];
  for( int i=0; i<N; i++ ) {
    a[i] = N / 10. * rand() / RAND_MAX;
    c[i] = a[i];
    p[i] = i;
  }
  double tic = get_time();
  radixsort(a,b,p,q,N);
  double toc = get_time();
  printf("N = %d, %lf s\n",N,toc-tic);
  for( int i=0; i<N; i++ )
    a[i] = c[p[i]];
  for( int i=1; i<N; i++ )
    assert( a[i] >= a[i-1] );
  delete[] a;
  delete[] b;
  delete[] c;
  delete[] p;
  delete[] q;
}
