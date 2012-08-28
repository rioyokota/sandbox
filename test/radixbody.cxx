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

struct Body {
  int  IBODY;
  int  IPROC;
  int  ICELL;
  float X[3];
  float SRC;
  float TRG[4];
};

void radixsort(int *a, int *b, int *p, int *q, int n) {
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int mask = stride - 1;
  int (*bucketPerThread)[stride] = new int [OMP_NUM_THREADS][stride]();
  int aMaxPerThread[OMP_NUM_THREADS] = {0};
  int aMax = 0;
#pragma omp parallel
  {
#pragma omp for
    for( int i=0; i<n; i++ )
      if( a[i] > aMaxPerThread[omp_get_thread_num()] )
        aMaxPerThread[omp_get_thread_num()] = a[i];
#pragma omp single
    for( int i=0; i<OMP_NUM_THREADS; i++ )
      if( aMaxPerThread[i] > aMax ) aMax = aMaxPerThread[i];
    while( aMax > 0 ) {
      int bucket[stride] = {0};
#pragma omp single
      for( int t=0; t<OMP_NUM_THREADS; t++ )
        for( int i=0; i<stride; i++ )
          bucketPerThread[t][i] = 0;
#pragma omp for
      for( int i=0; i<n; i++ )
        bucketPerThread[omp_get_thread_num()][a[i] & mask]++;
#pragma omp single
      {
      for( int t=0; t<OMP_NUM_THREADS; t++ )
        for( int i=0; i<stride; i++ )
          bucket[i] += bucketPerThread[t][i];
      for( int i=1; i<stride; i++ )
        bucket[i] += bucket[i-1];
      for( int i=n-1; i>=0; i-- )
        q[i] = --bucket[a[i] & mask];
      }
#pragma omp for
      for( int i=0; i<n; i++ )
        b[q[i]] = p[i];
#pragma omp for
      for( int i=0; i<n; i++ )
        p[i] = b[i];
#pragma omp for
      for( int i=0; i<n; i++ )
        b[q[i]] = a[i];
#pragma omp for
      for( int i=0; i<n; i++ )
        a[i] = b[i] >> bitStride;
#pragma omp single
      aMax >>= bitStride;
    }
  }
}
 
 
int main()
{
  const int N = 10000000;
  Body *bodies = new Body [N];
  Body *buffer = new Body [N];
  int *a = new int [N];
  int *b = new int [N];
  int *p = new int [N];
  int *q = new int [N];
  for( int i=0; i<N; i++ ) {
    bodies[i].ICELL = N / 10. * drand48();
    buffer[i] = bodies[i];
    a[i] = bodies[i].ICELL;
    p[i] = i;
  }
  printf("N       : %d\n",N);
  double tic = get_time();
  radixsort(a,b,p,q,N);
  double toc = get_time();
  printf("Sort    : %lf s\n",toc-tic);
  tic = get_time();
#pragma omp parallel for
  for( int i=0; i<N; i++ ) {
    bodies[i] = buffer[p[i]];
  }
  toc = get_time();
  printf("Permute : %lf s\n",toc-tic);
  for( int i=1; i<N; i++ )
    assert( bodies[i].ICELL >= bodies[i-1].ICELL );
  delete[] bodies;
  delete[] buffer;
  delete[] a;
  delete[] b;
  delete[] p;
  delete[] q;
}
