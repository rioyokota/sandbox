#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void radixsort(int *a, int *b, int n) {
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int stride1 = stride - 1;
  int amax = 0;
  for( int i=0; i<n; i++ )
    if( a[i] > amax ) amax = a[i];
  int exp = 0;
  while( (amax >> exp) > 0 ) {
    int bucket[stride] = {0};
    for( int i=0; i<n; i++ )
      bucket[(a[i] >> exp) & stride1]++;
    for( int i=1; i<stride; i++ )
      bucket[i] += bucket[i-1];
    for( int i=n-1; i>=0; i-- )
      b[--bucket[(a[i] >> exp) & stride1]] = a[i];
    for( int i=0; i<n; i++ )
      a[i] = b[i];
    exp += bitStride;
  }
}
 
 
int main()
{
  const int N = 10000000;
  int *a = new int [N];
  int *b = new int [N];
  for( int i=0; i<N; i++ )
    a[i] = N / 10. * rand() / RAND_MAX;
  double tic = get_time();
  radixsort(a,b,N);
  double toc = get_time();
  printf("N = %d, %lf s\n",N,toc-tic);
  for( int i=1; i<N; i++ )
    assert( a[i] >= a[i-1] );
  delete[] a;
  delete[] b;
}
