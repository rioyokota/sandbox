#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

struct Body {
  int  IBODY;
  int  IPROC;
  int  ISORT;
  float X[3];
  float SRC;
  float TRG[4];
};

void radixSort(int *key, int *value, int size) {
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int mask = stride - 1;
  int numThreads;
  int maxKey = 0;
  int (*bucketPerThread)[stride];
  int * maxKeyPerThread;
  int * buffer = new int [size];
  int * permutation = new int [size];
#pragma omp parallel
  {
    numThreads = omp_get_num_threads();
#pragma omp single
    {
      bucketPerThread = new int [numThreads][stride]();
      maxKeyPerThread = new int [numThreads];
      for (int i=0; i<numThreads; i++)
	maxKeyPerThread[i] = 0;
    }
#pragma omp for
    for( int i=0; i<size; i++ )
      if( key[i] > maxKeyPerThread[omp_get_thread_num()] )
        maxKeyPerThread[omp_get_thread_num()] = key[i];
#pragma omp single
    for( int i=0; i<numThreads; i++ )
      if( maxKeyPerThread[i] > maxKey ) maxKey = maxKeyPerThread[i];
    while( maxKey > 0 ) {
      int bucket[stride] = {0};
#pragma omp single
      for( int t=0; t<numThreads; t++ )
        for( int i=0; i<stride; i++ )
          bucketPerThread[t][i] = 0;
#pragma omp for
      for( int i=0; i<size; i++ )
        bucketPerThread[omp_get_thread_num()][key[i] & mask]++;
#pragma omp single
      {
	for( int t=0; t<numThreads; t++ )
	  for( int i=0; i<stride; i++ )
	    bucket[i] += bucketPerThread[t][i];
	for( int i=1; i<stride; i++ )
	  bucket[i] += bucket[i-1];
	for( int i=size-1; i>=0; i-- )
	  permutation[i] = --bucket[key[i] & mask];
      }
#pragma omp for
      for( int i=0; i<size; i++ )
        buffer[permutation[i]] = value[i];
#pragma omp for
      for( int i=0; i<size; i++ )
        value[i] = buffer[i];
#pragma omp for
      for( int i=0; i<size; i++ )
        buffer[permutation[i]] = key[i];
#pragma omp for
      for( int i=0; i<size; i++ )
        key[i] = buffer[i] >> bitStride;
#pragma omp single
      maxKey >>= bitStride;
    }
  }
  delete[] bucketPerThread;
  delete[] maxKeyPerThread;
  delete[] buffer;
  delete[] permutation;
}
 
 
int main()
{
  const int N = 10000000;
  Body *bodies = new Body [N];
  Body *buffer = new Body [N];
  int *key = new int [N];
  int *index = new int [N];
  for( int i=0; i<N; i++ ) {
    bodies[i].ISORT = N / 10. * drand48();
    buffer[i] = bodies[i];
    key[i] = bodies[i].ISORT;
    index[i] = i;
  }
  printf("N       : %d\n",N);
  double tic = get_time();
  radixSort(key,index,N);
  double toc = get_time();
  printf("Sort    : %lf s\n",toc-tic);
  tic = get_time();
#pragma omp parallel for
  for( int i=0; i<N; i++ ) {
    bodies[i] = buffer[index[i]];
  }
  toc = get_time();
  printf("Permute : %lf s\n",toc-tic);
  for( int i=1; i<N; i++ )
    assert( bodies[i].ISORT >= bodies[i-1].ISORT );
  delete[] bodies;
  delete[] buffer;
  delete[] key;
  delete[] index;
}
