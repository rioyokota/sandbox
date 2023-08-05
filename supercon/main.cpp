#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>
#include <sys/time.h>
#include <omp.h>

  double get_time() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec+tv.tv_usec*1e-6);
  }

namespace sc {
  constexpr int maxThreads = 48;
  double *X, *Y;

  void input(int N) {
    X = new double [N];
    Y = new double [N];
    std::uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for (int ib=0; ib<maxThreads; ib++) {
      std::mt19937 generator(ib);
      int begin = ib * (N / maxThreads);
      int end = (ib + 1) * (N / maxThreads);
      if(ib == maxThreads-1) end = N > end ? N : end;
      for (int i=begin; i<end; i++) {
        X[i] = dis(generator);
        Y[i] = dis(generator);
      }
    }
  }

  void finalize() {
    delete[] X;
    delete[] Y;    
  }
};

using namespace sc;

int main(int argc, char* argv[]) {
  const int N = atoi(argv[1]);
  double tic = get_time();
  sc::input(N);
  double toc = get_time();
  printf("Init       : %e s\n",toc-tic);

  const int level = atoi(argv[2]);
  const int ranking = 1000;
  const int Nx = 1 << level;
  const int range = Nx * Nx;
  printf("N          : %d\n",N);
  int *key = new int [N]; 
  int *bucket = new int [range];
  int **bucketPerThread = new int* [maxThreads];
  for (int i=0; i<maxThreads; i++)
    bucketPerThread[i] = new int [range];
  int *offset = new int [range+1];
  int *permutation = new int [N]; 
  double *X2 = new double [N];
  double *Y2 = new double [N];
  double memory = (double)N * 5 * 8 + (double)range * (maxThreads + 2) * 4;
  printf("Memory     : %e GB\n",memory/1e9);
  tic = get_time();
  printf("Alloc      : %e s\n",tic-toc);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    int ix = X[i] * Nx;
    int iy = Y[i] * Nx;
    int k = 0;
    for (int l=0; l<level; l++) {
       k |= (iy & 1 << l) <<  l;
       k |= (ix & 1 << l) << (l + 1);
    }
    key[i] = k;
  }
  toc = get_time();
  printf("Index      : %e s\n",toc-tic);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int threads = omp_get_num_threads();
    for (int i=0; i<range; i++)
      bucketPerThread[tid][i] = 0;
#pragma omp for
    for (int i=0; i<N; i++)
      bucketPerThread[tid][key[i]]++;
#pragma omp single
    for (int t=1; t<threads; t++)
      for (int i=0; i<range; i++)
	bucketPerThread[t][i] += bucketPerThread[t-1][i];
#pragma omp for
    for (int i=0; i<range; i++)
      bucket[i] = bucketPerThread[threads-1][i];
#pragma omp single
    for (int i=1; i<range; i++)
      bucket[i] += bucket[i-1];
    offset[0] = 0;
#pragma omp for
    for (int i=0; i<range; i++)
      offset[i+1] = bucket[i];
#pragma omp for
    for (int i=0; i<N; i++) {
      bucketPerThread[tid][key[i]]--;
      int inew = offset[key[i]] + bucketPerThread[tid][key[i]];
      permutation[inew] = i;
    }
  }
  tic = get_time();
  printf("Sort       : %e s\n",tic-toc);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    X2[i] = X[permutation[i]];
    Y2[i] = Y[permutation[i]];
  }
  toc = get_time();
  printf("Permute    : %e s\n",toc-tic);
  int minI[ranking*maxThreads],minJ[ranking*maxThreads];
  double minD[ranking*maxThreads];
  for (int i=0; i<ranking*maxThreads; i++) {
    minI[i] = minJ[i] = 0;
    minD[i] = 1;
  }
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int threadOffset = ranking*omp_get_thread_num();
    int ix = 0;
    int iy = 0;
    for (int l=0; l<level; l++) {
      iy |= (i & 1 <<  2 * l)      >>  l;
      ix |= (i & 1 << (2 * l + 1)) >> (l + 1);
    }
    int minjx = 0 > ix-1 ? 0 : ix-1;
    int maxjx = ix+1 < Nx-1 ? ix+1 : Nx-1;
    int minjy = 0 > iy-1 ? 0 : iy-1;
    int maxjy = iy+1 < Nx-1 ? iy+1 : Nx-1;
    for (int jx=minjx; jx<=maxjx; jx++) {
      for (int jy=minjy; jy<=maxjy; jy++) {
        int j = 0;
        for (int l=0; l<level; l++) {
           j |= (jy & 1 << l) <<  l;
           j |= (jx & 1 << l) << (l + 1);
        }
	if (j > i) break;
        for (int ii=offset[i]; ii<offset[i+1]; ii++) {
	  for (int jj=offset[j]; jj<offset[j+1]; jj++) {
            double dd = (X2[ii] - X2[jj]) * (X2[ii] - X2[jj]) + (Y2[ii] - Y2[jj]) * (Y2[ii] - Y2[jj]);
	    if (minD[ranking-1 + threadOffset] > dd && ii < jj) {
	      int k = ranking-1;
	      while (k>0 && dd < minD[k-1 + threadOffset]) {
                minI[k + threadOffset] = minI[k-1 + threadOffset];
                minJ[k + threadOffset] = minJ[k-1 + threadOffset];
                minD[k + threadOffset] = minD[k-1 + threadOffset];
		k--;
	      }
              minI[k + threadOffset] = ii;
	      minJ[k + threadOffset] = jj;
	      minD[k + threadOffset] = dd;
	    }
	  }
	}
      }
    }
  }
  tic = get_time();
  printf("Search     : %e s\n",tic-toc);
  std::vector<int> index(ranking*maxThreads);
  for (int i=0; i<ranking*maxThreads; i++)
    index[i] = i;
  sort(index.begin(), index.end(),
    [&](const int& a, const int& b) {
      return (minD[a] < minD[b]);
    }
  );
  toc = get_time();
  printf("Sort2      : %e s\n",toc-tic);
  for (int i=0; i<ranking; i++) {
    int ii = index[i];
    int minIp = permutation[minI[ii]];
    int minJp = permutation[minJ[ii]];
    if (i==0 || i==9 || i==99 || i==999 || i==9999) {
      printf("\n#%d closest\n",i+1);
      printf("Point I  : %d %10.15e %10.15e\n",minIp,X[minIp],Y[minIp]);
      printf("Point J  : %d %10.15e %10.15e\n",minJp,X[minJp],Y[minJp]);
      printf("Distance : %10.15e\n",sqrt(minD[ii]));
    }
  }
  delete[] key;
  delete[] bucket;
  for (int i=0; i<maxThreads; i++)
    delete[] bucketPerThread[i];
  delete[] bucketPerThread;
  delete[] offset;
  delete[] permutation;
  delete[] X2;
  delete[] Y2;
  sc::finalize();
  return 0;
}
