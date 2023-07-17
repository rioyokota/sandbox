#include <math.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <omp.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char* argv[]) {
  const uint64_t N = 1 << atoi(argv[1]);
  const int level = 11;
  const int threads = 48;
  const int ranking = 5;
  const int Nx = 1 << level;
  const uint64_t range = Nx * Nx;
  printf("N          : %llu\n",N);
  double tic = get_time();
  double *X = new double [N];
  double *Y = new double [N];
  uint64_t *key = new uint64_t [N]; 
  uint64_t *bucket = new uint64_t [range];
  uint64_t (*bucketPerThread)[range] = new uint64_t [threads][range]();
  uint64_t *offset = new uint64_t [range+1];
  uint64_t *permutation = new uint64_t [N]; 
  double *X2 = new double [N];
  double *Y2 = new double [N];
  double toc = get_time();
  printf("Alloc      : %e s\n",toc-tic);
  std::uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
  for (int ib=0; ib<threads; ib++) {
    std::mt19937 generator(ib);
    int begin = ib * (N / threads);
    int end = (ib + 1) * (N / threads);
    if(ib == threads-1) end = N > end ? N : end;
    for (uint64_t i=begin; i<end; i++) {
      X[i] = dis(generator);
      Y[i] = dis(generator);
    }
  }
  tic = get_time();
  printf("Init       : %e s\n",tic-toc);
#pragma omp parallel for
  for (uint64_t i=0; i<N; i++) {
    uint64_t ix = X[i] * Nx;
    uint64_t iy = Y[i] * Nx;
    uint64_t k = 0;
    for (int l=0; l<level; l++) {
       k |= (iy & (uint64_t)1 << l) <<  l;
       k |= (ix & (uint64_t)1 << l) << (l + 1);
    }
    key[i] = k;
  }
  toc = get_time();
  printf("Index      : %e s\n",toc-tic);
#pragma omp parallel
  {
    int t = omp_get_thread_num();
    for (uint64_t i=0; i<range; i++)
      bucketPerThread[t][i] = 0;
#pragma omp for
    for (uint64_t i=0; i<N; i++)
      bucketPerThread[t][key[i]]++;
#pragma omp barrier
#pragma omp single
    for (int t=1; t<threads; t++)
      for (uint64_t i=0; i<range; i++)
	bucketPerThread[t][i] += bucketPerThread[t-1][i];
#pragma omp single
    for (uint64_t i=0; i<range; i++)
      bucket[i] = bucketPerThread[threads-1][i];
#pragma omp single
    for (uint64_t i=1; i<range; i++)
      bucket[i] += bucket[i-1];
    offset[0] = 0;
#pragma omp for
    for (uint64_t i=0; i<range; i++)
      offset[i+1] = bucket[i];
#if 0
    t = omp_get_thread_num();
#pragma omp for
    for (int64_t i=N-1; i>=0; i--) {
      bucketPerThread[t][key[i]]--;
      uint64_t inew = bucketPerThread[t][key[i]];
      permutation[inew] = i;
    }
#else
#pragma omp single
    for (int64_t i=N-1; i>=0; i--) {
      bucket[key[i]]--;
      uint64_t inew = bucket[key[i]];
      permutation[inew] = i;
    }
#endif
  }
  tic = get_time();
  printf("Sort       : %e s\n",tic-toc);
#pragma omp parallel for
  for (uint64_t i=0; i<N; i++) {
    X2[i] = X[permutation[i]];
    Y2[i] = Y[permutation[i]];
    //printf("%llu %llu\n",i,key[permutation[i]]);
  }
  toc = get_time();
  printf("Permute    : %e s\n",toc-tic);
  uint64_t minI[ranking],minJ[ranking];
  double minD[ranking];
  for (int i=0; i<ranking; i++) {
    minI[i] = minJ[i] = 0;
    minD[i] = 1;
  }
#pragma omp parallel for
  for (uint64_t i=0; i<range; i++) {
    int ix = 0;
    int iy = 0;
    for (int l=0; l<level; l++) {
      iy |= (i & (uint64_t)1 <<  2 * l)      >>  l;
      ix |= (i & (uint64_t)1 << (2 * l + 1)) >> (l + 1);
    }
    int minjx = 0 > ix-1 ? 0 : ix-1;
    int maxjx = ix+1 < Nx-1 ? ix+1 : Nx-1;
    int minjy = 0 > iy-1 ? 0 : iy-1;
    int maxjy = iy+1 < Nx-1 ? iy+1 : Nx-1;
    for (int jx=minjx; jx<=maxjx; jx++) {
      for (int jy=minjy; jy<=maxjy; jy++) {
        uint64_t j = 0;
        for (int l=0; l<level; l++) {
           j |= (jy & (uint64_t)1 << l) <<  l;
           j |= (jx & (uint64_t)1 << l) << (l + 1);
        }
	if (j > i) break;
        for (int ii=offset[i]; ii<offset[i+1]; ii++) {
	  for (int jj=offset[j]; jj<offset[j+1]; jj++) {
            double dd = (X2[ii] - X2[jj]) * (X2[ii] - X2[jj]) + (Y2[ii] - Y2[jj]) * (Y2[ii] - Y2[jj]);
	    if (minD[ranking-1] > dd && ii < jj) {
	      int k = ranking-1;
	      while (k>0 && dd < minD[k-1]) {
                minI[k] = minI[k-1];
                minJ[k] = minJ[k-1];
                minD[k] = minD[k-1];
		k--;
	      }
              minI[k] = ii;
	      minJ[k] = jj;
	      minD[k] = dd;
	    }
	  }
	}
      }
    }
  }
  tic = get_time();
  printf("Search     : %e s\n",tic-toc);
  for (int i=0; i<ranking; i++) {
    minI[i] = permutation[minI[i]];
    minJ[i] = permutation[minJ[i]];
    printf("\n#%d closest\n",i+1);
    printf("Point I  : %llu %10.15e %10.15e\n",minI[i],X[minI[i]],Y[minI[i]]);
    printf("Point J  : %llu %10.15e %10.15e\n",minJ[i],X[minJ[i]],Y[minJ[i]]);
    printf("Distance : %10.15e\n",sqrt(minD[i]));
  }
  delete[] X;
  delete[] Y;
  delete[] key;
  delete[] bucket;
  delete[] bucketPerThread;
  delete[] offset;
  delete[] permutation;
  delete[] X2;
  delete[] Y2;
  return 0;
}
