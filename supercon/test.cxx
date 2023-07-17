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
  const int N = 1 << atoi(argv[1]);
  const int level = atoi(argv[2]);
  const int threads = 48;
  const int ranking = 10000;
  const int Nx = 1 << level;
  const int range = Nx * Nx;
  printf("N          : %d\n",N);
  double tic = get_time();
  double *X = new double [N];
  double *Y = new double [N];
  int *key = new int [N]; 
  int *bucket = new int [range];
  int **bucketPerThread = new int* [threads];
  for (int i=0; i<threads; i++)
    bucketPerThread[i] = new int [range];
  int *offset = new int [range+1];
  int *permutation = new int [N]; 
  double *X2 = new double [N];
  double *Y2 = new double [N];
  double memory = (double)N * 5 * 8 + (double)range * (threads + 2) * 4;
  printf("Memory     : %e GB\n",memory/1e9);
  double toc = get_time();
  printf("Alloc      : %e s\n",toc-tic);
  std::uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
  for (int ib=0; ib<threads; ib++) {
    std::mt19937 generator(ib);
    int begin = ib * (N / threads);
    int end = (ib + 1) * (N / threads);
    if(ib == threads-1) end = N > end ? N : end;
    for (int i=begin; i<end; i++) {
      X[i] = dis(generator);
      Y[i] = dis(generator);
    }
  }
  tic = get_time();
  printf("Init       : %e s\n",tic-toc);
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
    int t = omp_get_thread_num();
    for (int i=0; i<range; i++)
      bucketPerThread[t][i] = 0;
#pragma omp for
    for (int i=0; i<N; i++)
      bucketPerThread[t][key[i]]++;
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
    t = omp_get_thread_num();
#pragma omp for
    for (int64_t i=0; i<N; i++) {
      bucketPerThread[t][key[i]]--;
      int inew = offset[key[i]] + bucketPerThread[t][key[i]];
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
  int minI[ranking],minJ[ranking];
  double minD[ranking];
  for (int i=0; i<ranking; i++) {
    minI[i] = minJ[i] = 0;
    minD[i] = 1;
  }
#pragma omp parallel for
  for (int i=0; i<range; i++) {
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
    if (i==0 || i==9 || i==99 || i==999 || i==9999) {
      printf("\n#%d closest\n",i+1);
      printf("Point I  : %d %10.15e %10.15e\n",minI[i],X[minI[i]],Y[minI[i]]);
      printf("Point J  : %d %10.15e %10.15e\n",minJ[i],X[minJ[i]],Y[minJ[i]]);
      printf("Distance : %10.15e\n",sqrt(minD[i]));
    }
  }
  delete[] X;
  delete[] Y;
  delete[] key;
  delete[] bucket;
  for (int i=0; i<threads; i++)
    delete[] bucketPerThread[i];
  delete[] bucketPerThread;
  delete[] offset;
  delete[] permutation;
  delete[] X2;
  delete[] Y2;
  return 0;
}
