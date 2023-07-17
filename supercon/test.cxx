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

int main(int argc, char* argv[]) {
  const uint64_t N = 1 << atoi(argv[1]);
  const int level = atoi(argv[2]);
  const int Nx = 1 << level;
  const uint64_t range = Nx * Nx;
  printf("N          : %llu\n",N);
  double tic = get_time();
  double *X = new double [N];
  double *Y = new double [N];
  uint64_t *key = new uint64_t [N]; 
  uint64_t *bucket = new uint64_t [range];
  uint64_t *offset = new uint64_t [range+1];
  uint64_t *permutation = new uint64_t [N]; 
  double *X2 = new double [N];
  double *Y2 = new double [N];
  double toc = get_time();
  printf("Alloc      : %e s\n",toc-tic);
  srand48(1);
  for (uint64_t i=0; i<N; i++) {
    X[i] = drand48();
    Y[i] = drand48();
  }
  tic = get_time();
  printf("Init       : %e s\n",tic-toc);
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
  for (uint64_t i=0; i<range; i++)
    bucket[i] = 0;
  for (uint64_t i=0; i<N; i++)
    bucket[key[i]]++;
  for (uint64_t i=1; i<range; i++)
    bucket[i] += bucket[i-1];
  offset[0] = 0;
  for (uint64_t i=0; i<range; i++)
    offset[i+1] = bucket[i];
  for (int64_t i=N-1; i>=0; i--) {
    bucket[key[i]]--;
    uint64_t inew = bucket[key[i]];
    permutation[inew] = i;
  }
  tic = get_time();
  printf("Sort       : %e s\n",tic-toc);
  for (uint64_t i=0; i<N; i++) {
    X2[i] = X[permutation[i]];
    Y2[i] = Y[permutation[i]];
  }
  toc = get_time();
  printf("Permute    : %e s\n",toc-tic);
  uint64_t minI = 0;
  uint64_t minJ = 0;
  double minD2 = 2;
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
        for (int ii=offset[i]; ii<offset[i+1]; ii++) {
	  for (int jj=offset[j]; jj<offset[j+1]; jj++) {
            double D2 = (X2[ii] - X2[jj]) * (X2[ii] - X2[jj]) + (Y2[ii] - Y2[jj]) * (Y2[ii] - Y2[jj]);
	    if (minD2 > D2 && ii != jj) {
              minI = ii;
	      minJ = jj;
	      minD2 = D2;
	    }
	  }
	}
      }
    }
  }
  tic = get_time();
  printf("Search     : %e s\n",tic-toc);
  minI = permutation[minI];
  minJ = permutation[minJ];
  printf("%llu %e %e %llu %e %e %e\n",minI,X[minI],Y[minI],minJ,X[minJ],Y[minJ],minD2);
  delete X;
  delete Y;
  delete key;
  delete bucket;
  delete offset;
  delete permutation;
  delete X2;
  delete Y2;
  return 0;
}
