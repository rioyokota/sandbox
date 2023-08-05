#include <sc_header.hpp>
#include <vector>

int main(int argc, char* argv[]) {
  const int N = atoi(argv[1]);
  printf("N          : %d\n",N);
  double tic = sc::get_time();
  sc::input(N);
  double toc = sc::get_time();
  printf("Init       : %e s\n",toc-tic);

  const int level = atoi(argv[2]);
  const int Nx = 1 << level;
  const int range = Nx * Nx;
  int *key = new int [N]; 
  int *bucket = new int [range];
  int **bucketPerThread = new int* [sc::maxThreads];
  for (int i=0; i<sc::maxThreads; i++)
    bucketPerThread[i] = new int [range];
  int *offset = new int [range+1];
  int *permutation = new int [N]; 
  double *X2 = new double [N];
  double *Y2 = new double [N];
  tic = sc::get_time();
  printf("Alloc      : %e s\n",tic-toc);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    int ix = sc::X[i] * Nx;
    int iy = sc::Y[i] * Nx;
    int k = 0;
    for (int l=0; l<level; l++) {
       k |= (iy & 1 << l) <<  l;
       k |= (ix & 1 << l) << (l + 1);
    }
    key[i] = k;
  }
  toc = sc::get_time();
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
  tic = sc::get_time();
  printf("Sort       : %e s\n",tic-toc);
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    X2[i] = sc::X[permutation[i]];
    Y2[i] = sc::Y[permutation[i]];
  }
  toc = sc::get_time();
  printf("Permute    : %e s\n",toc-tic);
  int minI[sc::ranking*sc::maxThreads],minJ[sc::ranking*sc::maxThreads];
  double minD[sc::ranking*sc::maxThreads];
  for (int i=0; i<sc::ranking*sc::maxThreads; i++) {
    minI[i] = minJ[i] = 0;
    minD[i] = 1;
  }
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int threadOffset = sc::ranking*omp_get_thread_num();
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
	    if (minD[sc::ranking-1 + threadOffset] > dd && ii < jj) {
	      int k = sc::ranking-1;
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
  tic = sc::get_time();
  printf("Search     : %e s\n",tic-toc);
  std::vector<int> index(sc::ranking*sc::maxThreads);
  for (int i=0; i<sc::ranking*sc::maxThreads; i++)
    index[i] = i;
  sort(index.begin(), index.end(),
    [&](const int& a, const int& b) {
      return (minD[a] < minD[b]);
    }
  );
  toc = sc::get_time();
  printf("Sort2      : %e s\n",toc-tic);
  for (int i=0; i<sc::ranking; i++) {
    int ii = index[i];
    sc::pairs[i].i = permutation[minI[ii]];
    sc::pairs[i].j = permutation[minJ[ii]];
    sc::pairs[i].dist2 = minD[ii];
  }
  delete[] key;
  delete[] bucket;
  for (int i=0; i<sc::maxThreads; i++)
    delete[] bucketPerThread[i];
  delete[] bucketPerThread;
  delete[] offset;
  delete[] permutation;
  delete[] X2;
  delete[] Y2;

  sc::output();
  sc::finalize();
  return 0;
}
