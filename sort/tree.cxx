#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include <vector>

#define OMP_NUM_THREADS 32

struct Body {
//  int IBODY;
//  int IRANK;
  unsigned long long ICELL;
  float X[3];
//  float SRC;
//  float TRG[4];
  bool operator<(const Body &rhs) const {
    return this->ICELL < rhs.ICELL;
  }
};
typedef std::vector<Body> Bodies;
typedef std::vector<Body>::iterator B_iter;

struct Cell {
  unsigned NCHILD;
  unsigned NCLEAF;
  unsigned NDLEAF;
  unsigned PARENT;
  unsigned CHILD;
  B_iter   LEAF;
  float    X[3];
  float    R;
  float    RCRIT;
//  float    M[55];
//  float    L[55];
};
typedef std::vector<Cell> Cells;
typedef std::vector<Cell>::iterator C_iter;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

inline void getKey(Bodies &bodies, int * key, int level) {
  float d = 1.0 / (1 << level);
#pragma omp parallel for
  for (int b=0; b<int(bodies.size()); b++) {
    B_iter B=bodies.begin()+b;
    int ix = B->X[0] / d;
    int iy = B->X[1] / d;
    int iz = B->X[2] / d;
    int id = 0;
    for( int l=0; l!=level; ++l ) {
      id += (ix & 1) << (3 * l);
      id += (iy & 1) << (3 * l + 1);
      id += (iz & 1) << (3 * l + 2);
      ix >>= 1;
      iy >>= 1;
      iz >>= 1;
    }
    key[b] = id;
    B->ICELL = id;
  }
}

void radixSort(int * key, int * value, int size) {
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
    for (int i=0; i<size; i++)
      if (key[i] > maxKeyPerThread[omp_get_thread_num()])
        maxKeyPerThread[omp_get_thread_num()] = key[i];
#pragma omp single
    for (int i=0; i<numThreads; i++)
      if (maxKeyPerThread[i] > maxKey) maxKey = maxKeyPerThread[i];
    while (maxKey > 0) {
      int bucket[stride] = {0};
#pragma omp single
      for (int t=0; t<numThreads; t++)
        for (int i=0; i<stride; i++)
          bucketPerThread[t][i] = 0;
#pragma omp for
      for (int i=0; i<size; i++)
        bucketPerThread[omp_get_thread_num()][key[i] & mask]++;
#pragma omp single
      {
	for (int t=0; t<numThreads; t++)
	  for (int i=0; i<stride; i++)
	    bucket[i] += bucketPerThread[t][i];
	for (int i=1; i<stride; i++)
	  bucket[i] += bucket[i-1];
	for (int i=size-1; i>=0; i--)
	  permutation[i] = --bucket[key[i] & mask];
      }
#pragma omp for
      for (int i=0; i<size; i++)
        buffer[permutation[i]] = value[i];
#pragma omp for
      for (int i=0; i<size; i++)
        value[i] = buffer[i];
#pragma omp for
      for (int i=0; i<size; i++)
        buffer[permutation[i]] = key[i];
#pragma omp for
      for (int i=0; i<size; i++)
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

void permute(Bodies & bodies, int * index) {
  const int n = bodies.size();
  Bodies buffer = bodies;
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
  for( int b=0; b<n; b++ )
    bodies[b] = buffer[index[b]];
}

void bodies2leafs(Bodies &bodies, Cells &cells, int level) {
  int I = -1;
  C_iter C;
  cells.reserve(1 << (3 * level));
  float d = 1.0 / (1 << level);
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    int IC = B->ICELL;
    int ix = B->X[0] / d;
    int iy = B->X[1] / d;
    int iz = B->X[2] / d;
    if( IC != I ) {
      Cell cell;
      cell.NCHILD = 0;
      cell.NCLEAF = 0;
      cell.NDLEAF = 0;
      cell.CHILD  = 0;
      cell.LEAF   = B;
      cell.X[0]   = d * (ix + .5);
      cell.X[1]   = d * (iy + .5);
      cell.X[2]   = d * (iz + .5);
      cell.R      = d * .5;
      cells.push_back(cell);
      C = cells.end()-1;
      I = IC;
    }
    C->NCLEAF++;
    C->NDLEAF++;
  }
}

void leafs2cells(Bodies &bodies, Cells &cells, int level) {
  int begin = 0, end = cells.size();
  float d = 1.0 / (1 << level);
  for( int l=1; l!=level; ++l ) {
    int div = (1 << (3 * l));
    d *= 2;
    int I = -1;
    int p = end - 1;
    for( int c=begin; c!=end; ++c ) {
      B_iter B = cells[c].LEAF;
      int IC = B->ICELL / div;
      int ix = B->X[0] / d;
      int iy = B->X[1] / d;
      int iz = B->X[2] / d;
      if( IC != I ) {
        Cell cell;
        cell.NCHILD = 0;
        cell.NCLEAF = 0;
        cell.NDLEAF = 0;
        cell.CHILD  = c - begin;
        cell.LEAF   = cells[c].LEAF;
        cell.X[0]   = d * (ix + .5);
        cell.X[1]   = d * (iy + .5);
        cell.X[2]   = d * (iz + .5);
        cell.R      = d * .5;
        cells.push_back(cell);
        p++;
        I = IC;
      }
      cells[p].NCHILD++;
      cells[p].NDLEAF += cells[c].NDLEAF;
      cells[c].PARENT = p;
    }
    begin = end;
    end = cells.size();
  }
}

int main() {
  const int numBodies = 10000000;
  const int level = 7;
  int * key = new int [numBodies];
  int * index = new int [numBodies];
  int **index2 = new int* [5];
  for( int i=0; i<5; i++ ) index2[i] = new int [numBodies];
  double tic, toc;
  tic = get_time();
  Bodies bodies(numBodies);
  Cells cells;
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  int b = 0;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, b++) {
    for (int d=0; d!=3; d++) B->X[d] = drand48();
    index[b] = b;
  }
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  tic = get_time();
  getKey(bodies, key, level);
  toc = get_time();
  std::cout << "mort : " << toc-tic << std::endl;

  tic = get_time();
  radixSort(key, index, numBodies);
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;

  tic = get_time();
  permute(bodies, index);
  toc = get_time();
  std::cout << "perm : " << toc-tic << std::endl;

  tic = get_time();
  bodies2leafs(bodies,cells,level);
  toc = get_time();
  std::cout << "leaf : " << toc-tic << std::endl;

  tic = get_time();
  leafs2cells(bodies,cells,level);
  toc = get_time();
  std::cout << "cell : " << toc-tic << std::endl;
}
