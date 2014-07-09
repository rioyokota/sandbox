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

inline void getIndex(Bodies &bodies, int level) {
  float d = 1.0 / (1 << level);
#pragma omp parallel for
  for( int b=0; b<int(bodies.size()); b++ ) {
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
    B->ICELL = id;
  }
}

void radixSort(Bodies &bodies, int **index) {
  const int n = bodies.size();
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int mask = stride - 1;
  int (*bucket2D)[stride] = new int [OMP_NUM_THREADS][stride]();
  int aMaxPerThread[OMP_NUM_THREADS] = {0};
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
  for( int b=0; b<n; b++ ) {
    int i = bodies[b].ICELL;
    index[0][b] = i;
    index[2][b] = b;
    index[4][b] = i;
    if( i > aMaxPerThread[omp_get_thread_num()] )
      aMaxPerThread[omp_get_thread_num()] = i;
  }
  int aMax = 0;
  for( int i=0; i<OMP_NUM_THREADS; i++ )
    if( aMaxPerThread[i] > aMax ) aMax = aMaxPerThread[i];
  while( aMax > 0 ) {
    int bucket[stride] = {0};
    for( int t=0; t<OMP_NUM_THREADS; t++ )
      for( int i=0; i<stride; i++ )
        bucket2D[t][i] = 0;
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      bucket2D[omp_get_thread_num()][index[0][i] & mask]++;
    for( int t=0; t<OMP_NUM_THREADS; t++ )
      for( int i=0; i<stride; i++ )
        bucket[i] += bucket2D[t][i];
    for( int i=1; i<stride; i++ )
      bucket[i] += bucket[i-1];
    for( int i=n-1; i>=0; i-- )
      index[3][i] = --bucket[index[0][i] & mask];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      index[1][index[3][i]] = index[2][i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      index[2][i] = index[1][i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      index[1][index[3][i]] = index[0][i];
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
    for( int i=0; i<n; i++ )
      index[0][i] = index[1][i] >> bitStride;
    aMax >>= bitStride;
  }
  delete[] bucket2D;
}

void permute(Bodies &bodies, int ** index) {
  const int n = bodies.size();
  Bodies buffer = bodies;
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
  for( int b=0; b<n; b++ )
    bodies[b] = buffer[index[2][b]];
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
  int **index2 = new int* [5];
  for( int i=0; i<5; i++ ) index2[i] = new int [numBodies];
  double tic, toc;
  tic = get_time();
  Bodies bodies(numBodies);
  Cells cells;
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    for( int d=0; d!=3; ++d ) B->X[d] = drand48();
  }
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  tic = get_time();
  getIndex(bodies,level);
  toc = get_time();
  std::cout << "mort : " << toc-tic << std::endl;

  tic = get_time();
  radixSort(bodies,index2);
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;

  tic = get_time();
  permute(bodies,index2);
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
