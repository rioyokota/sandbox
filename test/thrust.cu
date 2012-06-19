#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct Body {
  int ICELL;
  float X[3];
  float M;
  float P;
  float F[3];
};

struct Cell {
  int ICELL;
  int NCHILD;
  int NCLEAF;
  int NDLEAF;
  int CHILD;
  int LEAF;
  float X[3];
  float R;
  float M[7];
};

void sort(Body *bodiesDevc, int *icellDevc, int N) {
  thrust::device_ptr<int> icellThrust(icellDevc);
  thrust::device_ptr<Body> bodiesThrust(bodiesDevc);
  thrust::sort_by_key(icellThrust,icellThrust+N,bodiesThrust);
}

extern __global__ void initCells(Body *bodies, Cell *cells, int *icell, int *ileaf, int *nleaf,
                          const int NCELL, const int LEVEL);

void buildCell(Body *bodiesDevc, Cell *cellsDevc,
               int *icellDevc, int *ncellDevc,
               int *ileafDevc, int *nleafDevc,
               int &ntwig, int maxlevel, int N, int THREADS) {
  thrust::device_ptr<int> icellThrust(icellDevc);
  thrust::device_ptr<int> ncellThrust(ncellDevc);
  thrust::device_ptr<int> ileafThrust(ileafDevc);
  thrust::device_ptr<int> nleafThrust(nleafDevc);
  thrust::sequence(ileafThrust,ileafThrust+N);
  thrust::fill(nleafThrust,nleafThrust+N,1);
  thrust::reverse(icellThrust,icellThrust+N);
  thrust::inclusive_scan_by_key(icellThrust,icellThrust+N,nleafThrust,nleafThrust);
  thrust::reverse(icellThrust,icellThrust+N);
  thrust::reverse(nleafThrust,nleafThrust+N);
  thrust::copy(icellThrust,icellThrust+N,ncellThrust);
  thrust::unique_by_key(ncellThrust,ncellThrust+N,ileafThrust);
  thrust::pair<thrust::device_ptr<int>,thrust::device_ptr<int> > new_end;
  new_end = thrust::unique_by_key(icellThrust,icellThrust+N,nleafThrust);
  ntwig = new_end.first - icellThrust;
  int nblocks = (ntwig - 1) / THREADS + 1;
  initCells<<<nblocks,THREADS>>>(bodiesDevc,cellsDevc,icellDevc,ileafDevc,nleafDevc,ntwig,maxlevel);
}

extern __global__ void initCells(Body *bodies, Cell *cells, int *icell, int *ileaf, int *nleaf,
                                 int *ichild, int *nchild, const int NBASE, const int NCELL, const int LEVEL);

void buildTree(Body *bodiesDevc, Cell *cellsDevc,
               int *icellDevc, int *ncellDevc,
               int *ileafDevc, int *nleafDevc,
               int *ichildDevc, int *nchildDevc,
               int &ntwig, int &ncell, int maxlevel, int THREADS) {
  thrust::device_ptr<int> icellThrust(icellDevc);
  thrust::device_ptr<int> ncellThrust(ncellDevc);
  thrust::device_ptr<int> ileafThrust(ileafDevc);
  thrust::device_ptr<int> nleafThrust(nleafDevc);
  thrust::device_ptr<int> ichildThrust(ichildDevc);
  thrust::device_ptr<int> nchildThrust(nchildDevc);
  int n = ntwig;
  int nbase = 0;
  for( int level=maxlevel-1; level>=0; level-- ) {
    thrust::sequence(ichildThrust,ichildThrust+n,nbase);
    thrust::fill(nchildThrust,nchildThrust+n,1);
    thrust::reverse(icellThrust,icellThrust+n);
    thrust::reverse(nleafThrust,nleafThrust+n);
    thrust::inclusive_scan_by_key(icellThrust,icellThrust+n,nleafThrust,nleafThrust);
    thrust::inclusive_scan_by_key(icellThrust,icellThrust+n,nchildThrust,nchildThrust);
    thrust::reverse(icellThrust,icellThrust+n);
    thrust::reverse(nleafThrust,nleafThrust+n);
    thrust::reverse(nchildThrust,nchildThrust+n);
    thrust::copy(icellThrust,icellThrust+n,ncellThrust);
    thrust::unique_by_key(ncellThrust,ncellThrust+n,ileafThrust);
    thrust::copy(icellThrust,icellThrust+n,ncellThrust);
    thrust::unique_by_key(ncellThrust,ncellThrust+n,nleafThrust);
    thrust::copy(icellThrust,icellThrust+n,ncellThrust);
    thrust::unique_by_key(ncellThrust,ncellThrust+n,ichildThrust);
    thrust::pair<thrust::device_ptr<int>,thrust::device_ptr<int> > new_end;
    new_end = thrust::unique_by_key(icellThrust,icellThrust+n,nchildThrust);
    nbase += n;
    n = new_end.first - icellThrust;
    int nblocks = (n - 1) / THREADS + 1;
    initCells<<<nblocks,THREADS>>>(bodiesDevc,cellsDevc,icellDevc,ileafDevc,nleafDevc,
                                   ichildDevc,nchildDevc,nbase,n,level);
  }
  ncell = nbase+n;
}

int reduce(int *NDevc, int ntwig) {
  thrust::device_ptr<int> NThrust(NDevc);
  return thrust::reduce(NThrust,NThrust+ntwig);
}
