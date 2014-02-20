#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

const int THREADS = 64;
const int N = THREADS * 20000;
const int NCRIT = 10;
const int MAXLEVEL = N >= NCRIT ? 1 + int(log(N / NCRIT)/M_LN2/3) : 0;
const float THETA = 0.75;
const float EPS2 = 0.00001;
const float X0 = .5;
const float R0 = .5;

double get_time() {
  struct timeval tv;
  cudaThreadSynchronize();
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

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

__global__ void getIndex(Body *bodies, int *icell, const int MAXLEVEL) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  float diameter = 2 * R0 / (1 << MAXLEVEL);
  int ix = (bodies[i].X[0] + R0 - X0) / diameter;
  int iy = (bodies[i].X[1] + R0 - X0) / diameter;
  int iz = (bodies[i].X[2] + R0 - X0) / diameter;
  int ICELL = 0;
  for( int l=0; l!=MAXLEVEL; ++l ) {
    ICELL += (ix & 1) << (3 * l);
    ICELL += (iy & 1) << (3 * l + 1);
    ICELL += (iz & 1) << (3 * l + 2);
    ix >>= 1;
    iy >>= 1;
    iz >>= 1;
  }
  icell[i] = ICELL;
  bodies[i].ICELL = ICELL;
}

__device__ float getBmax(const float *X, const Cell &cell) {
  float dx = cell.R + fabs(X[0] - cell.X[0]);
  float dy = cell.R + fabs(X[1] - cell.X[1]);
  float dz = cell.R + fabs(X[2] - cell.X[2]);
  return sqrtf( dx*dx + dy*dy + dz*dz );
}

__device__ void P2M(Body *bodies, Cell &cell) {
  float M = 0;
  float X[3] = {0,0,0};
  for( int l=0; l<cell.NCLEAF; l++ ) {
    Body leaf = bodies[cell.LEAF+l];
    M += leaf.M;
    X[0] += leaf.X[0] * leaf.M;
    X[1] += leaf.X[1] * leaf.M;
    X[2] += leaf.X[2] * leaf.M;
  }
  X[0] /= M;
  X[1] /= M;
  X[2] /= M;
  for( int l=0; l<cell.NCLEAF; l++ ) {
    Body leaf = bodies[cell.LEAF+l];
    float dx = X[0]-leaf.X[0];
    float dy = X[1]-leaf.X[1];
    float dz = X[2]-leaf.X[2];
    cell.M[0] += leaf.M;
    cell.M[1] += leaf.M * dx * dx / 2;
    cell.M[2] += leaf.M * dx * dy / 2;
    cell.M[3] += leaf.M * dx * dz / 2;
    cell.M[4] += leaf.M * dy * dy / 2;
    cell.M[5] += leaf.M * dy * dz / 2;
    cell.M[6] += leaf.M * dz * dz / 2;
  }
  cell.R = getBmax(X,cell);
  cell.X[0] = X[0];
  cell.X[1] = X[1];
  cell.X[2] = X[2];
}

__global__ void initCells(Body *bodies, Cell *cells, int *icell, int *ileaf, int *nleaf,
                          const int NCELL, const int LEVEL) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  int leaf = ileaf[i];
  if( i >= NCELL ) return;
  Cell cell;
  cell.ICELL  = icell[i];
  cell.LEAF   = leaf;
  cell.NCLEAF = nleaf[i];
  cell.NDLEAF = nleaf[i];
  cell.CHILD  = 0;
  cell.NCHILD = 0;
  float diameter = 2 * R0 / (1 << LEVEL);
  int ix = (bodies[leaf].X[0] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  int iy = (bodies[leaf].X[1] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  int iz = (bodies[leaf].X[2] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  cell.X[0] = diameter * (ix + .5) + X0 - R0;
  cell.X[1] = diameter * (iy + .5) + X0 - R0;
  cell.X[2] = diameter * (iz + .5) + X0 - R0;
  cell.R    = diameter * .5;
  for( int i=0; i<7; i++ ) cell.M[i] = 0;
  P2M(bodies,cell);
  cells[i] = cell;
  icell[i] /= 8;
}

extern void sort(Body *bodiesDevc, int *icellDevc, int N);

extern void buildCell(Body *bodiesDevc, Cell *cellsDevc,
                      int *icellDevc, int *ncellDevc,
                      int *ileafDevc, int *nleafDevc,
                      int &ntwig, int maxlevel, int N, int THREADS);

__device__ void M2M(Cell *cells, Cell &cell) {
  float M = 0;
  float X[3] = {0,0,0};
  for( int c=0; c<cell.NCHILD; c++ ) {
    Cell child = cells[cell.CHILD+c];
    M += fabs(child.M[0]);
    X[0] += child.X[0] * fabs(child.M[0]);
    X[1] += child.X[1] * fabs(child.M[0]);
    X[2] += child.X[2] * fabs(child.M[0]);
  }
  X[0] /= M;
  X[1] /= M;
  X[2] /= M;
  for( int c=0; c<cell.NCHILD; c++ ) {
    Cell child = cells[cell.CHILD+c];
    float dx = X[0] - child.X[0];
    float dy = X[1] - child.X[1];
    float dz = X[2] - child.X[2];
    cell.M[0] += child.M[0];
    cell.M[1] += child.M[1] + dx * dx * child.M[0] / 2;
    cell.M[2] += child.M[2];
    cell.M[3] += child.M[3];
    cell.M[4] += child.M[4] + dy * dy * child.M[0] / 2;
    cell.M[5] += child.M[5];
    cell.M[6] += child.M[6] + dz * dz * child.M[0] / 2;
  }
  cell.R = getBmax(X,cell);
  cell.X[0] = X[0];
  cell.X[1] = X[1];
  cell.X[2] = X[2];
}

__global__ void initCells(Body *bodies, Cell *cells, int *icell, int *ileaf, int *nleaf, int *ichild, int *nchild,
                          const int NBASE, const int NCELL, const int LEVEL) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  int leaf = ileaf[i];
  if( i >= NCELL ) return;
  Cell cell;
  cell.ICELL  = icell[i];
  cell.LEAF   = leaf;
  cell.NCLEAF = 0;
  cell.NDLEAF = nleaf[i];
  cell.CHILD  = ichild[i];
  cell.NCHILD = nchild[i];
  float diameter = 2 * R0 / (1 << LEVEL);
  int ix = (bodies[leaf].X[0] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  int iy = (bodies[leaf].X[1] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  int iz = (bodies[leaf].X[2] + R0 - X0) / diameter; // Calculate this from icell and get rid of bodies?
  cell.X[0] = diameter * (ix + .5) + X0 - R0;
  cell.X[1] = diameter * (iy + .5) + X0 - R0;
  cell.X[2] = diameter * (iz + .5) + X0 - R0;
  cell.R    = diameter * .5;
  for( int i=0; i<7; i++ ) cell.M[i] = 0;
  M2M(cells,cell);
  cells[NBASE+i] = cell;
  icell[i] /= 8;
}

extern void buildTree(Body *bodiesDevc, Cell *cellsDevc,
                      int *icellDevc, int *ncellDevc,
                      int *ileafDevc, int *nleafDevc,
                      int *ichildDevc, int *nchildDevc,
                      int &ntwig, int &ncell, int maxlevel, int THREADS);

__global__ void direct(Body *bodies) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  float3 pos = {bodies[i].X[0],bodies[i].X[1],bodies[i].X[2]};
  float4 target = {0,0,0,0};
  __shared__ float4 source[THREADS];
  for ( int jb=0; jb<N/THREADS; jb++ ) {
    __syncthreads();
    source[threadIdx.x].x = bodies[jb*THREADS+threadIdx.x].X[0];
    source[threadIdx.x].y = bodies[jb*THREADS+threadIdx.x].X[1];
    source[threadIdx.x].z = bodies[jb*THREADS+threadIdx.x].X[2];
    source[threadIdx.x].w = bodies[jb*THREADS+threadIdx.x].M;
    __syncthreads();
    for( int j=0; j<THREADS; j++ ) {
      float dx = source[j].x - pos.x;
      float dy = source[j].y - pos.y;
      float dz = source[j].z - pos.z;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = rsqrtf(R2);
      target.w += source[j].w * invR;
      float invR3 = invR * invR * invR * source[j].w;
      target.x += dx * invR3;
      target.y += dy * invR3;
      target.z += dz * invR3;
    }
  }
  bodies[i].F[0] += target.x;
  bodies[i].F[1] += target.y;
  bodies[i].F[2] += target.z;
  bodies[i].P    += target.w;
}

__global__ void copyBodies(Body *bodies, Body *bodies2, const float EPS2) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  bodies2[i].P = bodies[i].P;
  bodies2[i].F[0] = bodies[i].F[0];
  bodies2[i].F[1] = bodies[i].F[1];
  bodies2[i].F[2] = bodies[i].F[2];
  bodies[i].P = -bodies[i].M / sqrtf(EPS2);
  bodies[i].F[0] = bodies[i].F[1] = bodies[i].F[2] = 0;
}

__device__ inline void P2P(Body *bodies, Body *ileafs, int nileaf, const Cell &jcell) {
  for( int i=0; i<nileaf; i++ ) {
    Body ileaf = ileafs[i];
    for( int j=0; j<jcell.NDLEAF; j++ ) {
      Body jleaf = bodies[jcell.LEAF+j];
      float dx = ileaf.X[0] - jleaf.X[0];
      float dy = ileaf.X[1] - jleaf.X[1];
      float dz = ileaf.X[2] - jleaf.X[2];
      float invR = rsqrtf(dx * dx + dy * dy + dz * dz + EPS2);
      float invR3 = -jleaf.M * invR * invR * invR;
      ileaf.P += jleaf.M * invR;
      ileaf.F[0] += dx * invR3;
      ileaf.F[1] += dy * invR3;
      ileaf.F[2] += dz * invR3;
    }
    ileafs[i] = ileaf;
  }
}

__device__ inline void M2P(Body *ileafs, int nileaf, const Cell &jcell) {
  for( int i=0; i<nileaf; i++ ) {
    Body ileaf = ileafs[i];
    float dx = ileaf.X[0] - jcell.X[0];
    float dy = ileaf.X[1] - jcell.X[1];
    float dz = ileaf.X[2] - jcell.X[2];
    float invR = rsqrtf(dx * dx + dy * dy + dz * dz);
    float invR3 = -invR * invR * invR;
    float invR5 = -3 * invR3 * invR * invR;
    ileaf.P += jcell.M[0] * invR;
    ileaf.P += jcell.M[1] * (dx * dx * invR5 + invR3);
    ileaf.P += jcell.M[2] * dx * dy * invR5;
    ileaf.P += jcell.M[3] * dx * dz * invR5;
    ileaf.P += jcell.M[4] * (dy * dy * invR5 + invR3);
    ileaf.P += jcell.M[5] * dy * dz * invR5;
    ileaf.P += jcell.M[6] * (dz * dz * invR5 + invR3);
    ileaf.F[0] += jcell.M[0] * dx * invR3;
    ileaf.F[1] += jcell.M[0] * dy * invR3;
    ileaf.F[2] += jcell.M[0] * dz * invR3;
    ileafs[i] = ileaf;
  }
}

__global__ void evaluate(Body *bodies, Cell *cells, int *NM2P, int *NP2P,
                         const int NCELL, const int NTWIG) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  if( i >= NTWIG ) return;
  Cell icell = cells[i];
  int nileaf = icell.NDLEAF;
  if(nileaf>2*NCRIT) {
    printf("%d\n",nileaf);
    return;
  }
  Body ileafs[2*NCRIT];
  for( int i=0; i<nileaf; i++ ) {
    ileafs[i] = bodies[icell.LEAF+i];
  }
  int stack[30];
  int nstack = 0;
  stack[nstack++] = NCELL - 1;
  int nM2P = 0, nP2P = 0;
  while( nstack ) {
    if( nstack >= 30 ) {
      printf("nstack > %d\n",nstack);
      return;
    }
    Cell jparent = cells[stack[--nstack]];
    for( int j=0; j<jparent.NCHILD; j++ ) {
      Cell jcell = cells[jparent.CHILD+j];
      float dx = icell.X[0] - jcell.X[0];
      float dy = icell.X[1] - jcell.X[1];
      float dz = icell.X[2] - jcell.X[2];
      float R = sqrtf(dx * dx + dy * dy + dz * dz);
      if( jcell.R < THETA * R ) {
        M2P(ileafs,nileaf,jcell);
        nM2P++;
      } else if( jcell.NCHILD == 0 ) {
        P2P(bodies,ileafs,nileaf,jcell);
        nP2P++;
      } else {
        stack[nstack++] = jparent.CHILD+j;
      }
    }
  }
  for( int i=0; i<nileaf; i++ ) {
    bodies[icell.LEAF+i] = ileafs[i];
  }
  NM2P[i] = nM2P;
  NP2P[i] = nP2P;
}

int reduce(int *NDevc, int ntwig);

int main() {
  cudaSetDevice(2);
  Body *bodies = new Body [N];
  Body *bodies2 = new Body [N];
  Cell *cells = new Cell [N];
  int *icellDevc,*ncellDevc,*ileafDevc,*nleafDevc,*ichildDevc,*nchildDevc,*NM2PDevc,*NP2PDevc;
  Body *bodiesDevc, *bodies2Devc;
  Cell *cellsDevc;
  cudaMalloc((void**)&icellDevc,N*sizeof(int));
  cudaMalloc((void**)&ncellDevc,N*sizeof(int));
  cudaMalloc((void**)&ileafDevc,N*sizeof(int));
  cudaMalloc((void**)&nleafDevc,N*sizeof(int));
  cudaMalloc((void**)&ichildDevc,N*sizeof(int));
  cudaMalloc((void**)&nchildDevc,N*sizeof(int));
  cudaMalloc((void**)&NM2PDevc,N*sizeof(int));
  cudaMalloc((void**)&NP2PDevc,N*sizeof(int));
  cudaMalloc((void**)&bodiesDevc,N*sizeof(Body));
  cudaMalloc((void**)&bodies2Devc,N*sizeof(Body));
  cudaMalloc((void**)&cellsDevc,N*sizeof(Cell));
  std::cout << "N     : " << N << std::endl;
  size_t free, total;
  cuMemGetInfo(&free,&total);
  std::cout << "Memory: " << free/1024/1024 << "/" << total/1024/1024 << " MB" << std::endl << std::endl;
// Initialize
  for( int i=0; i!=N; ++i ) {
    bodies[i].X[0] = drand48();
    bodies[i].X[1] = drand48();
    bodies[i].X[2] = drand48();
    bodies[i].M = 1. / N;
    bodies[i].P = -bodies[i].M / sqrtf(EPS2);
    bodies[i].F[0] = bodies[i].F[1] = bodies[i].F[2] = 0;
  }
  cudaMemcpy(bodiesDevc,bodies,N*sizeof(Body),cudaMemcpyHostToDevice);
// Build tree
  double tic = get_time();
  getIndex<<<N/THREADS,THREADS>>>(bodiesDevc,icellDevc,MAXLEVEL);
  sort(bodiesDevc,icellDevc,N);

  int ntwig = 0, ncell = 0;
  buildCell(bodiesDevc,cellsDevc,icellDevc,ncellDevc,ileafDevc,nleafDevc,ntwig,MAXLEVEL,N,THREADS);
  double toc = get_time();
  std::cout << "Index : " << toc-tic << std::endl;
  tic = get_time();
  buildTree(bodiesDevc,cellsDevc,icellDevc,ncellDevc,ileafDevc,nleafDevc,
            ichildDevc,nchildDevc,ntwig,ncell,MAXLEVEL,THREADS);
  toc = get_time();
  std::cout << "Build : " << toc-tic << std::endl;

// Direct summation
  tic = get_time();
//  direct<<<N/THREADS,THREADS>>>(bodiesDevc);
  copyBodies<<<N/THREADS,THREADS>>>(bodiesDevc,bodies2Devc,EPS2);
  toc = get_time();
  std::cout << "Direct: " << toc-tic << std::endl;
// Evaluate
  int nblocks = (ntwig - 1) / THREADS + 1;
  tic = get_time();
  evaluate<<<nblocks,THREADS>>>(bodiesDevc,cellsDevc,NM2PDevc,NP2PDevc,ncell,ntwig);
  toc = get_time();
  int NM2P = reduce(NM2PDevc,ntwig);
  int NP2P = reduce(NP2PDevc,ntwig);
  std::cout << "FMM   : " << toc-tic << std::endl << std::endl;
  std::cout << "NP2P  : " << NP2P << std::endl;
  std::cout << "NM2P  : " << NM2P << std::endl << std::endl;
// Check accuracy
  cudaMemcpy(bodies,bodiesDevc,N*sizeof(Body),cudaMemcpyDeviceToHost);
  cudaMemcpy(bodies2,bodies2Devc,N*sizeof(Body),cudaMemcpyDeviceToHost);
  float errp = 0, relp = 0, errf = 0, relf = 0;
  std::ifstream fid("direct");
  for( int i=0; i<N; i++ ) {
    Body body = bodies[i];
    Body body2 = bodies2[i];
//    fid << body2.P << " " << body2.F[0] << " " << body2.F[1] << " " << body2.F[2] << std::endl;
    fid >> body2.P >> body2.F[0] >> body2.F[1] >> body2.F[2];
    errp += (body2.P - body.P) * (body2.P - body.P);
    relp += body2.P * body2.P;
    errf += (body2.F[0] - body.F[0]) * (body2.F[0] - body.F[0])
          + (body2.F[1] - body.F[1]) * (body2.F[1] - body.F[1])
          + (body2.F[2] - body.F[2]) * (body2.F[2] - body.F[2]);
    relf += body2.F[0] * body2.F[0] + body2.F[1] * body2.F[1] + body2.F[2] * body2.F[2];
  }
  fid.close();
  std::cout << "P err : " << sqrtf(errp/relp) << std::endl;
  std::cout << "F err : " << sqrtf(errf/relf) << std::endl;
  cudaFree(icellDevc);
  cudaFree(ncellDevc);
  cudaFree(ileafDevc);
  cudaFree(nleafDevc);
  cudaFree(ichildDevc);
  cudaFree(nchildDevc);
  cudaFree(NM2PDevc);
  cudaFree(NP2PDevc);
  cudaFree(bodiesDevc);
  cudaFree(bodies2Devc);
  cudaFree(cellsDevc);
  delete[] bodies;
  delete[] bodies2;
  delete[] cells;
}
