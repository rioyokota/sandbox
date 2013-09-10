#pragma once

#include <assert.h>
#include "cudamem.h"
#include "plummer.h"
#include <string>
#include <sstream>
#include <sys/time.h>

#define WARP_SIZE2 5
#define WARP_SIZE 32

struct float6 {
  float xx;
  float yy;
  float zz;
  float xy;
  float xz;
  float yz;
};

struct double6 {
  double xx;
  double yy;
  double zz;
  double xy;
  double xz;
  double yz;
};

static inline double get_time() {
  struct timeval tv;                                          // Time value
  gettimeofday(&tv, NULL);                                    // Get time of day in seconds and microseconds
  return double(tv.tv_sec+tv.tv_usec*1e-6);                   // Combine seconds and microseconds and return
}

static void kernelSuccess(const char kernel[] = "kernel") {
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    assert(0);
  }
}

struct CellData {
private:
  enum {CHILD_SHIFT = 29};
  enum {CHILD_MASK  = ~(0x7U << CHILD_SHIFT)};
  enum {LEVEL_SHIFT = 27};
  enum {LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT)};
  uint4 data;
public:
  __host__ __device__ CellData(const unsigned int level,
			       const unsigned int parent,
			       const unsigned int body,
			       const unsigned int nbody,
			       const unsigned int child = 0,
			       const unsigned int nchild = 0)
  {
    const unsigned int parentPack = parent | (level << LEVEL_SHIFT);
    const unsigned int childPack = child | (nchild << CHILD_SHIFT);
    data = make_uint4(parentPack, childPack, body, nbody);
  }

  __host__ __device__ CellData(const uint4 data) : data(data) {}

  __host__ __device__ int level()  const {return data.x >> LEVEL_SHIFT;}
  __host__ __device__ int parent() const {return data.x & LEVEL_MASK;}
  __host__ __device__ int child()  const {return data.y & CHILD_MASK;}
  __host__ __device__ int nchild() const {return (data.y >> CHILD_SHIFT)+1;}
  __host__ __device__ int body()   const {return data.z;}
  __host__ __device__ int nbody()  const {return data.w;}

  __host__ __device__ bool isLeaf() const {return data.y == 0;}
  __host__ __device__ bool isNode() const {return !isLeaf();}

  __host__ __device__ void setParent(const unsigned int parent) {
    data.x = parent | (level() << LEVEL_SHIFT);
  }
  __host__ __device__ void setChild(const unsigned int child) {
    data.y = child | (nchild()-1 << CHILD_SHIFT);
  }
};

struct Treecode
{
private:
  int numBody, numLevels, numSources, numLeaves, numTargets;
  int NCRIT, NLEAF;
  float THETA, EPS2;

public:
  int getNumBody() const { return numBody; }
  int getNumSources() const { return numSources; }
  int getNumLevels() const { return numLevels; }

  host_mem<float4> h_bodyPos, h_bodyVel, h_bodyAcc, h_bodyAcc2;
  cuda_mem<float4> d_bodyPos, d_bodyVel, d_bodyPos2, d_bodyAcc, d_bodyAcc2;
  cuda_mem<float4> d_domain;
  cuda_mem<float3> d_minmax;
  cuda_mem<int2> d_levelRange;

  int maxNode, maxCell, stackSize;
  cuda_mem<int>  d_stack_memory_pool;
  cuda_mem<CellData> d_sourceCells, d_sourceCells2;
  cuda_mem<int2> d_targetCells;
  cuda_mem<int>  d_leafCells;
  cuda_mem<int>  d_key, d_value;
  cuda_mem<float4> d_sourceCenter, d_Monopole;
  cuda_mem<float4> d_Quadrupole0;
  cuda_mem<float2> d_Quadrupole1;

  Treecode(const float eps = 0.01, const float theta = 0.75, const int ncrit = 2*WARP_SIZE) {
    EPS2  = eps * eps;
    THETA = theta;
    NCRIT = ncrit;
    d_domain.alloc(1);
    d_minmax.alloc(2048);
    d_levelRange.alloc(32);
  }

  void alloc(const int numBody)
  {
    this->numBody = numBody;
    h_bodyPos.alloc(numBody);
    h_bodyVel.alloc(numBody);
    h_bodyAcc.alloc(numBody);
    h_bodyAcc2.alloc(numBody);

    d_bodyPos.alloc(numBody);
    d_bodyVel.alloc(numBody);
    d_bodyPos2.alloc(numBody);
    d_bodyAcc.alloc(numBody);
    d_bodyAcc2.alloc(numBody);

    /* allocate stack memory */
    maxNode = numBody / 10;
    stackSize = (8+8+8+64+8)*maxNode;
    fprintf(stdout,"Stack size           : %g MB\n",sizeof(int)*stackSize/1024.0/1024.0);
    d_stack_memory_pool.alloc(stackSize);

    /* allocate celldata memory */
    maxCell = numBody;
    fprintf(stdout,"Cell data            : %g MB\n",maxCell*sizeof(CellData)/1024.0/1024.0);
    d_sourceCells.alloc(maxCell);
    d_sourceCells2.alloc(maxCell);
    d_key.alloc(maxCell);
    d_value.alloc(maxCell);
  };

  void body_d2h() {
    d_bodyPos2.d2h(h_bodyPos);
    d_bodyVel.d2h(h_bodyVel);
    d_bodyAcc.d2h(h_bodyAcc);
    d_bodyAcc2.d2h(h_bodyAcc2);
  }


  void body_h2d() {
    d_bodyPos.h2d(h_bodyPos);
    d_bodyVel.h2d(h_bodyVel);
    d_bodyAcc.h2d(h_bodyAcc);
    d_bodyAcc2.h2d(h_bodyAcc2);
  }

  void buildTree(const int NLEAF = 16);
  void computeMultipoles();
  void groupTargets(int levelSplit = 1, const int NCRIT = 64);
  float4 computeForces();
  void computeDirect(const int numTarget, const int numBlock);
};
