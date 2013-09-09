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
  enum {NLEAF_SHIFT = 29};
  enum {NLEAF_MASK  = ~(0x7U << NLEAF_SHIFT)};
  enum {LEVEL_SHIFT = 27};
  enum {LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT)};
  uint4 data;
public:
  __host__ __device__ CellData(
			       const int level,
			       const unsigned int parentCell,
			       const unsigned int nBeg,
			       const unsigned int nEnd,
			       const unsigned int first = 0xFFFFFFFF,
			       const unsigned int n = 0xFFFFFFFF)
  {
    int packed_firstleaf_n = 0xFFFFFFFF;
    if (n != 0xFFFFFFFF)
      packed_firstleaf_n = first | ((unsigned int)n << NLEAF_SHIFT);
    data = make_uint4(parentCell | (level << LEVEL_SHIFT), packed_firstleaf_n, nBeg, nEnd);
  }

  __host__ __device__ CellData(const uint4 data) : data(data) {}

  __host__ __device__ int n()      const {return (data.y >> NLEAF_SHIFT)+1;}
  __host__ __device__ int first()  const {return data.y & NLEAF_MASK;}
  __host__ __device__ int parent() const {return data.x & LEVEL_MASK;}
  __host__ __device__ int level()  const {return data.x >> LEVEL_SHIFT;}
  __host__ __device__ int pbeg()   const {return data.z;}
  __host__ __device__ int pend()   const {return data.w;}

  __host__ __device__ bool isLeaf() const {return data.y == 0xFFFFFFFF;}
  __host__ __device__ bool isNode() const {return !isLeaf();}

  __host__ __device__ void update_first(const int first) 
  {
    const int _n = n()-1;
    data.y = first | ((unsigned int)_n << NLEAF_SHIFT);
  }
  __host__ __device__ void update_parent(const int parent)
  {
    data.x = parent | (level() << LEVEL_SHIFT);
  }
};

struct Treecode
{
private:
  int nBody, numLevels, numSources, numLeaves, numTargets, nCrit, nLeaf;
  float theta, eps2;

public:
  int get_nBody() const { return nBody; }
  int get_nCrit() const { return nCrit; }
  int get_nLeaf() const { return nLeaf; }
  int getNumSources() const { return numSources; }
  int getNumLevels() const { return numLevels; }

  host_mem<float4> h_bodyPos, h_bodyVel, h_bodyAcc;
  host_mem<float4> h_bodyAcc2;
  cuda_mem<float4> d_bodyPos, d_bodyVel, d_bodyPos_tmp, d_bodyAcc;
  cuda_mem<float4> d_bodyAcc2;
  cuda_mem<float4> d_domain;
  cuda_mem<float3> d_minmax;
  cuda_mem<int2> d_levelRange;

  int node_max, cell_max, stack_size;
  cuda_mem<int>  d_stack_memory_pool;
  cuda_mem<CellData> d_sourceCells, d_sourceCells_tmp;
  cuda_mem<int2> d_targetCells;
  cuda_mem<int>  d_leafCells;
  cuda_mem<int>  d_key, d_value;
  cuda_mem<float4> d_sourceCenter, d_Monopole;
  cuda_mem<float4> d_Quadrupole0;
  cuda_mem<float2> d_Quadrupole1;

  Treecode(const float _eps = 0.01, const float _theta = 0.75, const int _ncrit = 2*WARP_SIZE)
  {
    eps2  = _eps*_eps;
    theta = _theta;
    nCrit = _ncrit;
    d_domain.alloc(1);
    d_minmax.alloc(2048);
    d_levelRange.alloc(32);
  }

  void alloc(const int nBody)
  {
    this->nBody = nBody;
    h_bodyPos.alloc(nBody);
    h_bodyVel.alloc(nBody);
    h_bodyAcc.alloc(nBody);
    h_bodyAcc2.alloc(nBody);

    d_bodyPos.alloc(nBody);
    d_bodyVel.alloc(nBody);
    d_bodyPos_tmp.alloc(nBody);
    d_bodyAcc.alloc(nBody);
    d_bodyAcc2.alloc(nBody);
 
    /* allocate stack memory */ 
    node_max = nBody/10;
    stack_size = (8+8+8+64+8)*node_max;
    fprintf(stdout,"Stack size           : %g MB\n",sizeof(int)*stack_size/1024.0/1024.0);
    d_stack_memory_pool.alloc(stack_size);
  
    /* allocate celldata memory */
    cell_max = nBody;
    fprintf(stdout,"Cell data            : %g MB\n",cell_max*sizeof(CellData)/1024.0/1024.0);
    d_sourceCells.alloc(cell_max);
    d_sourceCells_tmp.alloc(cell_max);
    d_key.alloc(cell_max);
    d_value.alloc(cell_max);
  };

  void body_d2h() {
    d_bodyPos_tmp.d2h(h_bodyPos);
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

  void buildTree(const int nLeaf = 16);
  void computeMultipoles();
  void groupTargets(int levelSplit = 1, const int nCrit = 64);
  float4 computeForces();
  void computeDirect(const int numTarget, const int numBlock);
};
