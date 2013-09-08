#pragma once

#include <assert.h>
#include "cudamem.h"
#include "plummer.h"
#include <string>
#include <sstream>
#include <sys/time.h>

#if 1
#define QUADRUPOLE
#endif

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

static void kernelSuccess(const char kernel[] = "kernel")
{
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    assert(0);
  }
}

struct GroupData
{
  private:
    int2 packed_data;
  public:
    __host__ __device__ GroupData(const int2 data) : packed_data(data) {}
    __host__ __device__ GroupData(const int pbeg, const int np)
    {
      packed_data.x = pbeg;
      packed_data.y = np;
    }

    __host__ __device__ int pbeg() const {return packed_data.x;}
    __host__ __device__ int np  () const {return packed_data.y;}
};

struct CellData
{
  private:
    enum {NLEAF_SHIFT = 29};
    enum {NLEAF_MASK  = ~(0x7U << NLEAF_SHIFT)};
    enum {LEVEL_SHIFT = 27};
    enum {LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT)};
    uint4 packed_data;
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
      packed_data = make_uint4(parentCell | (level << LEVEL_SHIFT), packed_firstleaf_n, nBeg, nEnd);
    }

    __host__ __device__ CellData(const uint4 data) : packed_data(data) {}

    __host__ __device__ int n()      const {return (packed_data.y >> NLEAF_SHIFT)+1;}
    __host__ __device__ int first()  const {return packed_data.y  & NLEAF_MASK;}
    __host__ __device__ int parent() const {return packed_data.x  & LEVEL_MASK;}
    __host__ __device__ int level()  const {return packed_data.x >> LEVEL_SHIFT;}
    __host__ __device__ int pbeg()   const {return packed_data.z;}
    __host__ __device__ int pend()   const {return packed_data.w;}

    __host__ __device__ bool isLeaf() const {return packed_data.y == 0xFFFFFFFF;}
    __host__ __device__ bool isNode() const {return !isLeaf();}

    __host__ __device__ void update_first(const int first) 
    {
      const int _n = n()-1;
      packed_data.y = first | ((unsigned int)_n << NLEAF_SHIFT);
    }
    __host__ __device__ void update_parent(const int parent)
    {
      packed_data.x = parent | (level() << LEVEL_SHIFT);
    }
};

struct Treecode
{
  float theta, eps2;
  private:
  int nPtcl, nLevels, nCells, nLeaves, nNodes, nGroups, nCrit, nLeaf;

  public:
    int get_nPtcl() const { return nPtcl; }
    int get_nCrit() const { return nCrit; }
    int get_nLeaf() const { return nLeaf; }
    int get_nCells() const { return nCells; }
    int get_nLevels() const { return nLevels; }

  host_mem<float4> h_ptclPos, h_ptclVel, h_ptclAcc;
  host_mem<float4> h_ptclAcc2;
  std::vector<float4> ptcl0;
  cuda_mem<float4> d_ptclPos, d_ptclVel, d_ptclPos_tmp, d_ptclAcc;
  cuda_mem<float4> d_ptclAcc2;
  cuda_mem<float4> d_domain;
  cuda_mem<float3> d_minmax;
  cuda_mem<int2> d_level_begIdx;

  int node_max, cell_max, stack_size;
  cuda_mem<int>  d_stack_memory_pool;
  cuda_mem<CellData> d_cellDataList, d_cellDataList_tmp;
  cuda_mem<GroupData> d_groupList;
  cuda_mem<int>  d_leafList;

  cuda_mem<int>  d_key, d_value;


  cuda_mem<float4> d_sourceCenter, d_cellMonopole;
  cuda_mem<float4> d_cellQuad0;
  cuda_mem<float2> d_cellQuad1;

  Treecode(const float _eps = 0.01, const float _theta = 0.75, const int _ncrit = 2*WARP_SIZE)
  {
    eps2  = _eps*_eps;
    theta = _theta;
    nCrit = _ncrit;
    d_domain.alloc(1);
    d_minmax.alloc(2048);
    d_level_begIdx.alloc(32);  /* max 32 levels */
    CUDA_SAFE_CALL(cudaMemset(d_level_begIdx,0, 32*sizeof(int2)));
  }

  void alloc(const int nPtcl)
  {
    this->nPtcl = nPtcl;
    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);
    h_ptclAcc.alloc(nPtcl);
    h_ptclAcc2.alloc(nPtcl);
    d_ptclPos.alloc(nPtcl);
    d_ptclVel.alloc(nPtcl);
    d_ptclPos_tmp.alloc(nPtcl);
    d_ptclAcc.alloc(nPtcl);
    d_ptclAcc2.alloc(nPtcl);
 
    /* allocate stack memory */ 
    node_max = nPtcl/10;
    stack_size = (8+8+8+64+8)*node_max;
    fprintf(stdout,"Stack size           : %g MB\n",sizeof(int)*stack_size/1024.0/1024.0);
    d_stack_memory_pool.alloc(stack_size);
  
    /* allocate celldata memory */
    cell_max = nPtcl;
    fprintf(stdout,"Cell data            : %g MB\n",cell_max*sizeof(CellData)/1024.0/1024.0);
    d_cellDataList.alloc(cell_max);
    d_cellDataList_tmp.alloc(cell_max);
    d_key.alloc(cell_max);
    d_value.alloc(cell_max);
  };

  void ptcl_d2h()
  {
    d_ptclPos_tmp.d2h(h_ptclPos);
    d_ptclVel.d2h(h_ptclVel);
    d_ptclAcc.d2h(h_ptclAcc);
    d_ptclAcc2.d2h(h_ptclAcc2);
  }


  void ptcl_h2d()
  {
    d_ptclPos.h2d(h_ptclPos);
    d_ptclVel.h2d(h_ptclVel);
    d_ptclAcc.h2d(h_ptclAcc);
    d_ptclAcc2.h2d(h_ptclAcc2);
    ptcl0.resize(nPtcl);
    for (int i = 0; i < nPtcl; i++)
      ptcl0[i] = h_ptclPos[i];
  }

  void buildTree(const int nLeaf = 16);
  void computeMultipoles();
  void makeGroups(int levelSplit = 1, const int nCrit = 64);
  float4 computeForces(const bool INTCOUNT = true);
  void computeDirect(const int numTarget, const int numBlock);
  void moveParticles();
  void computeEnergies();

};


