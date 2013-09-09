#include "Treecode.h"
#include <algorithm>

#include "cuda_primitives.h"

#define IF(x) (-(int)(x))

namespace computeForces
{

#define CELL_LIST_MEM_PER_WARP (4096*32)
  
  texture<uint4,  1, cudaReadModeElementType> texCellData;
  texture<float4, 1, cudaReadModeElementType> texSourceCenter;
  texture<float4, 1, cudaReadModeElementType> texCellMonopole;
  texture<float4, 1, cudaReadModeElementType> texCellQuad0;
  texture<float2, 1, cudaReadModeElementType> texCellQuad1;
  texture<float4, 1, cudaReadModeElementType> texPtcl;

  static __device__ __forceinline__ int ringAddr(const int i) {
    return i & (CELL_LIST_MEM_PER_WARP - 1);
  }

  static __device__ bool applyMAC(const float4 sourceCenter, const float3 targetCenter, 
    const float3 targetSize)
  {
    float3 dr = make_float3(fabsf(targetCenter.x - sourceCenter.x) - (targetSize.x),
                            fabsf(targetCenter.y - sourceCenter.y) - (targetSize.y),
                            fabsf(targetCenter.z - sourceCenter.z) - (targetSize.z));
    dr.x += fabsf(dr.x); dr.x *= 0.5f;
    dr.y += fabsf(dr.y); dr.y *= 0.5f;
    dr.z += fabsf(dr.z); dr.z *= 0.5f;
    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
    return ds2 < fabsf(sourceCenter.w);
  }

  static __device__ __forceinline__ float4 P2P(float4 acc, const float3 pos,
    const float4 posj, const float eps2)
  {
    const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
    const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;
    const float rinv   = rsqrtf(r2);
    const float rinv2  = rinv*rinv;
    const float mrinv  = posj.w * rinv;
    const float mrinv3 = mrinv * rinv2;
    acc.w -= mrinv;
    acc.x += mrinv3 * dr.x;
    acc.y += mrinv3 * dr.y;
    acc.z += mrinv3 * dr.z;
    return acc;
  }

  static __device__ __forceinline__ float4 M2P(float4 acc, const float3 pos,
    const float4 M0, const float4 Q0,  const float2 Q1, float eps2)
  {
    const float3 dr = make_float3(pos.x - M0.x, pos.y - M0.y, pos.z - M0.z);
    const float  r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + eps2;
    const float rinv  = rsqrtf(r2);
    const float rinv2 = rinv * rinv;
    const float mrinv  = M0.w * rinv;
    const float mrinv3 = rinv2 * mrinv;
    const float mrinv5 = rinv2 * mrinv3; 
    const float mrinv7 = rinv2 * mrinv5; // 16

    float  D0  =  mrinv;
    float  D1  = -mrinv3;
    float  D2  =  mrinv5 * 3.0f;
    float  D3  = -mrinv7 * 15.0f; // 3

    const float q11 = Q0.x;
    const float q22 = Q0.y;
    const float q33 = Q0.z;
    const float q12 = Q0.w;
    const float q13 = Q1.x;
    const float q23 = Q1.y;

    const float  q  = q11 + q22 + q33;
    const float3 qR = make_float3(
      q11 * dr.x + q12 * dr.y + q13 * dr.z,
      q12 * dr.x + q22 * dr.y + q23 * dr.z,
      q13 * dr.x + q23 * dr.y + q33 * dr.z);
    const float qRR = qR.x * dr.x + qR.y * dr.y + qR.z * dr.z; // 22

    acc.w  -= D0 + 0.5f * (D1*q + D2 * qRR);
    float C = D1 + 0.5f * (D2*q + D3 * qRR);
    acc.x  += C * dr.x + D2 * qR.x;
    acc.y  += C * dr.y + D2 * qR.y;
    acc.z  += C * dr.z + D2 * qR.z; // 23

    // total: 16 + 3 + 22 + 23 = 64 flops 
    return acc;
  }

  template<int NI, bool FULL>
  static __device__ __forceinline__ void approxAcc(float4 acc_i[NI], const float3 pos_i[NI],
    const int cellIdx, const float eps2) {
    float4 M0, Q0;
    float2 Q1;
    if (FULL || cellIdx >= 0) {
      M0 = tex1Dfetch(texCellMonopole, cellIdx);
      Q0 = tex1Dfetch(texCellQuad0,    cellIdx);
      Q1 = tex1Dfetch(texCellQuad1,    cellIdx);
    } else {
      M0 = Q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      Q1 = make_float2(0.0f, 0.0f);
    }
    for (int j=0; j<WARP_SIZE; j++) {
      const float4 jM0 = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
      const float4 jQ0 = make_float4(__shfl(Q0.x, j), __shfl(Q0.y, j), __shfl(Q0.z, j), __shfl(Q0.w,j));
      const float2 jQ1 = make_float2(__shfl(Q1.x, j), __shfl(Q1.y, j));
#pragma unroll
      for (int k=0; k<NI; k++)
	acc_i[k] = M2P(acc_i[k], pos_i[k], jM0, jQ0, jQ1, eps2);
    }
  }

  template<int BLOCKDIM2, int NI>
  static __device__ uint2 treewalk_warp(float4 acc_i[NI], const float3 pos_i[NI],
    const float3 targetCenter, const float3 targetSize, const float eps2, const int2 top_cells,
    int *shmem, int *cellList) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);

    uint2 counters = {0,0};

    volatile int *tmpList = shmem;

    int approxCellIdx, directPtclIdx;

    int directCounter = 0;
    int approxCounter = 0;


    for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
      if (root_cell + laneIdx < top_cells.y)
	cellList[ringAddr(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

    int nCells = top_cells.y - top_cells.x;

    int cellListBlock        = 0;
    int nextLevelCellCounter = 0;

    unsigned int cellListOffset = 0;

    /* process level with n_cells */
#if 1
    while (nCells > 0)
    {
      /* extract cell index from the current level cell list */
      const int cellListIdx = cellListBlock + laneIdx;
      const bool useCell    = cellListIdx < nCells;
      const int cellIdx     = cellList[ringAddr(cellListOffset + cellListIdx)];
      cellListBlock += min(WARP_SIZE, nCells - cellListBlock);

      /* read from gmem cell's info */
      const float4   sourceCenter = tex1Dfetch(texSourceCenter, cellIdx);
      const CellData cellData = tex1Dfetch(texCellData, cellIdx);

      const bool splitCell = applyMAC(sourceCenter, targetCenter, targetSize) ||
	(cellData.pend() - cellData.pbeg() < 3); /* force to open leaves with less than 3 particles */

      /**********************************************/
      /* split cells that satisfy opening condition */
      /**********************************************/

      const bool isNode = cellData.isNode();

      {
	const int firstChild = cellData.first();
	const int nChild= cellData.n();
	bool splitNode  = isNode && splitCell && useCell;

	/* use exclusive scan to compute scatter addresses for each of the child cells */
	const int2 childScatter = warpIntExclusiveScan(nChild & (-splitNode));

	/* make sure we still have available stack space */
	if (childScatter.y + nCells - cellListBlock > CELL_LIST_MEM_PER_WARP)
	  return make_uint2(0xFFFFFFFF,0xFFFFFFFF);

#if 1
	/* if so populate next level stack in gmem */
	if (splitNode)
	{
	  const int scatterIdx = cellListOffset + nCells + nextLevelCellCounter + childScatter.x;
	  for (int i = 0; i < nChild; i++)
	    cellList[ringAddr(scatterIdx + i)] = firstChild + i;
	}
#else  /* use scan operation to accomplish steps above, doesn't bring performance benefit */
	int nChildren  = childScatter.y;
	int nProcessed = 0;
	int2 scanVal   = {0,0};
	const int offset = cellListOffset + nCells + nextLevelCellCounter;
	while (nChildren > 0)
	{
	  tmpList[laneIdx] = 1;
	  if (splitNode && (childScatter.x - nProcessed < WARP_SIZE))
	  {
	    splitNode = false;
	    tmpList[childScatter.x - nProcessed] = -1-firstChild;
	  }
	  scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
	  if (laneIdx < nChildren)
	    cellList[ringAddr(offset + nProcessed + laneIdx)] = scanVal.x;
	  nChildren  -= WARP_SIZE;
	  nProcessed += WARP_SIZE;
	}
#endif
	nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */
      }

#if 1
      {
	/***********************************/
	/******       APPROX          ******/
	/***********************************/

	/* see which thread's cell can be used for approximate force calculation */
	const bool approxCell    = !splitCell && useCell;
	const int2 approxScatter = warpBinExclusiveScan(approxCell);

	/* store index of the cell */
	const int scatterIdx = approxCounter + approxScatter.x;
	tmpList[laneIdx] = approxCellIdx;
	if (approxCell && scatterIdx < WARP_SIZE)
	  tmpList[scatterIdx] = cellIdx;

	approxCounter += approxScatter.y;

	/* compute approximate forces */
	if (approxCounter >= WARP_SIZE)
	{
	  /* evalute cells stored in shmem */
	  approxAcc<NI,true>(acc_i, pos_i, tmpList[laneIdx], eps2);

	  approxCounter -= WARP_SIZE;
	  const int scatterIdx = approxCounter + approxScatter.x - approxScatter.y;
	  if (approxCell && scatterIdx >= 0)
	    tmpList[scatterIdx] = cellIdx;
	  counters.x += WARP_SIZE;
	}
	approxCellIdx = tmpList[laneIdx];
      }
#endif

#if 1
      {
	/***********************************/
	/******       DIRECT          ******/
	/***********************************/

	const bool isLeaf = !isNode;
	bool isDirect = splitCell && isLeaf && useCell;

	const int firstBody = cellData.pbeg();
	const int     nBody = cellData.pend() - cellData.pbeg();

	const int2 childScatter = warpIntExclusiveScan(nBody & (-isDirect));
	int nParticle  = childScatter.y;
	int nProcessed = 0;
	int2 scanVal   = {0,0};

	/* conduct segmented scan for all leaves that need to be expanded */
	while (nParticle > 0)
	{
	  tmpList[laneIdx] = 1;
	  if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
	  {
	    isDirect = false;
	    tmpList[childScatter.x - nProcessed] = -1-firstBody;
	  }
	  scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
	  const int  ptclIdx = scanVal.x;

	  if (nParticle >= WARP_SIZE)
	  {
	    const float4 M0 = tex1Dfetch(texPtcl, ptclIdx);
	    for (int j=0; j<WARP_SIZE; j++) {
	      const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	      for (int k=0; k<NI; k++)
		acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, eps2);
	    }
	    nParticle  -= WARP_SIZE;
	    nProcessed += WARP_SIZE;
	    counters.y += WARP_SIZE;
	  }
	  else 
	  {
	    const int scatterIdx = directCounter + laneIdx;
	    tmpList[laneIdx] = directPtclIdx;
	    if (scatterIdx < WARP_SIZE)
	      tmpList[scatterIdx] = ptclIdx;

	    directCounter += nParticle;

	    if (directCounter >= WARP_SIZE)
	    {
	      /* evalute cells stored in shmem */
	      const float4 M0 = tex1Dfetch(texPtcl, tmpList[laneIdx]);
	      for (int j=0; j<WARP_SIZE; j++) {
		const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
		for (int k=0; k<NI; k++)
		  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, eps2);
	      }
	      directCounter -= WARP_SIZE;
	      const int scatterIdx = directCounter + laneIdx - nParticle;
	      if (scatterIdx >= 0)
		tmpList[scatterIdx] = ptclIdx;
	      counters.y += WARP_SIZE;
	    }
	    directPtclIdx = tmpList[laneIdx];

	    nParticle = 0;
	  }
	}
      }
#endif

      /* if the current level is processed, schedule the next level */
      if (cellListBlock >= nCells)
      {
	cellListOffset += nCells;
	nCells = nextLevelCellCounter;
	cellListBlock = nextLevelCellCounter = 0;
      }

    }  /* level completed */
#endif

#if 1
    if (approxCounter > 0)
    {
      approxAcc<NI,false>(acc_i, pos_i, laneIdx < approxCounter ? approxCellIdx : -1, eps2);
      counters.x += approxCounter;
      approxCounter = 0;
    }
#endif

#if 1
    if (directCounter > 0)
    {
      const int ptclIdx = laneIdx < directCounter ? directPtclIdx : -1;
      const float4 M0 = ptclIdx >= 0 ? tex1Dfetch(texPtcl, ptclIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int j=0; j<WARP_SIZE; j++) {
	const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	for (int k=0; k<NI; k++)
	  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, eps2);
      }
      counters.y += directCounter;
      directCounter = 0;
    }
#endif

    return counters;
  }

  __device__ unsigned int retired_groupCount = 0;

  __device__ unsigned long long g_direct_sum = 0;
  __device__ unsigned int       g_direct_max = 0;

  __device__ unsigned long long g_approx_sum = 0;
  __device__ unsigned int       g_approx_max = 0;

  template<int NTHREAD2, int NI>
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__ 
    void treewalk(
        const int nGroups,
        const int2 *groupList,
        const float eps2,
        const int start_level,
        const int2 *level_begIdx,
        const float4 *ptclPos,
        float4 *acc,
        int    *gmem_pool)
    {
      const int NTHREAD = 1<<NTHREAD2;
      const int shMemSize = NTHREAD;
      __shared__ int shmem_pool[shMemSize];

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      const int warpIdx = threadIdx.x >> WARP_SIZE2;

      const int NWARP2 = NTHREAD2 - WARP_SIZE2;
      const int sh_offs = (shMemSize >> NWARP2) * warpIdx;
      int *shmem = shmem_pool + sh_offs;
      int *gmem  =  gmem_pool + CELL_LIST_MEM_PER_WARP*((blockIdx.x<<NWARP2) + warpIdx);

      int2 top_cells = level_begIdx[start_level];
      top_cells.y++;

      while(1)
      {
        int groupIdx = 0;
        if (laneIdx == 0)
          groupIdx = atomicAdd(&retired_groupCount, 1);
        groupIdx = __shfl(groupIdx, 0, WARP_SIZE);

        if (groupIdx >= nGroups) 
          return;

        const int2 group = groupList[groupIdx];
        const int begin = group.x;
        const int end   = group.x+group.y;

        float3 iPos[NI];

#pragma unroll
        for (int i = 0; i < NI; i++)
        {
          const float4 ptcl = ptclPos[min(begin + i*WARP_SIZE+laneIdx, end-1)];
          iPos [i] = make_float3(ptcl.x, ptcl.y, ptcl.z);
        }

        float3 rmin = {iPos[0].x, iPos[0].y, iPos[0].z};
        float3 rmax = rmin; 

#pragma unroll
        for (int i = 0; i < NI; i++) 
          addBoxSize(rmin, rmax, make_float3(iPos[i].x, iPos[i].y, iPos[i].z));

        rmin.x = __shfl(rmin.x,0);
        rmin.y = __shfl(rmin.y,0);
        rmin.z = __shfl(rmin.z,0);
        rmax.x = __shfl(rmax.x,0);
        rmax.y = __shfl(rmax.y,0);
        rmax.z = __shfl(rmax.z,0);

        const float half = 0.5f;
        const float3 targetCenter = {half*(rmax.x+rmin.x), half*(rmax.y+rmin.y), half*(rmax.z+rmin.z)};
        const float3 hvec = {half*(rmax.x-rmin.x), half*(rmax.y-rmin.y), half*(rmax.z-rmin.z)};

        float4 iAcc[NI] = {0.0f, 0.0f, 0.0f, 0.0f};

        uint2 counters;
        counters =  treewalk_warp<NTHREAD2,NI>
          (iAcc, iPos, targetCenter, hvec, eps2, top_cells, shmem, gmem);

        assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

        const int pidx = begin + laneIdx;

	int direct_max = counters.y;
	int direct_sum = 0;
	int approx_max = counters.x;
	int approx_sum = 0;

#pragma unroll
	for (int i = 0; i < NI; i++)
	  if (i*WARP_SIZE + pidx < end)
	  {
	    approx_sum += counters.x;
	    direct_sum += counters.y;
	  }

#pragma unroll
	for (int i = WARP_SIZE2-1; i >= 0; i--)
	{
	  direct_max  = max(direct_max, __shfl_xor(direct_max, 1<<i));
	  direct_sum += __shfl_xor(direct_sum, 1<<i);
	  approx_max  = max(approx_max, __shfl_xor(approx_max, 1<<i));
	  approx_sum += __shfl_xor(approx_sum, 1<<i);
	}

	if (laneIdx == 0)
	{
	  atomicMax(&g_direct_max,                     direct_max);
	  atomicAdd(&g_direct_sum, (unsigned long long)direct_sum);
	  atomicMax(&g_approx_max,                     approx_max);
	  atomicAdd(&g_approx_sum, (unsigned long long)approx_sum);
	}

#pragma unroll
        for (int i = 0; i < NI; i++)
          if (pidx + i*WARP_SIZE< end)
          {
            const float4 iacc = {iAcc[i].x, iAcc[i].y, iAcc[i].z, iAcc[i].w};
            acc[i*WARP_SIZE + pidx] = iacc;
          }
      }
    }

  static __global__
  void direct(const int numSource,
              const float eps2,
	      float4 *acc)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.x * numSource / gridDim.x;
    float pots, axs, ays ,azs;
    float potc, axc, ayc ,azc;
    float4 si = tex1Dfetch(texPtcl, threadIdx.x);
    __shared__ float4 s[512];
    for ( int jb=0; jb<numSource/blockDim.x/gridDim.x; jb++ ) {
      __syncthreads();
      s[threadIdx.x] = tex1Dfetch(texPtcl, offset+jb*blockDim.x+threadIdx.x);
      __syncthreads();
      for( int j=0; j<blockDim.x; j++ ) {
	float dx = s[j].x - si.x;
	float dy = s[j].y - si.y;
	float dz = s[j].z - si.z;
	float R2 = dx * dx + dy * dy + dz * dz + eps2;
	float invR = rsqrtf(R2);
        float y = - s[j].w * invR - potc;
        float t = pots + y;
        potc = (t - pots) - y;
        pots = t;
	float invR3 = invR * invR * invR * s[j].w;
        y = dx * invR3 - axc;
        t = axs + y;
        axc = (t - axs) - y;
        axs = t;
        y = dy * invR3 - ayc;
        t = ays + y;
        ayc = (t - ays) - y;
        ays = t;
        y = dz * invR3 - azc;
        t = azs + y;
        azc = (t - azs) - y;
        azs = t;
      }
    }
    acc[i].x = axs + axc;
    acc[i].y = ays + ayc;
    acc[i].z = azs + azc;
    acc[i].w = pots + potc;
  }
}

float4 Treecode::computeForces() {
  bindTexture(computeForces::texCellData,     (uint4* )d_cellDataList.ptr, nCells);
  bindTexture(computeForces::texSourceCenter,     d_sourceCenter.ptr, nCells);
  bindTexture(computeForces::texCellMonopole, d_cellMonopole.ptr, nCells);
  bindTexture(computeForces::texCellQuad0,    d_cellQuad0.ptr,    nCells);
  bindTexture(computeForces::texCellQuad1,    d_cellQuad1.ptr,    nCells);
  bindTexture(computeForces::texPtcl,         d_ptclPos.ptr,      nPtcl);

  const int NTHREAD2 = 7;
  const int NTHREAD  = 1<<NTHREAD2;
  cuda_mem<int> d_gmem_pool;

  const int nblock = 8*13;
  d_gmem_pool.alloc(CELL_LIST_MEM_PER_WARP*nblock*(NTHREAD/WARP_SIZE));

#if 0
  CUDA_SAFE_CALL(cudaMemset(d_ptclAcc, 0, sizeof(float4)*nPtcl));
#endif
  const int starting_level = 1;
  int value = 0;
  cudaDeviceSynchronize();
  const double t0 = get_time();
  unsigned long long lzero = 0;
  unsigned int       uzero = 0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::retired_groupCount, &value, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_sum, &lzero, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_max, &uzero, sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_sum, &lzero, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_max, &uzero, sizeof(unsigned int)));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,2>, cudaFuncCachePreferL1));
  computeForces::treewalk<NTHREAD2,2><<<nblock,NTHREAD>>>(
    nGroups, d_groupList, eps2, starting_level, d_level_begIdx,
    d_ptclPos_tmp, d_ptclAcc,
    d_gmem_pool);
  kernelSuccess("treewalk");
  const double dt = get_time() - t0;

  float4 interactions = {0.0, 0.0, 0.0, 0.0};
  unsigned long long direct_sum, approx_sum;
  unsigned int direct_max, approx_max;
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_sum, computeForces::g_direct_sum, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_max, computeForces::g_direct_max, sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_sum, computeForces::g_approx_sum, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_max, computeForces::g_approx_max, sizeof(unsigned int)));
  interactions.x = direct_sum*1.0/nPtcl;
  interactions.y = direct_max;
  interactions.z = approx_sum*1.0/nPtcl;
  interactions.w = approx_max;

  float flops = (interactions.x*20 + interactions.z*64)*nPtcl/dt/1e12;
  fprintf(stdout,"Traverse             : %.7f s (%.7f TFlops)\n",dt,flops);

  unbindTexture(computeForces::texPtcl);
  unbindTexture(computeForces::texCellQuad1);
  unbindTexture(computeForces::texCellQuad0);
  unbindTexture(computeForces::texCellMonopole);
  unbindTexture(computeForces::texSourceCenter);
  unbindTexture(computeForces::texCellData);

  return interactions;
}

void Treecode::computeDirect(const int numTarget, const int numBlock)
{
  bindTexture(computeForces::texPtcl,d_ptclPos_tmp.ptr,nPtcl);
  computeForces::direct<<<numBlock,numTarget>>>(nPtcl, eps2, d_ptclAcc2);
  unbindTexture(computeForces::texPtcl);
  cudaDeviceSynchronize();
}
