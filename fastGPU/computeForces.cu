#include "Treecode.h"
#include <algorithm>

#include "cuda_primitives.h"

#define CELL_LIST_MEM_PER_WARP (4096*32)
#define IF(x) (-(int)(x))

namespace computeForces {  
  texture<uint4,  1, cudaReadModeElementType> texCell;
  texture<float4, 1, cudaReadModeElementType> texCellCenter;
  texture<float4, 1, cudaReadModeElementType> texMonopole;
  texture<float4, 1, cudaReadModeElementType> texQuad0;
  texture<float2, 1, cudaReadModeElementType> texQuad1;
  texture<float4, 1, cudaReadModeElementType> texBody;

  static __device__ __forceinline__
  float6 make_float6(float xx, float yy, float zz, float xy, float xz, float yz) {
    float6 v;
    v.xx = xx;
    v.yy = yy;
    v.zz = zz;
    v.xy = xy;
    v.xz = xz;
    v.yz = yz;
    return v;
  }

  static __device__ __forceinline__
  int ringAddr(const int i) {
    return i & (CELL_LIST_MEM_PER_WARP - 1);
  }

  static __device__ __forceinline__
  bool applyMAC(const float4 sourceCenter,
                const float3 targetCenter,
                const float3 targetSize) {
    float3 dr = make_float3(fabsf(targetCenter.x - sourceCenter.x) - (targetSize.x),
                            fabsf(targetCenter.y - sourceCenter.y) - (targetSize.y),
                            fabsf(targetCenter.z - sourceCenter.z) - (targetSize.z));
    dr.x += fabsf(dr.x); dr.x *= 0.5f;
    dr.y += fabsf(dr.y); dr.y *= 0.5f;
    dr.z += fabsf(dr.z); dr.z *= 0.5f;
    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
    return ds2 < fabsf(sourceCenter.w);
  }

  static __device__ __forceinline__
  float4 P2P(float4 acc,
             const float3 pos,
	     const float4 posj,
	     const float EPS2) {
    const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
    const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + EPS2;
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

  static __device__
  float4 M2P(float4 acc,
	     const float3 pos,
	     const float4 M0,
	     const float6 Q0,
	     float EPS2) {
    const float3 dr = make_float3(pos.x - M0.x, pos.y - M0.y, pos.z - M0.z);
    const float  r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPS2;
    const float rinv  = rsqrtf(r2);
    const float rinv2 = rinv * rinv;
    const float mrinv  = M0.w * rinv;
    const float mrinv3 = rinv2 * mrinv;
    const float mrinv5 = rinv2 * mrinv3; 
    const float mrinv7 = rinv2 * mrinv5;
    const float  D0 =  mrinv;
    const float  D1 = -mrinv3;
    const float  D2 =  mrinv5 * 3.0f;
    const float  D3 = -mrinv7 * 15.0f;
    const float q11 = Q0.xx;
    const float q22 = Q0.yy;
    const float q33 = Q0.zz;
    const float q12 = Q0.xy;
    const float q13 = Q0.xz;
    const float q23 = Q0.yz;
    const float  q  = q11 + q22 + q33;
    const float3 qR = make_float3(q11 * dr.x + q12 * dr.y + q13 * dr.z,
				  q12 * dr.x + q22 * dr.y + q23 * dr.z,
				  q13 * dr.x + q23 * dr.y + q33 * dr.z);
    const float qRR = qR.x * dr.x + qR.y * dr.y + qR.z * dr.z;
    acc.w  -= D0 + 0.5f * (D1*q + D2 * qRR);
    const float C = D1 + 0.5f * (D2*q + D3 * qRR);
    acc.x  += C * dr.x + D2 * qR.x;
    acc.y  += C * dr.y + D2 * qR.y;
    acc.z  += C * dr.z + D2 * qR.z;
    return acc;
  }

  template<int NI, bool FULL>
  static __device__
  void approxAcc(float4 acc_i[NI],
		 const float3 pos_i[NI],
		 const int cellIdx,
		 const float EPS2) {
    float4 M0, Q0;
    float2 Q1;
    if (FULL || cellIdx >= 0) {
      M0 = tex1Dfetch(texMonopole, cellIdx);
      Q0 = tex1Dfetch(texQuad0,    cellIdx);
      Q1 = tex1Dfetch(texQuad1,    cellIdx);
    } else {
      M0 = Q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      Q1 = make_float2(0.0f, 0.0f);
    }
    for (int j=0; j<WARP_SIZE; j++) {
      const float4 jM0 = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
      const float6 jQ0 = make_float6(__shfl(Q0.x, j), __shfl(Q0.y, j), __shfl(Q0.z, j), __shfl(Q0.w,j),
				     __shfl(Q1.x, j), __shfl(Q1.y, j));
#pragma unroll
      for (int k=0; k<NI; k++)
	acc_i[k] = M2P(acc_i[k], pos_i[k], jM0, jQ0, EPS2);
    }
  }

  template<int BLOCKDIM2, int NI>
  static __device__
  uint2 traverse_warp(float4 acc_i[NI],
		      const float3 pos_i[NI],
		      const float3 targetCenter,
		      const float3 targetSize,
		      const float EPS2,
		      const int2 rootRange,
		      volatile int *tempQueue,
		      int *cellQueue) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);

    uint2 counters = {0,0};

    int approxCellIdx, directBodyIdx;

    int directCounter = 0;
    int approxCounter = 0;

    for (int root=rootRange.x; root<rootRange.y; root+=WARP_SIZE)
      if (root + laneIdx < rootRange.y)
	cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx;

    int numSources = rootRange.y - rootRange.x;

    int cellQueueBlock       = 0;
    int nextLevelCellCounter = 0;

    unsigned int cellQueueOffset = 0;

    /* process level with n_cells */
    while (numSources > 0) {
      /* extract cell index from the current level cell list */
      const int cellQueueIdx = cellQueueBlock + laneIdx;
      const bool useCell    = cellQueueIdx < numSources;
      const int cellIdx     = cellQueue[ringAddr(cellQueueOffset + cellQueueIdx)];
      cellQueueBlock += min(WARP_SIZE, numSources - cellQueueBlock);

      /* read from gmem cell's info */
      const float4   sourceCenter = tex1Dfetch(texCellCenter, cellIdx);
      const CellData cellData = tex1Dfetch(texCell, cellIdx);

      const bool splitCell = applyMAC(sourceCenter, targetCenter, targetSize) ||
	(cellData.nbody() < 3); /* force to open leaves with less than 3 particles */

      /**********************************************/
      /* split cells that satisfy opening condition */
      /**********************************************/

      const bool isNode = cellData.isNode();

      const int firstChild = cellData.child();
      const int nChild= cellData.nchild();
      bool splitNode  = isNode && splitCell && useCell;

      /* use exclusive scan to compute scatter addresses for each of the child cells */
      int2 childScatter = warpIntExclusiveScan(nChild & (-splitNode));

      /* make sure we still have available stack space */
      if (childScatter.y + numSources - cellQueueBlock > CELL_LIST_MEM_PER_WARP)
	return make_uint2(0xFFFFFFFF,0xFFFFFFFF);

      /* if so populate next level stack in gmem */
      if (splitNode)
	{
	  const int scatterIdx = cellQueueOffset + numSources + nextLevelCellCounter + childScatter.x;
	  for (int i = 0; i < nChild; i++)
	    cellQueue[ringAddr(scatterIdx + i)] = firstChild + i;
	}
      nextLevelCellCounter += childScatter.y;  /* increment nextLevelCounter by total # of children */

      /***********************************/
      /******       APPROX          ******/
      /***********************************/

      /* see which thread's cell can be used for approximate force calculation */
      const bool approxCell    = !splitCell && useCell;
      const int2 approxScatter = warpBinExclusiveScan(approxCell);

      /* store index of the cell */
      const int scatterIdx = approxCounter + approxScatter.x;
      tempQueue[laneIdx] = approxCellIdx;
      if (approxCell && scatterIdx < WARP_SIZE)
	tempQueue[scatterIdx] = cellIdx;

      approxCounter += approxScatter.y;

      /* compute approximate forces */
      if (approxCounter >= WARP_SIZE)
	{
	  /* evalute cells stored in shmem */
	  approxAcc<NI,true>(acc_i, pos_i, tempQueue[laneIdx], EPS2);

	  approxCounter -= WARP_SIZE;
	  const int scatterIdx = approxCounter + approxScatter.x - approxScatter.y;
	  if (approxCell && scatterIdx >= 0)
	    tempQueue[scatterIdx] = cellIdx;
	  counters.x += WARP_SIZE;
	}
      approxCellIdx = tempQueue[laneIdx];

      /***********************************/
      /******       DIRECT          ******/
      /***********************************/

      const bool isLeaf = !isNode;
      bool isDirect = splitCell && isLeaf && useCell;

      const int body = cellData.body();
      const int numBodies = cellData.nbody();

      childScatter = warpIntExclusiveScan(numBodies & (-isDirect));
      int nParticle  = childScatter.y;
      int nProcessed = 0;
      int2 scanVal   = {0,0};

      /* conduct segmented scan for all leaves that need to be expanded */
      while (nParticle > 0)
	{
	  tempQueue[laneIdx] = 1;
	  if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
	    {
	      isDirect = false;
	      tempQueue[childScatter.x - nProcessed] = -1-body;
	    }
	  scanVal = inclusive_segscan_warp(tempQueue[laneIdx], scanVal.y);
	  const int  bodyIdx = scanVal.x;

	  if (nParticle >= WARP_SIZE)
	    {
	      const float4 M0 = tex1Dfetch(texBody, bodyIdx);
	      for (int j=0; j<WARP_SIZE; j++) {
		const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
		for (int k=0; k<NI; k++)
		  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
	      }
	      nParticle  -= WARP_SIZE;
	      nProcessed += WARP_SIZE;
	      counters.y += WARP_SIZE;
	    }
	  else 
	    {
	      const int scatterIdx = directCounter + laneIdx;
	      tempQueue[laneIdx] = directBodyIdx;
	      if (scatterIdx < WARP_SIZE)
		tempQueue[scatterIdx] = bodyIdx;

	      directCounter += nParticle;

	      if (directCounter >= WARP_SIZE)
		{
		  /* evalute cells stored in shmem */
		  const float4 M0 = tex1Dfetch(texBody, tempQueue[laneIdx]);
		  for (int j=0; j<WARP_SIZE; j++) {
		    const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
		    for (int k=0; k<NI; k++)
		      acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
		  }
		  directCounter -= WARP_SIZE;
		  const int scatterIdx = directCounter + laneIdx - nParticle;
		  if (scatterIdx >= 0)
		    tempQueue[scatterIdx] = bodyIdx;
		  counters.y += WARP_SIZE;
		}
	      directBodyIdx = tempQueue[laneIdx];

	      nParticle = 0;
	    }
	}

      /* if the current level is processed, schedule the next level */
      if (cellQueueBlock >= numSources) {
	cellQueueOffset += numSources;
	numSources = nextLevelCellCounter;
	cellQueueBlock = nextLevelCellCounter = 0;
      }

    }  /* level completed */

    if (approxCounter > 0) {
      approxAcc<NI,false>(acc_i, pos_i, laneIdx < approxCounter ? approxCellIdx : -1, EPS2);
      counters.x += approxCounter;
      approxCounter = 0;
    }

    if (directCounter > 0) {
      const int bodyIdx = laneIdx < directCounter ? directBodyIdx : -1;
      const float4 M0 = bodyIdx >= 0 ? tex1Dfetch(texBody, bodyIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int j=0; j<WARP_SIZE; j++) {
	const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	for (int k=0; k<NI; k++)
	  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
      }
      counters.y += directCounter;
      directCounter = 0;
    }

    return counters;
  }

  __device__ unsigned int retiredTargetCount = 0;
  __device__ unsigned long long sumP2PGlob = 0;
  __device__ unsigned int       maxP2PGlob = 0;
  __device__ unsigned long long sumM2PGlob = 0;
  __device__ unsigned int       maxM2PGlob = 0;

  template<int NTHREAD2, int NI>
  __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__ 
    void traverse(
		  const int numTargets,
		  const int2 *targetCells,
		  const float EPS2,
		  const int2 *levelRange,
		  const float4 *pos,
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

    while (1) {
      int targetIdx = 0;
      if (laneIdx == 0)
	targetIdx = atomicAdd(&retiredTargetCount, 1);
      targetIdx = __shfl(targetIdx, 0, WARP_SIZE);

      if (targetIdx >= numTargets) 
	return;

      const int2 target = targetCells[targetIdx];
      const int begin = target.x;
      const int end   = target.x+target.y;

      float3 pos_i[NI];
#pragma unroll
      for (int i=0; i<NI; i++) {
	const float4 body = pos[min(begin+i*WARP_SIZE+laneIdx,end-1)];
	pos_i[i] = make_float3(body.x, body.y, body.z);
      }
      float3 rmin = pos_i[0];
      float3 rmax = rmin; 
#pragma unroll
      for (int i = 0; i < NI; i++) 
	addBoxSize(rmin, rmax, pos_i[i]);
      rmin.x = __shfl(rmin.x,0);
      rmin.y = __shfl(rmin.y,0);
      rmin.z = __shfl(rmin.z,0);
      rmax.x = __shfl(rmax.x,0);
      rmax.y = __shfl(rmax.y,0);
      rmax.z = __shfl(rmax.z,0);

      const float half = 0.5f;
      const float3 targetCenter = {half*(rmax.x+rmin.x), half*(rmax.y+rmin.y), half*(rmax.z+rmin.z)};
      const float3 hvec = {half*(rmax.x-rmin.x), half*(rmax.y-rmin.y), half*(rmax.z-rmin.z)};

      float4 acc_i[NI] = {0.0f, 0.0f, 0.0f, 0.0f};

      uint2 counters = traverse_warp<NTHREAD2,NI>
	(acc_i, pos_i, targetCenter, hvec, EPS2, levelRange[1], shmem, gmem);

      assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

      const int pidx = begin + laneIdx;

      int maxP2P = counters.y;
      int sumP2P = 0;
      int maxM2P = counters.x;
      int sumM2P = 0;

#pragma unroll
      for (int i = 0; i < NI; i++)
	if (i*WARP_SIZE + pidx < end)
	  {
	    sumM2P += counters.x;
	    sumP2P += counters.y;
	  }

#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
	{
	  maxP2P  = max(maxP2P, __shfl_xor(maxP2P, 1<<i));
	  sumP2P += __shfl_xor(sumP2P, 1<<i);
	  maxM2P  = max(maxM2P, __shfl_xor(maxM2P, 1<<i));
	  sumM2P += __shfl_xor(sumM2P, 1<<i);
	}

      if (laneIdx == 0)
	{
	  atomicMax(&maxP2PGlob,                     maxP2P);
	  atomicAdd(&sumP2PGlob, (unsigned long long)sumP2P);
	  atomicMax(&maxM2PGlob,                     maxM2P);
	  atomicAdd(&sumM2PGlob, (unsigned long long)sumM2P);
	}

#pragma unroll
      for (int i=0; i<NI; i++)
	if (pidx + i * WARP_SIZE < end)
	  acc[i*WARP_SIZE + pidx] = acc_i[i];
    }
  }

  static __global__
  void direct(const int numSource,
              const float EPS2,
	      float4 *acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.x * numSource / gridDim.x;
    float pots, axs, ays ,azs;
    float potc, axc, ayc ,azc;
    float4 si = tex1Dfetch(texBody, threadIdx.x);
    __shared__ float4 s[512];
    for ( int jb=0; jb<numSource/blockDim.x/gridDim.x; jb++ ) {
      __syncthreads();
      s[threadIdx.x] = tex1Dfetch(texBody, offset+jb*blockDim.x+threadIdx.x);
      __syncthreads();
      for( int j=0; j<blockDim.x; j++ ) {
	float dx = s[j].x - si.x;
	float dy = s[j].y - si.y;
	float dz = s[j].z - si.z;
	float R2 = dx * dx + dy * dy + dz * dz + EPS2;
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

float4 Treecode::computeForces(const int numBodies,
			       const int numTargets,
			       const int numSources,
			       const float eps,
			       CellData * d_sourceCells,
			       int2 * d_targetCells,
			       float4 * d_sourceCenter,
			       float4 * d_Monopole,
			       float4 * d_Quadrupole0,
			       float2 * d_Quadrupole1,
			       int2 * d_levelRange) {
  bindTexture(computeForces::texCell,(uint4*)d_sourceCells, numSources);
  bindTexture(computeForces::texCellCenter,  d_sourceCenter,numSources);
  bindTexture(computeForces::texMonopole,    d_Monopole,    numSources);
  bindTexture(computeForces::texQuad0,       d_Quadrupole0, numSources);
  bindTexture(computeForces::texQuad1,       d_Quadrupole1, numSources);
  bindTexture(computeForces::texBody,        d_bodyPos.ptr, numBodies);

  const int NTHREAD2 = 7;
  const int NTHREAD  = 1<<NTHREAD2;
  cuda_mem<int> d_gmem_pool;

  const int nblock = 8*13;
  d_gmem_pool.alloc(CELL_LIST_MEM_PER_WARP*nblock*(NTHREAD/WARP_SIZE));

  cudaDeviceSynchronize();
  const double t0 = get_time();
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::traverse<NTHREAD2,2>, cudaFuncCachePreferL1));
  computeForces::traverse<NTHREAD2,2><<<nblock,NTHREAD>>>(numTargets, d_targetCells, eps*eps, d_levelRange,
							  d_bodyPos2, d_bodyAcc,
							  d_gmem_pool);
  kernelSuccess("traverse");
  const double dt = get_time() - t0;

  unsigned long long sumP2P, sumM2P;
  unsigned int maxP2P, maxM2P;
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumP2P, computeForces::sumP2PGlob, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxP2P, computeForces::maxP2PGlob, sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumM2P, computeForces::sumM2PGlob, sizeof(unsigned long long)));
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxM2P, computeForces::maxM2PGlob, sizeof(unsigned int)));
  float4 interactions;
  interactions.x = sumP2P * 1.0 / numBodies;
  interactions.y = maxP2P;
  interactions.z = sumM2P * 1.0 / numBodies;
  interactions.w = maxM2P;
  float flops = (interactions.x * 20 + interactions.z * 64) * numBodies / dt / 1e12;
  fprintf(stdout,"Traverse             : %.7f s (%.7f TFlops)\n",dt,flops);

  unbindTexture(computeForces::texBody);
  unbindTexture(computeForces::texQuad1);
  unbindTexture(computeForces::texQuad0);
  unbindTexture(computeForces::texMonopole);
  unbindTexture(computeForces::texCellCenter);
  unbindTexture(computeForces::texCell);
  return interactions;
}

void Treecode::computeDirect(const int numBodies, const int numTarget, const int numBlock, const float eps) {
  bindTexture(computeForces::texBody,d_bodyPos2.ptr,numBodies);
  computeForces::direct<<<numBlock,numTarget>>>(numBodies, eps*eps, d_bodyAcc2);
  unbindTexture(computeForces::texBody);
  cudaDeviceSynchronize();
}
