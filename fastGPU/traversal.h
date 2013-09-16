#pragma once

#include <algorithm>

#define CELL_LIST_MEM_PER_WARP (4096*32)
#define IF(x) (-(int)(x))

namespace {
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
                const CellData sourceData,
                const float3 targetCenter,
                const float3 targetSize) {
    float3 dr = make_float3(fabsf(targetCenter.x - sourceCenter.x) - (targetSize.x),
                            fabsf(targetCenter.y - sourceCenter.y) - (targetSize.y),
                            fabsf(targetCenter.z - sourceCenter.z) - (targetSize.z));
    dr.x += fabsf(dr.x); dr.x *= 0.5f;
    dr.y += fabsf(dr.y); dr.y *= 0.5f;
    dr.z += fabsf(dr.z); dr.z *= 0.5f;
    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
    return ds2 < fabsf(sourceCenter.w) || sourceData.nbody() < 3;
  }

  static __device__ __forceinline__
  float4 P2P(float4 acc,
             const float3 pos,
	     const float4 posj,
	     const float EPS2) {
    const float3 dr    = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
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
    int approxQueue, directQueue;

    for (int root=rootRange.x; root<rootRange.y; root+=WARP_SIZE)
      if (root + laneIdx < rootRange.y)
	cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx;
    int numSources = rootRange.y - rootRange.x;
    int newSources = 0;
    int oldSources = 0;
    int sourceOffset = 0;
    int numApprox = 0;
    int numDirect = 0;

    while (numSources > 0) {
      const int sourceIdx = sourceOffset + laneIdx;             // Source cell index of current lane
      const int sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)];// Global source cell index in queue
      const float4 sourceCenter = tex1Dfetch(texCellCenter, sourceQueue);// Source cell center
      const CellData sourceData = tex1Dfetch(texCell, sourceQueue);// Source cell data
      const bool isNode = sourceData.isNode();                  // Is non-leaf cell
      const bool isClose = applyMAC(sourceCenter, sourceData, targetCenter, targetSize);// Is too close for MAC
      const bool isSource = sourceIdx < numSources;             // Source index is within bounds
      const bool isSplit = isNode && isClose && isSource;       // Source cell must be split

      // Split
      const int childBegin = sourceData.child();                // First child cell
      const int numChild = sourceData.nchild() & IF(isSplit);   // Number of child cells (masked by split flag)
      const int numChildScan = inclusiveScan<WARP_SIZE2>(numChild);// Inclusive scan of numChild
      const int childLaneIdx = numChildScan - numChild;         // Exclusive scan of numChild
      const int numChildWarp = __shfl(numChildScan, WARP_SIZE-1);// Total numChild of current warp
      sourceOffset += min(WARP_SIZE, numSources - sourceOffset);// Increment source offset
      if (numChildWarp + numSources - sourceOffset > CELL_LIST_MEM_PER_WARP)// If cell queue overflows
	return make_uint2(0xFFFFFFFF,0xFFFFFFFF);               // Exit kernel
      int childIdx = oldSources + numSources + newSources + childLaneIdx;// Child index of current lane
      for (int i=0; i<numChild; i++)                            // Loop over child cells for each lane
	cellQueue[ringAddr(childIdx + i)] = childBegin + i;	//  Queue child cells
      newSources += numChildWarp;                               // Increment source cell count for next loop

      // Approx
      const bool isApprox = !isClose && isSource;               // Source cell can be used for M2P
      const uint approxBallot = __ballot(isApprox);             // Gather approx flags
      const int approxLaneIdx = __popc(approxBallot & lanemask_lt());// Exclusive scan of approx flags
      const int numApproxWarp = __popc(approxBallot);           // Total isApprox for current warp
      int approxIdx = numApprox + approxLaneIdx;                // Approx cell index of current lane
      tempQueue[laneIdx] = approxQueue;                         // Fill queue with remaining sources for approx
      if (isApprox && approxIdx < WARP_SIZE)                    // If approx flag is true and index is within bounds
	tempQueue[approxIdx] = sourceQueue;                     //  Fill approx queue with current sources
      if (numApprox + numApproxWarp >= WARP_SIZE) {             // If approx queue is larger than the warp size
	approxAcc<NI,true>(acc_i, pos_i, tempQueue[laneIdx], EPS2);// Call M2P kernel
	numApprox -= WARP_SIZE;                                 //  Decrement approx queue size
	approxIdx = numApprox + approxLaneIdx;                  //  Update approx index using new queue size
	if (isApprox && approxIdx >= 0)                         //  If approx flag is true and index is within bounds
	  tempQueue[approxIdx] = sourceQueue;                   //   Fill approx queue with current sources
	counters.x += WARP_SIZE;                                //  Increment M2P counter
      }                                                         // End if for approx queue size
      approxQueue = tempQueue[laneIdx];                         // Free temp queue for use in direct
      numApprox += numApproxWarp;                               // Increment approx queue offset

      // Direct
      const bool isLeaf = !isNode;
      bool isDirect = isClose && isLeaf && isSource;
      const int bodyBegin = sourceData.body();
      const int numBodies = sourceData.nbody() & IF(isDirect);
      const int numBodiesScan = inclusiveScan<WARP_SIZE2>(numBodies);
      int bodyLaneIdx = numBodiesScan - numBodies;
      int numBodiesWarp = __shfl(numBodiesScan, WARP_SIZE-1);
      int2 scanVal = {0,0};

      while (numBodiesWarp > 0) {
	tempQueue[laneIdx] = 1;
	if (isDirect && (bodyLaneIdx < WARP_SIZE)) {
	  isDirect = false;
	  tempQueue[bodyLaneIdx] = -1-bodyBegin;
	}
	scanVal = inclusive_segscan_warp(tempQueue[laneIdx], scanVal.y);
	const int  bodyIdx = scanVal.x;

	if (numBodiesWarp >= WARP_SIZE) {
	  const float4 M0 = tex1Dfetch(texBody, bodyIdx);
	  for (int j=0; j<WARP_SIZE; j++) {
	    const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	    for (int k=0; k<NI; k++)
	      acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
	    }
	  numBodiesWarp -= WARP_SIZE;
	  bodyLaneIdx -= WARP_SIZE;
	  counters.y += WARP_SIZE;
	} else {
	  const int scatterIdx = numDirect + laneIdx;
	  tempQueue[laneIdx] = directQueue;
	  if (scatterIdx < WARP_SIZE)
	    tempQueue[scatterIdx] = bodyIdx;

	  numDirect += numBodiesWarp;

	  if (numDirect >= WARP_SIZE) {
	    const float4 M0 = tex1Dfetch(texBody, tempQueue[laneIdx]);
	    for (int j=0; j<WARP_SIZE; j++) {
	      const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	      for (int k=0; k<NI; k++)
		acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
	    }
	    numDirect -= WARP_SIZE;
	    const int scatterIdx = numDirect + laneIdx - numBodiesWarp;
	    if (scatterIdx >= 0)
	      tempQueue[scatterIdx] = bodyIdx;
	      counters.y += WARP_SIZE;
	  }
	  directQueue = tempQueue[laneIdx];
	  numBodiesWarp = 0;
	}
      }
      if (sourceOffset >= numSources) {                         // If the current level is done
	oldSources += numSources;                               //  Update finished source size
	numSources = newSources;                                //  Update current source size
	sourceOffset = newSources = 0;                          //  Initialize next source size and offset
      }
    }

    if (numApprox > 0) {
      approxAcc<NI,false>(acc_i, pos_i, laneIdx < numApprox ? approxQueue : -1, EPS2);
      counters.x += numApprox;
      numApprox = 0;
    }

    if (numDirect > 0) {
      const int bodyIdx = laneIdx < numDirect ? directQueue : -1;
      const float4 M0 = bodyIdx >= 0 ? tex1Dfetch(texBody, bodyIdx) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int j=0; j<WARP_SIZE; j++) {
	const float4 pos_j = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
#pragma unroll
	for (int k=0; k<NI; k++)
	  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
      }
      counters.y += numDirect;
      numDirect = 0;
    }

    return counters;
  }

  __device__ unsigned int   retiredTargets = 0;
  __device__ unsigned long long sumP2PGlob = 0;
  __device__ unsigned int       maxP2PGlob = 0;
  __device__ unsigned long long sumM2PGlob = 0;
  __device__ unsigned int       maxM2PGlob = 0;

  template<int NTHREAD2, int NI>
  __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__ 
    void traverse(const int numTargets,
		  const int2 *targetCells,
		  const float EPS2,
		  const int2 *levelRange,
		  const float4 *pos,
		  float4 *acc,
		  int    *gmem_pool) {
    const int NTHREAD = 1<<NTHREAD2;
    __shared__ int shmem_pool[NTHREAD];

    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    int *shmem = shmem_pool + WARP_SIZE * warpIdx;
    int *gmem  =  gmem_pool + CELL_LIST_MEM_PER_WARP*((blockIdx.x<<NWARP2) + warpIdx);

    while (1) {
      int targetIdx = 0;
      if (laneIdx == 0)
        targetIdx = atomicAdd(&retiredTargets, 1);
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
      for (int i=0; i<NI; i++) 
	getMinMax(rmin, rmax, pos_i[i]);
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

  template<int NTHREAD2>
  static __global__
  void directKernel(const int numSource,
		    const float EPS2,
		    float4 *acc) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NTHREAD = 1 << NTHREAD2;
    const int i = blockIdx.x * NTHREAD + threadIdx.x;
    int offset = blockIdx.x * numSource / gridDim.x;
    float pots, axs, ays ,azs;
    float potc, axc, ayc ,azc;
    float4 si = tex1Dfetch(texBody, threadIdx.x);
    __shared__ float4 shared[NTHREAD];
    float4 * s = shared + WARP_SIZE * warpIdx;
    for ( int jb=0; jb<numSource/WARP_SIZE/gridDim.x; jb++ ) {
      s[laneIdx] = tex1Dfetch(texBody, offset+jb*WARP_SIZE+laneIdx);
      for( int j=0; j<WARP_SIZE; j++ ) {
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

class Traversal {
 public:
  float4 approx(const int numBodies,
		const int numTargets,
		const int numSources,
		const float eps,
		float4 * d_bodyPos,
		float4 * d_bodyPos2,
		float4 * d_bodyAcc,
		CellData * d_sourceCells,
		int2 * d_targetCells,
		float4 * d_sourceCenter,
		float4 * d_Monopole,
		float4 * d_Quadrupole0,
		float2 * d_Quadrupole1,
		int2 * d_levelRange) {
    bindTexture(texCell,(uint4*)d_sourceCells, numSources);
    bindTexture(texCellCenter,  d_sourceCenter,numSources);
    bindTexture(texMonopole,    d_Monopole,    numSources);
    bindTexture(texQuad0,       d_Quadrupole0, numSources);
    bindTexture(texQuad1,       d_Quadrupole1, numSources);
    bindTexture(texBody,        d_bodyPos,     numBodies);

    const int NTHREAD2 = 7;
    const int NTHREAD  = 1<<NTHREAD2;
    cuda_mem<int> d_gmem_pool;

    const int nblock = 8*13;
    d_gmem_pool.alloc(CELL_LIST_MEM_PER_WARP*nblock*(NTHREAD/WARP_SIZE));

    cudaDeviceSynchronize();
    const double t0 = get_time();
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&traverse<NTHREAD2,2>, cudaFuncCachePreferL1));
    traverse<NTHREAD2,2><<<nblock,NTHREAD>>>(numTargets, d_targetCells, eps*eps, d_levelRange,
							    d_bodyPos2, d_bodyAcc,
							    d_gmem_pool);
    kernelSuccess("traverse");
    const double dt = get_time() - t0;

    unsigned long long sumP2P, sumM2P;
    unsigned int maxP2P, maxM2P;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumP2P, sumP2PGlob, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxP2P, maxP2PGlob, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumM2P, sumM2PGlob, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxM2P, maxM2PGlob, sizeof(unsigned int)));
    float4 interactions;
    interactions.x = sumP2P * 1.0 / numBodies;
    interactions.y = maxP2P;
    interactions.z = sumM2P * 1.0 / numBodies;
    interactions.w = maxM2P;
    float flops = (interactions.x * 20 + interactions.z * 64) * numBodies / dt / 1e12;
    fprintf(stdout,"Traverse             : %.7f s (%.7f TFlops)\n",dt,flops);

    unbindTexture(texBody);
    unbindTexture(texQuad1);
    unbindTexture(texQuad0);
    unbindTexture(texMonopole);
    unbindTexture(texCellCenter);
    unbindTexture(texCell);
    return interactions;
  }

  void direct(const int numBodies, const int numTarget, const int numBlock, const float eps,
	      float4 * d_bodyPos2, float4 * d_bodyAcc2) {
    const int NTHREAD2 = 9;
    const int NTHREAD  = 1 << NTHREAD2;
    const int NBLOCK2  = 7;
    const int NBLOCK   = 1 << NBLOCK2;
    assert(numTarget == NTHREAD);
    assert(numBlock == NBLOCK);
    bindTexture(texBody,d_bodyPos2,numBodies);
    directKernel<9><<<NBLOCK,NTHREAD>>>(numBodies, eps*eps, d_bodyAcc2);
    unbindTexture(texBody);
    cudaDeviceSynchronize();
  }
};
