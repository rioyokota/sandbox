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
    int approxOffset = 0;
    int bodyOffset = 0;

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
      const int numChildScan = inclusiveScanInt(numChild);      // Inclusive scan of numChild
      const int numChildLane = numChildScan - numChild;         // Exclusive scan of numChild
      const int numChildWarp = __shfl(numChildScan, WARP_SIZE-1);// Total numChild of current warp
      sourceOffset += min(WARP_SIZE, numSources - sourceOffset);// Increment source offset
      if (numChildWarp + numSources - sourceOffset > CELL_LIST_MEM_PER_WARP)// If cell queue overflows
	return make_uint2(0xFFFFFFFF,0xFFFFFFFF);               // Exit kernel
      int childIdx = oldSources + numSources + newSources + numChildLane;// Child index of current lane
      for (int i=0; i<numChild; i++)                            // Loop over child cells for each lane
	cellQueue[ringAddr(childIdx + i)] = childBegin + i;	//  Queue child cells
      newSources += numChildWarp;                               // Increment source cell count for next loop

      // Approx
      const bool isApprox = !isClose && isSource;               // Source cell can be used for M2P
      const uint approxBallot = __ballot(isApprox);             // Gather approx flags
      const int numApproxLane = __popc(approxBallot & lanemask_lt());// Exclusive scan of approx flags
      const int numApproxWarp = __popc(approxBallot);           // Total isApprox for current warp
      int approxIdx = approxOffset + numApproxLane;             // Approx cell index of current lane
      tempQueue[laneIdx] = approxQueue;                         // Fill queue with remaining sources for approx
      if (isApprox && approxIdx < WARP_SIZE)                    // If approx flag is true and index is within bounds
	tempQueue[approxIdx] = sourceQueue;                     //  Fill approx queue with current sources
      if (approxOffset + numApproxWarp >= WARP_SIZE) {          // If approx queue is larger than the warp size
	approxAcc<NI,true>(acc_i, pos_i, tempQueue[laneIdx], EPS2);// Call M2P kernel
	approxOffset -= WARP_SIZE;                              //  Decrement approx queue size
	approxIdx = approxOffset + numApproxLane;               //  Update approx index using new queue size
	if (isApprox && approxIdx >= 0)                         //  If approx flag is true and index is within bounds
	  tempQueue[approxIdx] = sourceQueue;                   //   Fill approx queue with current sources
	counters.x += WARP_SIZE;                                //  Increment M2P counter
      }                                                         // End if for approx queue size
      approxQueue = tempQueue[laneIdx];                         // Free temp queue for use in direct
      approxOffset += numApproxWarp;                            // Increment approx queue offset

      // Direct
      const bool isLeaf = !isNode;                              // Is leaf cell
      bool isDirect = isClose && isLeaf && isSource;            // Source cell can be used for P2P
      const int bodyBegin = sourceData.body();                  // First body in cell
      const int numBodies = sourceData.nbody() & IF(isDirect);  // Number of bodies in cell
      const int numBodiesScan = inclusiveScanInt(numBodies);    // Inclusive scan of numBodies
      int numBodiesLane = numBodiesScan - numBodies;            // Exclusive scan of numBodies
      int numBodiesWarp = __shfl(numBodiesScan, WARP_SIZE-1);   // Total numBodies of current warp
      int tempOffset = 0;                                       // Initialize temp queue offset
      while (numBodiesWarp > 0) {                               // While there are bodies to process
	tempQueue[laneIdx] = 1;                                 //  Initialize body queue
	if (isDirect && (numBodiesLane < WARP_SIZE)) {          //  If direct flag is true and index is within bounds
	  isDirect = false;                                     //   Set flag as processed
	  tempQueue[numBodiesLane] = -1-bodyBegin;              //   Put body in queue
	}                                                       //  End if for direct flag
        const int bodyQueue = inclusiveSegscanInt(tempQueue[laneIdx], tempOffset);// Inclusive segmented scan of temp queue
        tempOffset = __shfl(bodyQueue, WARP_SIZE-1);            //  Last lane has the temp queue offset
	if (numBodiesWarp >= WARP_SIZE) {                       //  If warp is full of bodies
	  const float4 pos = tex1Dfetch(texBody, bodyQueue);    //   Load position of source bodies
	  for (int j=0; j<WARP_SIZE; j++) {                     //   Loop over the warp size
	    const float4 pos_j = make_float4(__shfl(pos.x, j),  //    Get source x value from lane j
					     __shfl(pos.y, j),  //    Get source y value from lane j
					     __shfl(pos.z, j),  //    Get source z value from lane j
					     __shfl(pos.w, j)); //    Get source w value from lane j
#pragma unroll                                                  //    Unroll loop
	    for (int k=0; k<NI; k++)                            //    Loop over NI targets
	      acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);  //     Call P2P kernel
	  }                                                     //   End loop over the warp size
	  numBodiesWarp -= WARP_SIZE;                           //   Decrement body queue size
	  numBodiesLane -= WARP_SIZE;                           //   Derecment lane offset of body index
	  counters.y += WARP_SIZE;                              //   Increment P2P counter
	} else {                                                //  If warp is not entirely full of bodies
	  int bodyIdx = bodyOffset + laneIdx;                   //   Body index of current lane
	  tempQueue[laneIdx] = directQueue;                     //   Initialize body queue with saved values
	  if (bodyIdx < WARP_SIZE)                              //   If body index is less than the warp size
	    tempQueue[bodyIdx] = bodyQueue;                     //    Push bodies into queue
	  bodyOffset += numBodiesWarp;                          //   Increment body queue offset
	  if (bodyOffset >= WARP_SIZE) {                        //   If this causes the body queue to spill
	    const float4 pos = tex1Dfetch(texBody, tempQueue[laneIdx]);// Load position of source bodies
	    for (int j=0; j<WARP_SIZE; j++) {                   //    Loop over the warp size
	      const float4 pos_j = make_float4(__shfl(pos.x, j),//     Get source x value from lane j
					       __shfl(pos.y, j),//     Get source y value from lane j
					       __shfl(pos.z, j),//     Get source z value from lane j
					       __shfl(pos.w, j));//    Get source w value from lane j
#pragma unroll                                                  //     Unroll loop
	      for (int k=0; k<NI; k++)                          //     Loop over NI targets
		acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);//      Call P2P kernel
	    }                                                   //    End loop over the warp size
	    bodyOffset -= WARP_SIZE;                            //    Decrement body queue size
	    bodyIdx -= WARP_SIZE;                               //    Decrement body index of current lane
	    if (bodyIdx >= 0)                                   //    If body index is valid
	      tempQueue[bodyIdx] = bodyQueue;                   //     Push bodies into queue
	    counters.y += WARP_SIZE;                            //    Increment P2P counter
	  }                                                     //   End if for body queue spill
	  directQueue = tempQueue[laneIdx];                     //   Free temp queue for use in approx
	  numBodiesWarp = 0;                                    //   Reset numBodies of current warp
	}                                                       //  End if for warp full of bodies
      }                                                         // End while loop for bodies to process
      if (sourceOffset >= numSources) {                         // If the current level is done
	oldSources += numSources;                               //  Update finished source size
	numSources = newSources;                                //  Update current source size
	sourceOffset = newSources = 0;                          //  Initialize next source size and offset
      }                                                         // End if for level finalization
    }
    if (approxOffset > 0) {
      approxAcc<NI,false>(acc_i, pos_i, laneIdx < approxOffset ? approxQueue : -1, EPS2);
      counters.x += approxOffset;
      approxOffset = 0;
    }
    if (bodyOffset > 0) {
      const int bodyQueue = laneIdx < bodyOffset ? directQueue : -1;
      const float4 pos = bodyQueue >= 0 ? tex1Dfetch(texBody, bodyQueue) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      for (int j=0; j<WARP_SIZE; j++) {
	const float4 pos_j = make_float4(__shfl(pos.x, j),
					 __shfl(pos.y, j),
					 __shfl(pos.z, j),
					 __shfl(pos.w, j));
#pragma unroll
	for (int k=0; k<NI; k++)
	  acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, EPS2);
      }
      counters.y += bodyOffset;
      bodyOffset = 0;
    }
    return counters;
  }

  __device__ unsigned long long sumP2PGlob = 0;
  __device__ unsigned int       maxP2PGlob = 0;
  __device__ unsigned long long sumM2PGlob = 0;
  __device__ unsigned int       maxM2PGlob = 0;

  template<int NTHREAD2, int NI>
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__ 
    void traverse(const int numTargets,
		  const float EPS2,
		  const int2 *levelRange,
		  const float4 *pos,
		  float4 *acc,
		  const int2 *targetRange,
		  int    *globalPool) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NTHREAD = 1<<NTHREAD2;
    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    __shared__ int sharedPool[NTHREAD];
    int *tempQueue = sharedPool + WARP_SIZE * warpIdx;
    int *cellQueue = globalPool + CELL_LIST_MEM_PER_WARP*((blockIdx.x<<NWARP2) + warpIdx);

    while (1) {
      int targetIdx = 0;
      if (laneIdx == 0)
        targetIdx = atomicAdd(&counterGlob, 1);
      targetIdx = __shfl(targetIdx, 0, WARP_SIZE);
      if (targetIdx >= numTargets) return;

      const int2 target = targetRange[targetIdx];
      const int bodyBegin = target.x;
      const int bodyEnd   = target.x+target.y;
      float3 pos_i[NI];
      for (int i=0; i<NI; i++) {
	const float4 body = pos[min(bodyBegin+i*WARP_SIZE+laneIdx,bodyEnd-1)];
	pos_i[i] = make_float3(body.x, body.y, body.z);
      }
      float3 rmin = pos_i[0];
      float3 rmax = rmin; 
      for (int i=0; i<NI; i++) 
	getMinMax(rmin, rmax, pos_i[i]);
      rmin.x = __shfl(rmin.x,0);
      rmin.y = __shfl(rmin.y,0);
      rmin.z = __shfl(rmin.z,0);
      rmax.x = __shfl(rmax.x,0);
      rmax.y = __shfl(rmax.y,0);
      rmax.z = __shfl(rmax.z,0);
      const float3 targetCenter = {.5f*(rmax.x+rmin.x), .5f*(rmax.y+rmin.y), .5f*(rmax.z+rmin.z)};
      const float3 targetSize = {.5f*(rmax.x-rmin.x), .5f*(rmax.y-rmin.y), .5f*(rmax.z-rmin.z)};
      float4 acc_i[NI] = {0.0f, 0.0f, 0.0f, 0.0f};
      const uint2 counters = traverse_warp<NTHREAD2,NI>
	(acc_i, pos_i, targetCenter, targetSize, EPS2, levelRange[1], tempQueue, cellQueue);
      assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

      int maxP2P = counters.y;
      int sumP2P = 0;
      int maxM2P = counters.x;
      int sumM2P = 0;
      const int bodyIdx = bodyBegin + laneIdx;
      for (int i=0; i<NI; i++)
	if (i*WARP_SIZE + bodyIdx < bodyEnd) {
	  sumM2P += counters.x;
	  sumP2P += counters.y;
	}
#pragma unroll
      for (int i=0; i<WARP_SIZE2; i++) {
	maxP2P  = max(maxP2P, __shfl_xor(maxP2P, 1<<i));
	sumP2P += __shfl_xor(sumP2P, 1<<i);
	maxM2P  = max(maxM2P, __shfl_xor(maxM2P, 1<<i));
	sumM2P += __shfl_xor(sumM2P, 1<<i);
      }
      if (laneIdx == 0) {
	atomicMax(&maxP2PGlob,                     maxP2P);
	atomicAdd(&sumP2PGlob, (unsigned long long)sumP2P);
	atomicMax(&maxM2PGlob,                     maxM2P);
	atomicAdd(&sumM2PGlob, (unsigned long long)sumM2P);
      }
      for (int i=0; i<NI; i++)
	if (bodyIdx + i * WARP_SIZE < bodyEnd)
	  acc[i*WARP_SIZE + bodyIdx] = acc_i[i];
    }
  }

  static __global__
    void directKernel(const int numSource,
		      const float EPS2,
		      float4 * bodyAcc) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int numChunk = (numSource - 1) / gridDim.x + 1;
    const int numWarpChunk = (numChunk - 1) / WARP_SIZE + 1;
    const int blockOffset = blockIdx.x * numChunk;
    float4 accs, accc, posi = tex1Dfetch(texBody, threadIdx.x);
    for (int jb=0; jb<numWarpChunk; jb++) {
      const int sourceIdx = min(blockOffset+jb*WARP_SIZE+laneIdx, numSource-1);
      float4 posj = tex1Dfetch(texBody, sourceIdx);
      if (sourceIdx >= numSource) posj.w = 0;
      for (int j=0; j<WARP_SIZE; j++) {
	float dx = __shfl(posj.x,j) - posi.x;
	float dy = __shfl(posj.y,j) - posi.y;
	float dz = __shfl(posj.z,j) - posi.z;
	float R2 = dx * dx + dy * dy + dz * dz + EPS2;
	float invR = rsqrtf(R2);
        float y = - __shfl(posj.w,j) * invR - accc.w;
        float t = accs.w + y;
        accc.w = (t - accs.w) - y;
        accs.w = t;
	float invR3 = invR * invR * invR * __shfl(posj.w,j);
        y = dx * invR3 - accc.x;
        t = accs.x + y;
        accc.x = (t - accs.x) - y;
        accs.x = t;
        y = dy * invR3 - accc.y;
        t = accs.y + y;
        accc.y = (t - accs.y) - y;
        accs.y = t;
        y = dz * invR3 - accc.z;
        t = accs.z + y;
        accc.z = (t - accs.z) - y;
        accs.z = t;
      }
    }
    const int targetIdx = blockIdx.x * blockDim.x + threadIdx.x;
    bodyAcc[targetIdx].x = accs.x + accc.x;
    bodyAcc[targetIdx].y = accs.y + accc.y;
    bodyAcc[targetIdx].z = accs.z + accc.z;
    bodyAcc[targetIdx].w = accs.w + accc.w;
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
		int2 * d_targetRange,
		CellData * d_sourceCells,
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
    cuda_mem<int> d_globalPool;

    const int NBLOCK = numTargets / NTHREAD;
    d_globalPool.alloc(CELL_LIST_MEM_PER_WARP*NBLOCK*(NTHREAD/WARP_SIZE));

    cudaDeviceSynchronize();
    const double t0 = get_time();
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&traverse<NTHREAD2,2>, cudaFuncCachePreferL1));
    traverse<NTHREAD2,2><<<NBLOCK,NTHREAD>>>(numTargets, eps*eps, d_levelRange,
					     d_bodyPos2, d_bodyAcc, d_targetRange, d_globalPool);
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

  void direct(const int numBodies,
	      const int numTarget,
	      const int numBlock,
	      const float eps,
	      float4 * d_bodyPos2,
	      float4 * d_bodyAcc2) {
    const int NTHREAD2 = 9;
    const int NTHREAD  = 1 << NTHREAD2;
    const int NBLOCK2  = 7;
    const int NBLOCK   = 1 << NBLOCK2;
    assert(numTarget == NTHREAD);
    assert(numBlock == NBLOCK);
    bindTexture(texBody,d_bodyPos2,numBodies);
    directKernel<<<NBLOCK,NTHREAD>>>(numBodies, eps*eps, d_bodyAcc2);
    unbindTexture(texBody);
    cudaDeviceSynchronize();
  }
};
