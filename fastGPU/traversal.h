#pragma once
#include <algorithm>

#define MEM_PER_WARP (4096 * WARP_SIZE)
#define IF(x) (-(int)(x))

namespace {
  texture<uint4,  1, cudaReadModeElementType> texCell;
  texture<float4, 1, cudaReadModeElementType> texCellCenter;
  texture<float4, 1, cudaReadModeElementType> texMultipole;
  texture<float4, 1, cudaReadModeElementType> texBody;

  static __device__ __forceinline__
    int ringAddr(const int i) {
    return i & (MEM_PER_WARP - 1);
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

  static __device__ __forceinline__
    float4 M2P(float4 acc,
	       const float3 pos,
	       const float4 * __restrict__ M,
	       float EPS2) {
    const float3 dr = make_float3(pos.x - M[0].x, pos.y - M[0].y, pos.z - M[0].z);
    const float  r2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z + EPS2;
    const float rinv  = rsqrtf(r2);
    const float rinv2 = rinv * rinv;
    const float mrinv  = M[0].w * rinv;
    const float mrinv3 = rinv2 * mrinv;
    const float mrinv5 = rinv2 * mrinv3;
    const float mrinv7 = rinv2 * mrinv5;
    const float  D0 =  mrinv;
    const float  D1 = -mrinv3;
    const float  D2 =  mrinv5 * 3.0f;
    const float  D3 = -mrinv7 * 15.0f;
    const float q11 = M[1].x;
    const float q22 = M[1].y;
    const float q33 = M[1].z;
    const float q12 = M[1].w;
    const float q13 = M[2].x;
    const float q23 = M[2].y;
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

  template<bool FULL>
    static __device__
    void approxAcc(float4 acc_i[2],
		   const float3 pos_i[2],
		   const int cellIdx,
		   const float EPS2) {
    float4 MLane[3], M[3];
    if (FULL || cellIdx >= 0) {
#pragma unroll
      for (int i=0; i<3; i++) MLane[i] = tex1Dfetch(texMultipole, 3*cellIdx+i);
    } else {
#pragma unroll
      for (int i=0; i<3; i++) MLane[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j=0; j<WARP_SIZE; j++) {
#pragma unroll
      for (int i=0; i<3; i++) M[i] = make_float4(__shfl(MLane[i].x, j), __shfl(MLane[i].y, j), __shfl(MLane[i].z, j), __shfl(MLane[i].w,j));
      for (int k=0; k<2; k++)
	acc_i[k] = M2P(acc_i[k], pos_i[k], M, EPS2);
    }
  }

  static __device__
    uint2 traverseWarp(float4 * acc_i,
		       const float3 * pos_i,
		       const float3 targetCenter,
		       const float3 targetSize,
		       const float EPS2,
		       const int2 rootRange,
		       volatile int * tempQueue,
		       int * cellQueue) {
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

      // Split
      const bool isSplit = isNode && isClose && isSource;       // Source cell must be split
      const int childBegin = sourceData.child();                // First child cell
      const int numChild = sourceData.nchild() & IF(isSplit);   // Number of child cells (masked by split flag)
      const int numChildScan = inclusiveScanInt(numChild);      // Inclusive scan of numChild
      const int numChildLane = numChildScan - numChild;         // Exclusive scan of numChild
      const int numChildWarp = __shfl(numChildScan, WARP_SIZE-1);// Total numChild of current warp
      sourceOffset += min(WARP_SIZE, numSources - sourceOffset);// Increment source offset
      if (numChildWarp + numSources - sourceOffset > MEM_PER_WARP)// If cell queue overflows
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
	approxAcc<true>(acc_i, pos_i, tempQueue[laneIdx], EPS2);// Call M2P kernel
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
	    for (int k=0; k<2; k++)                             //    Loop over two targets
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
	      for (int k=0; k<2; k++)                           //     Loop over two targets
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
      approxAcc<false>(acc_i, pos_i, laneIdx < approxOffset ? approxQueue : -1, EPS2);
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
	for (int k=0; k<2; k++)
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

  static __global__ __launch_bounds__(NTHREAD, 4)
    void traverse(const int numTargets,
		  const float EPS2,
		  const int2 * levelRange,
		  const float4 * bodyPos,
		  float4 * bodyAcc,
		  const int2 * targetRange,
		  int * globalPool) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    __shared__ int sharedPool[NTHREAD];
    int * tempQueue = sharedPool + WARP_SIZE * warpIdx;
    int * cellQueue = globalPool + MEM_PER_WARP * ((blockIdx.x<<NWARP2) + warpIdx);
    while (1) {
      int targetIdx = 0;
      if (laneIdx == 0)
        targetIdx = atomicAdd(&counterGlob, 1);
      targetIdx = __shfl(targetIdx, 0, WARP_SIZE);
      if (targetIdx >= numTargets) return;

      const int2 target = targetRange[targetIdx];
      const int bodyBegin = target.x;
      const int bodyEnd   = target.x+target.y;
      float3 pos_i[2];
      for (int i=0; i<2; i++) {
        const int bodyIdx = min(bodyBegin+i*WARP_SIZE+laneIdx, bodyEnd-1);
	const float4 pos = bodyPos[bodyIdx];
	pos_i[i] = make_float3(pos.x, pos.y, pos.z);
      }
      float3 rmin = pos_i[0];
      float3 rmax = rmin;
      for (int i=0; i<2; i++)
	getMinMax(rmin, rmax, pos_i[i]);
      rmin.x = __shfl(rmin.x,0);
      rmin.y = __shfl(rmin.y,0);
      rmin.z = __shfl(rmin.z,0);
      rmax.x = __shfl(rmax.x,0);
      rmax.y = __shfl(rmax.y,0);
      rmax.z = __shfl(rmax.z,0);
      const float3 targetCenter = {.5f*(rmax.x+rmin.x), .5f*(rmax.y+rmin.y), .5f*(rmax.z+rmin.z)};
      const float3 targetSize = {.5f*(rmax.x-rmin.x), .5f*(rmax.y-rmin.y), .5f*(rmax.z-rmin.z)};
      float4 acc_i[2] = {0.0f, 0.0f, 0.0f, 0.0f};
      const uint2 counters = traverseWarp(acc_i, pos_i, targetCenter, targetSize, EPS2,
					  
levelRange[1], tempQueue, cellQueue);
      assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

      int maxP2P = counters.y;
      int sumP2P = 0;
      int maxM2P = counters.x;
      int sumM2P = 0;
      const int bodyIdx = bodyBegin + laneIdx;
      for (int i=0; i<2; i++)
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
      for (int i=0; i<2; i++)
	if (bodyIdx + i * WARP_SIZE < bodyEnd)
	  bodyAcc[i*WARP_SIZE + bodyIdx] = acc_i[i];
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
  float4 approx(const int numTargets,
		const float eps,
		cudaVec<float4> & bodyPos,
		cudaVec<float4> & bodyPos2,
		cudaVec<float4> & bodyAcc,
		cudaVec<int2> & targetRange,
		cudaVec<CellData> & sourceCells,
		cudaVec<float4> & sourceCenter,
		cudaVec<float4> & Multipole,
		cudaVec<int2> & levelRange) {
    const int NWARP = 1 << (NTHREAD2 - WARP_SIZE2);
    const int NBLOCK = numTargets / NTHREAD;
    const int poolSize = MEM_PER_WARP * NWARP * NBLOCK;
    const int numBodies = bodyPos.size();
    const int numSources = sourceCells.size();
    sourceCells.bind(texCell);
    sourceCenter.bind(texCellCenter);
    Multipole.bind(texMultipole);
    bodyPos.bind(texBody);
    cudaVec<int> globalPool(poolSize);
    cudaDeviceSynchronize();
    const double t0 = get_time();
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&traverse, cudaFuncCachePreferL1));
    traverse<<<NBLOCK,NTHREAD>>>(numTargets, eps*eps, levelRange.d(),
				 bodyPos2.d(), bodyAcc.d(),
				 targetRange.d(), globalPool.d());
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

    sourceCells.unbind(texCell);
    sourceCenter.unbind(texCellCenter);
    Multipole.unbind(texMultipole);
    bodyPos.unbind(texBody);
    return interactions;
  }

  void direct(const int numTarget,
	      const int numBlock,
	      const float eps,
	      cudaVec<float4> & bodyPos2,
	      cudaVec<float4> & bodyAcc2) {
    const int numBodies = bodyPos2.size();
    bodyPos2.bind(texBody);
    directKernel<<<numBlock,numTarget>>>(numBodies, eps*eps, bodyAcc2.d());
    bodyPos2.unbind(texBody);
    cudaDeviceSynchronize();
  }
};
