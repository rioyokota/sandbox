#pragma once

#define NWARPS2 3
#define NWARPS (1<<NWARPS2)

extern void sort(const int size, int * key, int * value);

namespace {
  __constant__ int maxNodeGlob;
  __device__ unsigned int counterGlob = 0;
  __device__ unsigned int numChildGlob = 0;
  __device__ unsigned int numLeafsGlob = 0;
  __device__ unsigned int numLevelsGlob = 0;
  __device__ unsigned int numPerLeafGlob = 0;
  __device__ unsigned int numCellsGlob = 0;
  __device__ int * octantSizePool;
  __device__ int * octantSizeScanPool;
  __device__ int * subOctantSizeScanPool;
  __device__ int * blockCounterPool;
  __device__ int2 * bodyRangePool;
  __device__ CellData * sourceCells;

  static __device__ __forceinline__
  int getOctant(const float4 &box, const float4 &body) {
    return ((box.x <= body.x) << 0) + ((box.y <= body.y) << 1) + ((box.z <= body.z) << 2);
  }

  static __device__ __forceinline__
  float4 getChild(const float4 &box, const int octant) {
    const float R = 0.5f * box.w;
    return make_float4(box.x + R * (octant & 1 ? 1.0f : -1.0f),
		       box.y + R * (octant & 2 ? 1.0f : -1.0f),
		       box.z + R * (octant & 4 ? 1.0f : -1.0f),
		       R);
  }

  static __device__ __forceinline__
  void getKernelSize(dim3 &grid, dim3 &block, const int N) {
    const int NTHREADS = NWARPS * WARP_SIZE;
    const int NGRIDS = min(max(N / NTHREADS, 1), 512);
    block = dim3(NTHREADS);
    assert(N > 0);
    grid = dim3(NGRIDS);
  }

  template<int NTHREAD2>
  static __device__
  float2 getMinMax(float2 range) {
    const int NTHREAD = 1 << NTHREAD2;
    extern __shared__ float shared[];
    float *sharedMin = shared;
    float *sharedMax = shared + NTHREAD;
    sharedMin[threadIdx.x] = range.x;
    sharedMax[threadIdx.x] = range.y;
    __syncthreads();
#pragma unroll
    for (int i=NTHREAD2-1; i>=6; i--) {
      const int offset = 1 << i;
      if (threadIdx.x < offset) {
	sharedMin[threadIdx.x] = range.x = fminf(range.x, sharedMin[threadIdx.x + offset]);
	sharedMax[threadIdx.x] = range.y = fmaxf(range.y, sharedMax[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    if (threadIdx.x < WARP_SIZE) {
      volatile float *warpMin = sharedMin;
      volatile float *warpMax = sharedMax;
#pragma unroll
      for (int i=5; i>=0; i--) {
	const int offset = 1 << i;
	warpMin[threadIdx.x] = range.x = fminf(range.x, warpMin[threadIdx.x + offset]);
	warpMax[threadIdx.x] = range.y = fmaxf(range.y, warpMax[threadIdx.x + offset]);
      }
    }
    __syncthreads();
    return range;
  }

  template<const int NTHREAD2>
  static __global__
  void getBounds(const int numBodies,
		 float3 * bounds,
		 float4 * domain,
		 const float4 * bodyPos) {
    const int NTHREAD = 1 << NTHREAD2;
    const int NBLOCK = NTHREAD;
    const int begin = blockIdx.x * NTHREAD + threadIdx.x;
    float3 Xmin = {bodyPos[0].x, bodyPos[0].y, bodyPos[0].z};
    float3 Xmax = Xmin;
    for (int i=begin; i<numBodies; i+=NBLOCK*NTHREAD) {
      if (i < numBodies) {
	const float4 pos = bodyPos[i];
	Xmin.x = fmin(Xmin.x, pos.x);
	Xmin.y = fmin(Xmin.y, pos.y);
	Xmin.z = fmin(Xmin.z, pos.z);
	Xmax.x = fmax(Xmax.x, pos.x);
	Xmax.y = fmax(Xmax.y, pos.y);
	Xmax.z = fmax(Xmax.z, pos.z);
      }
    }
    float2 range;
    range = getMinMax<NTHREAD2>(make_float2(Xmin.x, Xmax.x)); Xmin.x = range.x; Xmax.x = range.y;
    range = getMinMax<NTHREAD2>(make_float2(Xmin.y, Xmax.y)); Xmin.y = range.x; Xmax.y = range.y;
    range = getMinMax<NTHREAD2>(make_float2(Xmin.z, Xmax.z)); Xmin.z = range.x; Xmax.z = range.y;
    if (threadIdx.x == 0) {
      bounds[blockIdx.x         ] = Xmin;
      bounds[blockIdx.x + NBLOCK] = Xmax;
    }
    __shared__ bool lastBlock;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const int blockCount = atomicInc(&counterGlob, NBLOCK);
      lastBlock = (blockCount == NBLOCK-1);
    }
    __syncthreads();
    if (lastBlock) {
      Xmin = bounds[threadIdx.x];
      Xmax = bounds[threadIdx.x + NBLOCK];
      range = getMinMax<NTHREAD2>(make_float2(Xmin.x, Xmax.x)); Xmin.x = range.x; Xmax.x = range.y;
      range = getMinMax<NTHREAD2>(make_float2(Xmin.y, Xmax.y)); Xmin.y = range.x; Xmax.y = range.y;
      range = getMinMax<NTHREAD2>(make_float2(Xmin.z, Xmax.z)); Xmin.z = range.x; Xmax.z = range.y;
      __syncthreads();
      if (threadIdx.x == 0) {
	const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
	const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
	const float r = fmax(R.x, fmax(R.y, R.z)) * 1.1f;
	const float4 box = {X.x, X.y, X.z, r};
	*domain = box;
	counterGlob = 0;
      }
    }
  }

  template<int NLEAF, bool ISROOT>
  static __global__ __launch_bounds__(256, 8)
  void buildOctant(float4 box,
		   const int cellParentIndex,
		   const int cellIndexBase,
		   const int packedOctant,
		   int * octantSizeBase,
		   int * octantSizeScanBase,
		   int * subOctantSizeScanBase,
		   int * blockCounterBase,
		   int2 * bodyRangeBase,
		   float4 * bodyPos,
		   float4 * bodyPos2,
		   const int level = 0) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;

    int * octantSize = octantSizeBase + blockIdx.y * 8;
    int * octantSizeScan = octantSizeScanBase + blockIdx.y * 8;
    int * subOctantSizeScan = subOctantSizeScanBase + blockIdx.y * 64;
    int * blockCounter = blockCounterBase + blockIdx.y;
    int2 * bodyRange = bodyRangeBase + blockIdx.y;

    const int bodyBegin = bodyRange->x + blockIdx.x * blockDim.x + warpIdx * WARP_SIZE;
    const int bodyEnd   = bodyRange->y;
    const int numBodies = bodyRange->y - bodyRange->x;

    const int childOctant = (packedOctant >> (3*blockIdx.y)) & 0x7;
    if (!ISROOT) box = getChild(box, childOctant);

    __shared__ int subOctantSizeLane[NWARPS*8*8];
    __shared__ int subOctantSize[8*8];

    float4 *childBox = (float4*)subOctantSize;

    for (int i=0; i<8*8*NWARPS; i+=blockDim.x)
      if (i+threadIdx.x < 8*8*NWARPS)
	subOctantSizeLane[i+threadIdx.x] = 0;

    if (laneIdx == 0)
      childBox[warpIdx] = getChild(box, warpIdx);               // One child per warp

    __syncthreads();

    for (int i=bodyBegin; i<bodyEnd; i+=gridDim.x*blockDim.x) {
      const int bodyIdx = min(i+laneIdx, bodyEnd-1);
      float4 pos = bodyPos[bodyIdx];
      int bodyOctant = getOctant(box, pos);
      int bodySubOctant = getOctant(childBox[bodyOctant], pos);  
      if (i+laneIdx > bodyIdx)
	bodyOctant = bodySubOctant = 8;                         // Out of bounds lanes

      int octantSizeScanLane = 0;
#pragma unroll
      for (int octant=0; octant<8; octant++) {
	const int sumLane = reduceBool(bodyOctant == octant);   // Count current octant in warp
	if (octant == laneIdx)
	  octantSizeScanLane = sumLane;                          // Use lanes 0-7 for each octant
      }
      int octantOffset;
      if (laneIdx < 8)
	octantOffset = atomicAdd(&octantSizeScan[laneIdx], octantSizeScanLane);// Global scan

      int bodyIdx2 = -1;
#pragma unroll
      for (int octant=0; octant<8; octant++) {
	const int sumLane = reduceBool(bodyOctant == octant);
	if (sumLane > 0) {                                      // Avoid redundant instructions
	  const int index = exclusiveScanBool(bodyOctant == octant);// Sparse lane index
	  const int offset = __shfl(octantOffset, octant);      // Global offset
	  if (bodyOctant == octant)                             // Prevent overwrite
	    bodyIdx2 = offset + index;                          // Sorted index
	}
      }
      if (bodyIdx2 >= 0)
        bodyPos2[bodyIdx2] = pos;                               // Assign value to sort buffer

      int remainder = 32;
#pragma unroll
      for (int octant=0; octant<8; octant++) {
	if (remainder == 0) break;
	const int sumLane = reduceBool(bodyOctant == octant);
	if (sumLane > 0) {
	  const int bodySubOctantValid = bodyOctant == octant ? bodySubOctant : 8;
#pragma unroll
	  for (int subOctant=0; subOctant<8; subOctant+=4) {
	    const int4 sum4 = make_int4(reduceBool(subOctant+0 == bodySubOctantValid),
					reduceBool(subOctant+1 == bodySubOctantValid),
					reduceBool(subOctant+2 == bodySubOctantValid),
					reduceBool(subOctant+3 == bodySubOctantValid));
	    if (laneIdx == 0) {
	      int4 subOctantTemp = *(int4*)&subOctantSizeLane[warpIdx*64+octant*8+subOctant];
	      subOctantTemp.x += sum4.x;
	      subOctantTemp.y += sum4.y;
	      subOctantTemp.z += sum4.z;
	      subOctantTemp.w += sum4.w;
	      *(int4*)&subOctantSizeLane[warpIdx*64+octant*8+subOctant] = subOctantTemp;
	    }
	  }
	  remainder -= sumLane;
	}
      }
    }
    __syncthreads();                                            // Sync subOctantSizeLane

#pragma unroll
    for (int k=0; k<8; k+=4) {
      int4 subOctantTemp = laneIdx < NWARPS ? (*(int4*)&subOctantSizeLane[laneIdx*64+warpIdx*8+k]) : make_int4(0,0,0,0);
#pragma unroll
      for (int i=NWARPS2-1; i>=0; i--) {
	subOctantTemp.x += __shfl_xor(subOctantTemp.x, 1<<i, NWARPS);
	subOctantTemp.y += __shfl_xor(subOctantTemp.y, 1<<i, NWARPS);
	subOctantTemp.z += __shfl_xor(subOctantTemp.z, 1<<i, NWARPS);
	subOctantTemp.w += __shfl_xor(subOctantTemp.w, 1<<i, NWARPS);
      }
      if (laneIdx == 0)
	*(int4*)&subOctantSize[warpIdx*8+k] = subOctantTemp;
    }
    if (laneIdx < 8)
      if (subOctantSize[warpIdx*8+laneIdx] > 0)
	atomicAdd(&subOctantSizeScan[warpIdx*8+laneIdx], subOctantSize[warpIdx*8+laneIdx]);
    __syncthreads();                                            // Sync subOctantSizeScan, subOctantSize

    __shared__ bool lastBlock;
    if (threadIdx.x == 0) {
      const int blockCount = atomicAdd(blockCounter, 1);
      lastBlock = (blockCount == gridDim.x-1);
    }
    __syncthreads();                                            // Sync lastBlock

    if (!lastBlock) return;
    __syncthreads();                                            // Sync return

    if (threadIdx.x == 0)
      atomicCAS(&numLevelsGlob, level, level+1);
    __syncthreads();                                            // Sync numLevelsGlob

    const int numBodiesOctant = octantSize[warpIdx];
    const int bodyEndOctant = octantSizeScan[warpIdx];
    const int bodyBeginOctant = bodyEndOctant - numBodiesOctant;
    const int numBodiesOctantLane = laneIdx < 8 ? octantSize[laneIdx] : 0;
    const int numChildLane = exclusiveScanBool(numBodiesOctantLane > NLEAF);
    const int numLeafsLane = exclusiveScanBool(0 < numBodiesOctantLane && numBodiesOctantLane <= NLEAF);
    int * numChild = subOctantSize;
    int * numLeafs = subOctantSize + 8;
    if (warpIdx == 0 && laneIdx < 8) {
      numChild[laneIdx] = numChildLane;
      numLeafs[laneIdx] = numLeafsLane;
    }

    int maxBodiesOctant = numBodiesOctantLane;
#pragma unroll
    for (int i=2; i>=0; i--)
      maxBodiesOctant = max(maxBodiesOctant, __shfl_xor(maxBodiesOctant, 1<<i));
    const int numChildWarp = reduceBool(numBodiesOctantLane > NLEAF);
    /* if there is at least one cell to split, increment nuumber of the nodes */
    if (threadIdx.x == 0 && numChildWarp > 0) {
      subOctantSize[16] = atomicAdd(&numChildGlob,numChildWarp);
#if 1   /* temp solution, a better one is to use RingBuffer */
      assert(subOctantSize[16] < maxNodeGlob);
#endif
    }

    /* writing linking info, parent, child and particle's list */
    const int nChildrenCell = reduceBool(numBodiesOctantLane > 0);
    if (threadIdx.x == 0 && nChildrenCell > 0)
      {
	const int cellFirstChildIndex = atomicAdd(&numCellsGlob, nChildrenCell);
	/*** keep in mind, the 0-level will be overwritten ***/
	assert(nChildrenCell > 0);
	assert(nChildrenCell <= 8);
	const CellData cellData(level,cellParentIndex, bodyRange->x, numBodies, cellFirstChildIndex, nChildrenCell-1);
	assert(cellData.child() < numCellsGlob);
	assert(cellData.isNode());
	sourceCells[cellIndexBase + blockIdx.y] = cellData;
	subOctantSize[17] = cellFirstChildIndex;
      }

    __syncthreads();
    const int cellFirstChildIndex = subOctantSize[17];
    /* compute atomic data offset for cell that need to be split */
    const int next_node = subOctantSize[16];
    octantSizeBase = octantSizePool + next_node * 8;
    octantSizeScanBase = octantSizeScanPool + next_node * 8;
    subOctantSizeScanBase = subOctantSizeScanPool + next_node * 64;
    blockCounterBase = blockCounterPool + next_node;
    bodyRangeBase = bodyRangePool + next_node;

    const int nodeOffset = numChild[warpIdx];
    const int leafOffset = numLeafs[warpIdx];

    /* if cell needs to be split, populate it shared atomic data */
    if (numBodiesOctant > NLEAF)
      {
	octantSize = octantSizeBase + nodeOffset * 8;
	octantSizeScan = octantSizeScanBase + nodeOffset * 8;
	blockCounter = blockCounterBase + nodeOffset;

	/* number of particles in each cell's subcells */
	const int nSubCell = laneIdx < 8 ? subOctantSizeScan[warpIdx*8 + laneIdx] : 0;

	/* compute offsets */
        int cellOffset = inclusiveScanInt<3>(nSubCell);
	cellOffset -= nSubCell;

	/* store offset in memory */

	cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
	if (laneIdx < 8) cellOffset = nSubCell;
	else             cellOffset += bodyBeginOctant;
	if (laneIdx < 8)
	  octantSize[laneIdx] = nSubCell;
        else if (laneIdx < 16)
	  octantSizeScan[laneIdx-8] = cellOffset;
        if (laneIdx == 0) *blockCounter = 0;
	bodyRange = bodyRangeBase + nodeOffset;
        if (laneIdx == 1) bodyRange->x = bodyBeginOctant;
        if (laneIdx == 2) bodyRange->y = bodyEndOctant;
      }

    /***************************/
    /*  launch  child  kernel  */
    /***************************/

    /* warps coorperate so that only 1 kernel needs to be launched by a thread block
     * with larger degree of paralellism */
    if (numChildWarp > 0 && warpIdx == 0)
      {
	/* build octant mask */
	int packedOctant = numBodiesOctantLane > NLEAF ?  (laneIdx << (3*numChildLane)) : 0;
#pragma unroll
	for (int i = 4; i >= 0; i--)
	  packedOctant |= __shfl_xor(packedOctant, 1<<i, WARP_SIZE);

	if (threadIdx.x == 0)
	  {
	    dim3 grid, block;
	    getKernelSize(grid, block, maxBodiesOctant);
	    grid.y = numChildWarp;  /* each y-coordinate of the grid will be busy for each parent cell */
	    buildOctant<NLEAF,false><<<grid,block>>>
	      (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
	       packedOctant, octantSizeBase, octantSizeScanBase, subOctantSizeScanBase, blockCounterBase, bodyRangeBase, bodyPos2, bodyPos, level+1);
	  }
      }

    /******************/
    /* process leaves */
    /******************/

    if (numBodiesOctant <= NLEAF && numBodiesOctant > 0)
      {
	if (laneIdx == 0)
	  {
	    atomicAdd(&numLeafsGlob,1);
	    atomicAdd(&numPerLeafGlob, bodyEndOctant-bodyBeginOctant);
	    const CellData leafData(level+1, cellIndexBase+blockIdx.y, bodyBeginOctant, bodyEndOctant-bodyBeginOctant);
	    assert(leafData.isLeaf());
	    sourceCells[cellFirstChildIndex + numChildWarp + leafOffset] = leafData;
	  }
	if (!(level&1))
	  {
	    for (int i = bodyBeginOctant+laneIdx; i < bodyEndOctant; i += WARP_SIZE)
	      if (i < bodyEndOctant) {
		float4 pos = bodyPos2[i];
		pos.w = 8;
		bodyPos[i] = pos;
	      }
	  }
	else
	  {
	    for (int i = bodyBeginOctant+laneIdx; i < bodyEndOctant; i += WARP_SIZE)
	      if (i < bodyEndOctant) {
		float4 pos = bodyPos2[i];
		pos.w = 8;
		bodyPos2[i] = pos;
	      }
	  }
      }
  }

  static __global__ void getRootOctantSize(const int numBodies,
				           int * octantSize,
				           const float4 box,
				           const float4 *bodyPos) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int begin = blockIdx.x * blockDim.x + threadIdx.x;
    int octantSizeLane[8] = {0};
    for (int i=begin; i<numBodies; i+=gridDim.x*blockDim.x) {
      const float4 pos = bodyPos[i];
      const int octant = getOctant(box, pos);
      octantSizeLane[0] += (octant == 0);
      octantSizeLane[1] += (octant == 1);
      octantSizeLane[2] += (octant == 2);
      octantSizeLane[3] += (octant == 3);
      octantSizeLane[4] += (octant == 4);
      octantSizeLane[5] += (octant == 5);
      octantSizeLane[6] += (octant == 6);
      octantSizeLane[7] += (octant == 7);
    }
#pragma unroll
    for (int k=0; k<8; k++) {
      int octantSizeTemp = octantSizeLane[k];
#pragma unroll
      for (int i=4; i>=0; i--)
	octantSizeTemp += __shfl_xor(octantSizeTemp, 1<<i);
      if (laneIdx == 0)
	atomicAdd(&octantSize[k], octantSizeTemp);
    }
  }

  template<int NLEAF>
    static __global__ void buildOctree(const int numBodies,
				       const float4 *domain,
				       CellData * d_sourceCells,
				       int * d_octantSizePool,
				       int * d_octantSizeScanPool,
				       int * d_subOctantSizeScanPool,
				       int * d_blockCounterPool,
				       int2 * d_bodyRangePool,
				       float4 * d_bodyPos,
				       float4 * d_bodyPos2,
				       int * numCellsGlob_return = NULL)
    {
      sourceCells = d_sourceCells;
      octantSizePool = d_octantSizePool;
      octantSizeScanPool = d_octantSizeScanPool;
      subOctantSizeScanPool = d_subOctantSizeScanPool;
      blockCounterPool = d_blockCounterPool;
      bodyRangePool = d_bodyRangePool;

      int *octantSize = new int[8];
      for (int k=0; k<8; k++)
	octantSize[k] = 0;
      getRootOctantSize<<<256, 256>>>(numBodies, octantSize, *domain, d_bodyPos);
      assert(cudaGetLastError() == cudaSuccess);
      cudaDeviceSynchronize();

      int * octantSizeScan = new int[8];
      int * subOctantSizeScan = new int[64];
      int * blockCounter = new int;
      int2 * bodyRange = new int2;
#pragma unroll
      for (int k=0; k<8; k++) {
	octantSizeScan[k] = k == 0 ? 0 : octantSizeScan[k-1] + octantSize[k-1];
      }
#pragma unroll
      for (int k=0; k<64; k++)
	subOctantSizeScan[k] = 0;

      numChildGlob = 0;
      numLeafsGlob = 0;
      numLevelsGlob = 0;
      numCellsGlob  = 0;
      numPerLeafGlob = 0;

      *blockCounter = 0;
      bodyRange->x = 0;
      bodyRange->y = numBodies;

      dim3 grid, block;
      getKernelSize(grid, block, numBodies);
      buildOctant<NLEAF,true><<<grid, block>>>
	(*domain, 0, 0, 0, octantSize, octantSizeScan, subOctantSizeScan, blockCounter, bodyRange, d_bodyPos, d_bodyPos2);
      assert(cudaDeviceSynchronize() == cudaSuccess);

      if (numCellsGlob_return != NULL)
	*numCellsGlob_return = numCellsGlob;

      delete [] octantSize;
      delete [] octantSizeScan;
      delete [] subOctantSizeScan;
      delete [] blockCounter;
      delete [] bodyRange;
    }


  static __global__ void
    get_cell_levels(const int n, const CellData sourceCells[], CellData sourceCells2[], int key[], int value[])
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const CellData cell = sourceCells[idx];
    key  [idx] = cell.level();
    value[idx] = idx;
    sourceCells2[idx] = cell;
  }

  static __global__ void
    write_newIdx(const int n, const int value[], int moved_to_idx[])
  {
    const int newIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (newIdx >= n) return;

    const int oldIdx = value[newIdx];
    moved_to_idx[oldIdx] = newIdx;
  }


  static __global__ void
    getLevelRange(const int n, const int levels[], int2 levelRange[])
  {
    const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx >= n) return;

    extern __shared__ int shLevels[];

    const int tid = threadIdx.x;
    shLevels[tid+1] = levels[gidx];

    int shIdx = 0;
    int gmIdx = max(blockIdx.x*blockDim.x-1,0);
    if (tid == 1)
      {
	shIdx = blockDim.x+1;
	gmIdx = min(blockIdx.x*blockDim.x + blockDim.x,n-1);
      }
    if (tid < 2)
      shLevels[shIdx] = levels[gmIdx];

    __syncthreads();

    const int idx = tid+1;
    const int currLevel = shLevels[idx];
    const int prevLevel = shLevels[idx-1];
    if (currLevel != prevLevel || gidx == 0)
      levelRange[currLevel].x = gidx;

    const int nextLevel = shLevels[idx+1];
    if (currLevel != nextLevel || gidx == n-1)
      levelRange[currLevel].y = gidx+1;
  }

  __device__  unsigned int leafIdx_counter = 0;
  static __global__ void
    shuffle_cells(const int n, const int value[], const int moved_to_idx[], const CellData sourceCells2[], CellData sourceCells[])
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int mapIdx = value[idx];
    CellData cell = sourceCells2[mapIdx];
    if (cell.isNode())
      {
	const int firstOld = cell.child();
	const int firstNew = moved_to_idx[firstOld];
	cell.setChild(firstNew);
      }
    if (cell.parent() > 0)
      cell.setParent(moved_to_idx[cell.parent()]);

    sourceCells[idx] = cell;

    if (threadIdx.x == 0 && blockIdx.x == 0)
      leafIdx_counter = 0;
  }

  template<int NTHREAD2>
    static __global__
    void collect_leaves(const int n, const CellData *sourceCells, int *leafCells)
    {
      const int gidx = blockDim.x*blockIdx.x + threadIdx.x;

      const CellData cell = sourceCells[min(gidx,n-1)];

      __shared__ int shdata[1<<NTHREAD2];

      int value = gidx < n && cell.isLeaf();
      shdata[threadIdx.x] = value;
#pragma unroll
      for (int offset2 = 0; offset2 < NTHREAD2; offset2++)
	{
	  const int offset = 1 << offset2;
	  __syncthreads();
	  if (threadIdx.x >= offset)
	    value += shdata[threadIdx.x - offset];
	  __syncthreads();
	  shdata[threadIdx.x] = value;
	}

      const int nwrite  = shdata[threadIdx.x];
      const int scatter = nwrite - (gidx < n && cell.isLeaf());

      __syncthreads();

      if (threadIdx.x == blockDim.x-1 && nwrite > 0)
	shdata[0] = atomicAdd(&leafIdx_counter, nwrite);

      __syncthreads();

      if (cell.isLeaf())
	leafCells[shdata[0] + scatter] = gidx;
    }
}

class Build {
 public:
  int2 tree(const int numBodies, float4 * d_bodyPos, float4 * d_bodyPos2,
	    float4 * d_domain, int2 * d_levelRange, CellData * d_sourceCells, const int NLEAF) {
    const int NTHREAD2 = 8;
    const int NTHREAD  = 1 << NTHREAD2;

    cuda_mem<float3> d_bounds;
    cuda_mem<int> d_octantSizePool;
    cuda_mem<int> d_octantSizeScanPool;
    cuda_mem<int> d_subOctantSizeScanPool;
    cuda_mem<int> d_blockCounterPool;
    cuda_mem<int2> d_bodyRangePool;
    cuda_mem<CellData> d_sourceCells2;
    cuda_mem<int> d_leafCells;
    cuda_mem<int> d_key, d_value;

    d_bounds.alloc(2048);
    const int maxNode = numBodies / 10;
    fprintf(stdout,"Stack size           : %g MB\n",sizeof(int)*83*maxNode/1024.0/1024.0);
    d_octantSizePool.alloc(8*maxNode);
    d_octantSizeScanPool.alloc(8*maxNode);
    d_subOctantSizeScanPool.alloc(64*maxNode);
    d_blockCounterPool.alloc(maxNode);
    d_bodyRangePool.alloc(maxNode);
    d_sourceCells2.alloc(numBodies);

    fprintf(stdout,"Cell data            : %g MB\n",numBodies*sizeof(CellData)/1024.0/1024.0);
    d_key.alloc(numBodies);
    d_value.alloc(numBodies);

    cudaDeviceSynchronize();
    double t0 = get_time();
    getBounds<NTHREAD2><<<NTHREAD,NTHREAD,NTHREAD*sizeof(float2)>>>
      (numBodies, d_bounds, d_domain, d_bodyPos);
    kernelSuccess("cudaDomainSize");
    double dt = get_time() - t0;
    fprintf(stdout,"Get bounds           : %.7f s\n",  dt);

    /*** build tree ***/

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(maxNodeGlob, &maxNode, sizeof(int), 0, cudaMemcpyHostToDevice));

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,16384);

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<16,true>,  cudaFuncCachePreferShared));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<16,false>, cudaFuncCachePreferShared));

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<24,true>,  cudaFuncCachePreferShared));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<24,false>, cudaFuncCachePreferShared));

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<32,true>,  cudaFuncCachePreferShared));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<32,false>, cudaFuncCachePreferShared));

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<48,true>,  cudaFuncCachePreferShared));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<48,false>, cudaFuncCachePreferShared));

    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<64,true>,  cudaFuncCachePreferShared));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<64,false>, cudaFuncCachePreferShared));

    CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

    CUDA_SAFE_CALL(cudaMemset(d_octantSizePool,0,8*maxNode*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_octantSizeScanPool,0,8*maxNode*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_subOctantSizeScanPool,0,64*maxNode*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_blockCounterPool,0,maxNode*sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_bodyRangePool,0,maxNode*sizeof(int2)));
    cudaDeviceSynchronize();
    t0 = get_time();
    switch (NLEAF) {
    case 16:
      buildOctree<16><<<1,1>>>(numBodies, d_domain, d_sourceCells, d_octantSizePool, d_octantSizeScanPool, d_subOctantSizeScanPool,
			       d_blockCounterPool, d_bodyRangePool, d_bodyPos, d_bodyPos2);
      break;
    case 24:
      buildOctree<24><<<1,1>>>(numBodies, d_domain, d_sourceCells, d_octantSizePool, d_octantSizeScanPool, d_subOctantSizeScanPool,
			       d_blockCounterPool, d_bodyRangePool, d_bodyPos, d_bodyPos2);
      break;
    case 32:
      buildOctree<32><<<1,1>>>(numBodies, d_domain, d_sourceCells, d_octantSizePool, d_octantSizeScanPool, d_subOctantSizeScanPool,
			       d_blockCounterPool, d_bodyRangePool, d_bodyPos, d_bodyPos2);
      break;
    case 48:
      buildOctree<48><<<1,1>>>(numBodies, d_domain, d_sourceCells, d_octantSizePool, d_octantSizeScanPool, d_subOctantSizeScanPool,
			       d_blockCounterPool, d_bodyRangePool, d_bodyPos, d_bodyPos2);
      break;
    case 64:
      buildOctree<64><<<1,1>>>(numBodies, d_domain, d_sourceCells, d_octantSizePool, d_octantSizeScanPool, d_subOctantSizeScanPool,
			       d_blockCounterPool, d_bodyRangePool, d_bodyPos, d_bodyPos2);
      break;
    default:
      assert(0);
    }
    kernelSuccess("buildOctree");
    dt = get_time() - t0;
    int numLevels, numSources, numLeafs;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLevels, numLevelsGlob,sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numSources,numCellsGlob, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLeafs, numLeafsGlob,sizeof(int)));
    fprintf(stdout,"Grow tree            : %.7f s\n",  dt);

    /* sort nodes by level */
    cudaDeviceSynchronize();
    t0 = get_time();
    const int NBLOCK = (numSources-1) / NTHREAD + 1;
    get_cell_levels<<<NBLOCK,NTHREAD>>>(numSources, d_sourceCells, d_sourceCells2, d_key, d_value);
    sort(numSources, d_key.ptr, d_value.ptr);

    /* compute begining & end of each level */
    getLevelRange<<<NBLOCK,NTHREAD,(NTHREAD+2)*sizeof(int)>>>(numSources, d_key, d_levelRange);
    write_newIdx <<<NBLOCK,NTHREAD>>>(numSources, d_value, d_key);
    shuffle_cells<<<NBLOCK,NTHREAD>>>(numSources, d_value, d_key, d_sourceCells2, d_sourceCells);

    /* group leaves */
    d_leafCells.alloc(numLeafs);
    collect_leaves<NTHREAD2><<<NBLOCK,NTHREAD>>>(numSources, d_sourceCells, d_leafCells);
    kernelSuccess("shuffle");
    dt = get_time() - t0;
    fprintf(stdout,"Link tree            : %.7f s\n", dt);
    int2 numLS = {numLevels, numSources};
    return numLS;
  }
};
