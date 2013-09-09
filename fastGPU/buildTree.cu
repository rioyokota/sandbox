#include "Treecode.h"

#define NWARPS2 3
#define NWARPS  (1<<NWARPS2)

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "cuda_primitives.h"

namespace treeBuild
{
  static __device__ __forceinline__ int Octant(const float4 &lhs, const float4 &rhs) {
    return ((lhs.x <= rhs.x) << 0) + ((lhs.y <= rhs.y) << 1) + ((lhs.z <= rhs.z) << 2);
  };

  static __device__ __forceinline__ float4 ChildBox(const float4 &box, const int oct) {
    const float s = 0.5f * box.w;
    return make_float4(box.x + s * ((oct&1) ? 1.0f : -1.0f),
		       box.y + s * ((oct&2) ? 1.0f : -1.0f),
		       box.z + s * ((oct&4) ? 1.0f : -1.0f),
		       s);
  }

  static __device__ __forceinline__ void computeGridAndBlockSize(dim3 &grid, dim3 &block, const int np)
  {
    const int NTHREADS = (1<<NWARPS2) * WARP_SIZE;
    block = dim3(NTHREADS);
    assert(np > 0);
    grid = dim3(min(max(np/(NTHREADS*4),1), 512));
  }

  __device__ unsigned int retirementCount = 0;

  __constant__ int d_node_max;
  __constant__ int d_cell_max;

  __device__ unsigned int nnodes = 0;
  __device__ unsigned int nleaves = 0;
  __device__ unsigned int nlevels = 0;
  __device__ unsigned int nbodies_leaf = 0;
  __device__ unsigned int ncells = 0;

  __device__   int *memPool;
  __device__   CellData *sourceCells;
  __device__   void *ptclVel_tmp;

  template<int NTHREAD2>
  static __device__ float2 minmax_block(float2 sum)
  {
    extern __shared__ float shdata[];
    float *shMin = shdata;
    float *shMax = shdata + (1<<NTHREAD2);

    const int tid = threadIdx.x;
    shMin[tid] = sum.x;
    shMax[tid] = sum.y;
    __syncthreads();

#pragma unroll    
    for (int i = NTHREAD2-1; i >= 6; i--)
      {
        const int offset = 1 << i;
        if (tid < offset)
	  {
	    shMin[tid] = sum.x = fminf(sum.x, shMin[tid + offset]);
	    shMax[tid] = sum.y = fmaxf(sum.y, shMax[tid + offset]);
	  }
        __syncthreads();
      }

    if (tid < 32)
      {
        volatile float *vshMin = shMin;
        volatile float *vshMax = shMax;
#pragma unroll
        for (int i = 5; i >= 0; i--)
	  {
	    const int offset = 1 << i;
	    vshMin[tid] = sum.x = fminf(sum.x, vshMin[tid + offset]);
	    vshMax[tid] = sum.y = fmaxf(sum.y, vshMax[tid + offset]);
	  }
      }

    __syncthreads();

    return sum;
  }

  template<const int NTHREAD2>
  static __global__ void computeBoundingBox(
					    const int n,
					    float3 *minmax_ptr,
					    float4 *box_ptr,
					    const float4 *ptclPos)
  {
    const int NTHREAD = 1<<NTHREAD2;
    const int NBLOCK  = NTHREAD;

    float3 bmin = {+1e10f, +1e10f, +1e10f};
    float3 bmax = {-1e10f, -1e10f, -1e10f};

    const int nbeg = blockIdx.x * NTHREAD + threadIdx.x;
    for (int i = nbeg; i < n; i += NBLOCK*NTHREAD)
      if (i < n)
        {
          const float4 pos = ptclPos[i];
          bmin.x = fmin(bmin.x, pos.x);
          bmin.y = fmin(bmin.y, pos.y);
          bmin.z = fmin(bmin.z, pos.z);
          bmax.x = fmax(bmax.x, pos.x);
          bmax.y = fmax(bmax.y, pos.y);
          bmax.z = fmax(bmax.z, pos.z);
        }  
 
    float2 res;
    res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
    res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
    res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

    if (threadIdx.x == 0) 
      {
        minmax_ptr[blockIdx.x         ] = bmin;
        minmax_ptr[blockIdx.x + NBLOCK] = bmax;
      }

    __shared__ bool lastBlock;
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0)
      {
        const int ticket = atomicInc(&retirementCount, NBLOCK);
        lastBlock = (ticket == NBLOCK - 1);
      }

    __syncthreads();

    if (lastBlock)
      {

        bmin = minmax_ptr[threadIdx.x];
        bmax = minmax_ptr[threadIdx.x + NBLOCK];

        float2 res;
        res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

        __syncthreads();

        if (threadIdx.x == 0)
	  {
#if 0
	    printf("bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
	    printf("bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
#endif
	    const float3 cvec = {(bmax.x+bmin.x)*0.5f, (bmax.y+bmin.y)*0.5f, (bmax.z+bmin.z)*0.5f};
	    const float3 hvec = {(bmax.x-bmin.x)*0.5f, (bmax.y-bmin.y)*0.5f, (bmax.z-bmin.z)*0.5f};
	    const float h = fmax(hvec.z, fmax(hvec.y, hvec.x));
	    float hsize = 1.0f;
	    while (hsize > h) hsize *= 0.5f;
	    while (hsize < h) hsize *= 2.0f;

	    const int NMAXLEVEL = 20;

	    const float hquant = hsize / float(1<<NMAXLEVEL);
	    const long long nx = (long long)(cvec.x/hquant);
	    const long long ny = (long long)(cvec.y/hquant);
	    const long long nz = (long long)(cvec.z/hquant);

	    const float4 box = {hquant * float(nx), hquant * float(ny), hquant * float(nz), hsize};

	    *box_ptr = box;
	    retirementCount = 0;
	  }
      }
  }

  /*******************/

  template<int NLEAF, bool STOREIDX>
  static __global__ void 
  __launch_bounds__( 256, 8)
  buildOctant(
	      float4 box,
	      const int cellParentIndex,
	      const int cellIndexBase,
	      const int octantMask,
	      int *octCounterBase,
	      float4 *ptcl,
	      float4 *buff,
	      const int level = 0)
  {
    /* compute laneIdx & warpIdx for each of the threads:
     *   the thread block contains only 8 warps
     *   a warp is responsible for a single octant of the cell 
     */   
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;

    /* We launch a 2D grid:
     *   the y-corrdinate carries info about which parent cell to process
     *   the x-coordinate is just a standard approach for CUDA parallelism 
     */
    const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

    /* get the pointer to atomic data that for a given octant */
    int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

    /* read data about the current cell */
    const int data  = octCounter[laneIdx];
    const int nBeg  = __shfl(data, 1, WARP_SIZE);
    const int nEnd  = __shfl(data, 2, WARP_SIZE);
    /* if we are not at the root level, compute the geometric box
     * of the cell */
    if (!STOREIDX)
      box = ChildBox(box, octant2process);


    /* countes number of particles in each octant of a child octant */
    __shared__ int nShChildrenFine[NWARPS][9][8];
    __shared__ int nShChildren[8][8];

    float4 *shChildBox = (float4*)&nShChildren[0][0];

    int *shdata = (int*)&nShChildrenFine[0][0][0];
#pragma unroll 
    for (int i = 0; i < 8*9*NWARPS; i += NWARPS*WARP_SIZE)
      if (i + threadIdx.x < 8*9*NWARPS)
	shdata[i + threadIdx.x] = 0;

    if (laneIdx == 0 && warpIdx < 8)
      shChildBox[warpIdx] = ChildBox(box, warpIdx);

    __syncthreads();

    /* process particle array */
    const int nBeg_block = nBeg + blockIdx.x * blockDim.x;
    for (int i = nBeg_block; i < nEnd; i += gridDim.x * blockDim.x)
      {
        float4 p4 = ptcl[min(i+threadIdx.x, nEnd-1)];

        int p4octant = __float_as_int(p4.w) & 0xF;
        if (STOREIDX)
	  {
	    const int oct = __float_as_int(p4.w) & 0xF;
	    p4.w = __int_as_float(((i + threadIdx.x) << 4) | oct);
	    p4octant = Octant(box, p4);
	  }

        p4octant = i+threadIdx.x < nEnd ? p4octant : 0xF; 

        /* compute suboctant of the octant into which particle will fall */
        if (p4octant < 8)
	  {
	    const int p4subOctant = Octant(shChildBox[p4octant], p4);
	    const int idx = (__float_as_int(p4.w) >> 4) & 0xF0000000;
	    p4.w = __int_as_float((idx << 4) | p4subOctant);
	  }

        /* compute number of particles in each of the octants that will be processed by thead block */
        int np = 0;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
	  {
	    const int sum = warpBinReduce(p4octant == octant);
	    if (octant == laneIdx)
	      np = sum;
	  }

        /* increment atomic counters in a single instruction for thread-blocks to participated */
        int addrB0;
        if (laneIdx < 8)
          addrB0 = atomicAdd(&octCounter[8+8+laneIdx], np);

        /* compute addresses where to write data */
        int cntr = 32;
        int addrW = -1;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
	  {
	    const int sum = warpBinReduce(p4octant == octant);

	    if (sum > 0)
	      {
		const int offset = warpBinExclusiveScan1(p4octant == octant);
		const int addrB = __shfl(addrB0, octant, WARP_SIZE);
		if (p4octant == octant)
		  addrW = addrB + offset;
		cntr -= sum;
		if (cntr == 0) break;
	      }
	  }

        /* write the data in a single instruction */ 
        if (addrW >= 0)
          buff[addrW] = p4;

        /* count how many particles in suboctants in each of the octants */
        cntr = 32;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
	  {
	    if (cntr == 0) break;
	    const int sum = warpBinReduce(p4octant == octant);
	    if (sum > 0)
	      {
		const int subOctant = p4octant == octant ? (__float_as_int(p4.w) & 0xF) : -1;
#pragma unroll
		for (int k = 0; k < 8; k += 4)
		  {
		    const int4 sum = make_int4(
					       warpBinReduce(k+0 == subOctant),
					       warpBinReduce(k+1 == subOctant),
					       warpBinReduce(k+2 == subOctant),
					       warpBinReduce(k+3 == subOctant));
		    if (laneIdx == 0)
		      {
			int4 value = *(int4*)&nShChildrenFine[warpIdx][octant][k];
			value.x += sum.x;
			value.y += sum.y;
			value.z += sum.z;
			value.w += sum.w;
			*(int4*)&nShChildrenFine[warpIdx][octant][k] = value;
		      }
		  }
		cntr -= sum;
	      }
	  }
      }
    __syncthreads();

    if (warpIdx >= 8) return;


#pragma unroll
    for (int k = 0; k < 8; k += 4)
      {
        int4 nSubOctant = laneIdx < NWARPS ? (*(int4*)&nShChildrenFine[laneIdx][warpIdx][k]) : make_int4(0,0,0,0);
#pragma unroll
        for (int i = NWARPS2-1; i >= 0; i--)
	  {
	    nSubOctant.x += __shfl_xor(nSubOctant.x, 1<<i, NWARPS);
	    nSubOctant.y += __shfl_xor(nSubOctant.y, 1<<i, NWARPS);
	    nSubOctant.z += __shfl_xor(nSubOctant.z, 1<<i, NWARPS);
	    nSubOctant.w += __shfl_xor(nSubOctant.w, 1<<i, NWARPS);
	  }
        if (laneIdx == 0)
          *(int4*)&nShChildren[warpIdx][k] = nSubOctant;
      }

    __syncthreads();

    if (laneIdx < 8)
      if (nShChildren[warpIdx][laneIdx] > 0)
	atomicAdd(&octCounter[8+16+warpIdx*8 + laneIdx], nShChildren[warpIdx][laneIdx]);

    __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */


    /* detect last thread block for unique y-coordinate of the grid:
     * mind, this cannot be done on the host, because we don't detect last 
     * block on the grid, but instead the last x-block for each of the y-coordainte of the grid
     * this should increase the degree of parallelism
     */

    int *shmem = &nShChildren[0][0];
    if (warpIdx == 0)
      shmem[laneIdx] = 0;

    int &lastBlock = shmem[0];
    if (threadIdx.x == 0)
      {
        const int ticket = atomicAdd(octCounter, 1);
        lastBlock = (ticket == gridDim.x-1);
      }
    __syncthreads();

    if (!lastBlock) return;

    __syncthreads();

    /* okay, we are in the last thread block, do the analysis and decide what to do next */

    if (warpIdx == 0)
      shmem[laneIdx] = 0;

    if (threadIdx.x == 0)
      atomicCAS(&nlevels, level, level+1);

    __syncthreads();

    /* compute beginning and then end addresses of the sorted particles  in the child cell */

    const int nCell = __shfl(data, 8+warpIdx, WARP_SIZE);
    const int nEnd1 = octCounter[8+8+warpIdx];
    const int nBeg1 = nEnd1 - nCell;

    if (laneIdx == 0)
      shmem[warpIdx] = nCell;
    __syncthreads();

    const int npCell = laneIdx < 8 ? shmem[laneIdx] : 0;

    /* compute number of children that needs to be further split, and cmopute their offsets */
    const int2 nSubNodes = warpBinExclusiveScan(npCell > NLEAF);
    const int2 numLeaves   = warpBinExclusiveScan(npCell > 0 && npCell <= NLEAF);
    if (warpIdx == 0 && laneIdx < 8)
      {
        shmem[8 +laneIdx] = nSubNodes.x;
        shmem[16+laneIdx] = numLeaves.x;
      }

    int nCellmax = npCell;
#pragma unroll
    for (int i = 2; i >= 0; i--)
      nCellmax = max(nCellmax, __shfl_xor(nCellmax, 1<<i, WARP_SIZE));

    /* if there is at least one cell to split, increment nuumber of the nodes */
    if (threadIdx.x == 0 && nSubNodes.y > 0)
      {
        shmem[16+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 1   /* temp solution, a better one is to use RingBuffer */
        assert(shmem[16+8] < d_node_max);
#endif
      }

    /* writing linking info, parent, child and particle's list */
    const int nChildrenCell = warpBinReduce(npCell > 0);
    if (threadIdx.x == 0 && nChildrenCell > 0)
      {
        const int cellFirstChildIndex = atomicAdd(&ncells, nChildrenCell);
#if 1
        assert(cellFirstChildIndex + nChildrenCell < d_cell_max);
#endif
        /*** keep in mind, the 0-level will be overwritten ***/
        assert(nChildrenCell > 0);
        assert(nChildrenCell <= 8);
        const CellData cellData(level,cellParentIndex, nBeg, nEnd, cellFirstChildIndex, nChildrenCell-1);
        assert(cellData.first() < ncells);
        assert(cellData.isNode());
        sourceCells[cellIndexBase + blockIdx.y] = cellData;
        shmem[16+9] = cellFirstChildIndex;
      }

    __syncthreads();
    const int cellFirstChildIndex = shmem[16+9];
    /* compute atomic data offset for cell that need to be split */
    const int next_node = shmem[16+8];
    int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

    const int nodeOffset = shmem[8 +warpIdx];
    const int leafOffset = shmem[16+warpIdx];

    /* if cell needs to be split, populate it shared atomic data */
    if (nCell > NLEAF)
      {
        int *octCounterN = octCounterNbase + nodeOffset*(8+8+8+64+8);

        /* number of particles in each cell's subcells */
        const int nSubCell = laneIdx < 8 ? octCounter[8+16+warpIdx*8 + laneIdx] : 0;

        /* compute offsets */
        int cellOffset = nSubCell;
#pragma unroll
        for(int i = 0; i < 3; i++)  /* log2(8) steps */
          cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
        cellOffset -= nSubCell;

        /* store offset in memory */

        cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);
        if (laneIdx < 8) cellOffset = nSubCell;
        else            cellOffset += nBeg1;
        cellOffset = __shfl_up(cellOffset, 8, WARP_SIZE);

        if (laneIdx <  8) cellOffset = 0;
        if (laneIdx == 1) cellOffset = nBeg1;
        if (laneIdx == 2) cellOffset = nEnd1;

        if (laneIdx < 24)
          octCounterN[laneIdx] = cellOffset;
      }

    /***************************/
    /*  launch  child  kernel  */
    /***************************/

    /* warps coorperate so that only 1 kernel needs to be launched by a thread block
     * with larger degree of paralellism */
    if (nSubNodes.y > 0 && warpIdx == 0)
      {
        /* build octant mask */
        int octant_mask = npCell > NLEAF ?  (laneIdx << (3*nSubNodes.x)) : 0;
#pragma unroll
        for (int i = 4; i >= 0; i--)
          octant_mask |= __shfl_xor(octant_mask, 1<<i, WARP_SIZE);

        if (threadIdx.x == 0)
	  {
	    dim3 grid, block;
	    computeGridAndBlockSize(grid, block, nCellmax);
	    grid.y = nSubNodes.y;  /* each y-coordinate of the grid will be busy for each parent cell */
#if defined(FASTMODE) && NWARPS==8
	    if (nCellmax <= block.x)
	      {
		grid.x = 1;
		buildOctantSingle<NLEAF><<<grid,block>>>
		  (box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
		   octant_mask, octCounterNbase, buff, ptcl, level+1);
	      }
	    else
#endif
	      buildOctant<NLEAF,false><<<grid,block>>>
		(box, cellIndexBase+blockIdx.y, cellFirstChildIndex,
		 octant_mask, octCounterNbase, buff, ptcl, level+1);
	    const cudaError_t err = cudaGetLastError();
	    if (err != cudaSuccess)
	      {
		printf(" launch failed 1: %s  level= %d n =%d \n", cudaGetErrorString(err), level);
		assert(0);
	      }
	  }
      }

    /******************/
    /* process leaves */
    /******************/

    if (nCell <= NLEAF && nCell > 0)
      {
        if (laneIdx == 0)
	  {
	    atomicAdd(&nleaves,1);
	    atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
	    const CellData leafData(level+1, cellIndexBase+blockIdx.y, nBeg1, nEnd1);
	    assert(!leafData.isNode());
	    sourceCells[cellFirstChildIndex + nSubNodes.y + leafOffset] = leafData;
	  }
        if (!(level&1))
	  {
	    for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
	      if (i < nEnd1)
		{
		  float4 pos = buff[i];
		  int index = (__float_as_int(pos.w) >> 4) & 0xF0000000;
		  float4 vel = ((float4*)ptclVel_tmp)[index];
		  pos.w = vel.w;
		  ptcl[i] = pos;
		  buff[i] = vel;
		}
	  }
        else
	  {
	    for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
	      if (i < nEnd1)
		{
		  float4 pos = buff[i];
		  int index = (__float_as_int(pos.w) >> 4) & 0xF0000000;
		  float4 vel = ((float4*)ptclVel_tmp)[index];
		  pos.w = vel.w;
		  buff[i] = pos;
		  ptcl[i] = vel;
		}
	  }
      }
  }

  template<typename T>
  static __global__ void countAtRootNode(
					 const int n,
					 int *octCounter,
					 const float4 box,
					 const float4 *ptclPos)
  {
    int np_octant[8] = {0};
    const int beg = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = beg; i < n; i += gridDim.x * blockDim.x)
      if (i < n)
        {
          const float4 pos = ptclPos[i];
          const int octant = Octant(box, pos);
          np_octant[0] += (octant == 0);
          np_octant[1] += (octant == 1);
          np_octant[2] += (octant == 2);
          np_octant[3] += (octant == 3);
          np_octant[4] += (octant == 4);
          np_octant[5] += (octant == 5);
          np_octant[6] += (octant == 6);
          np_octant[7] += (octant == 7);
        };

    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
#pragma unroll
    for (int k = 0; k < 8; k++)
      {
        int np = np_octant[k];
#pragma unroll
        for (int i = 4; i >= 0; i--)
          np += __shfl_xor(np, 1<<i, WARP_SIZE);
        if (laneIdx == 0)
          atomicAdd(&octCounter[8+k],np);
      }
  }

  template<int NLEAF>
  static __global__ void buildOctree(
				     const int n,
				     const float4 *domain,
				     CellData *d_sourceCells,
				     int *stack_memory_pool,
				     float4 *ptcl,
				     float4 *buff,
				     float4 *d_ptclVel,
				     int *ncells_return = NULL)
  {
    sourceCells = d_sourceCells;
    ptclVel_tmp  = (void*)d_ptclVel;

    memPool = stack_memory_pool;

    int *octCounter = new int[8+8];
    for (int k = 0; k < 16; k++)
      octCounter[k] = 0;
    countAtRootNode<float><<<256, 256>>>(n, octCounter, *domain, ptcl);
    assert(cudaGetLastError() == cudaSuccess);
    cudaDeviceSynchronize();

    int *octCounterN = new int[8+8+8+64+8];
#pragma unroll
    for (int k = 0; k < 8; k++)
      {
        octCounterN[     k] = 0;
        octCounterN[8+   k] = octCounter[8+k  ];
        octCounterN[8+8 +k] = k == 0 ? 0 : octCounterN[8+8+k-1] + octCounterN[8+k-1];
        octCounterN[8+16+k] = 0;
      }
#pragma unroll
    for (int k = 8; k < 64; k++)
      octCounterN[8+16+k] = 0;

#ifdef IOCOUNT
    io_words = 0;
#endif
    nnodes = 0;
    nleaves = 0;
    nlevels = 0;
    ncells  = 0;
    nbodies_leaf = 0;


    octCounterN[1] = 0;
    octCounterN[2] = n;

    dim3 grid, block;
    computeGridAndBlockSize(grid, block, n);
#if 1
    buildOctant<NLEAF,true><<<grid, block>>>
      (*domain, 0, 0, 0, octCounterN, ptcl, buff);
    assert(cudaDeviceSynchronize() == cudaSuccess);
#endif

    if (ncells_return != NULL)
      *ncells_return = ncells;

#ifdef IOCOUNT
    printf(" io= %g MB \n" ,io_words*4.0/1024.0/1024.0);
#endif
    delete [] octCounter;
    delete [] octCounterN;
  }


  static __global__ void
  get_cell_levels(const int n, const CellData cellList[], CellData cellListOut[], int key[], int value[])
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const CellData cell = cellList[idx];
    key  [idx] = cell.level();
    value[idx] = idx;
    cellListOut[idx] = cell;
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
  shuffle_cells(const int n, const int value[], const int moved_to_idx[], const CellData cellListIn[], CellData cellListOut[])
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int mapIdx = value[idx];
    CellData cell = cellListIn[mapIdx];
    if (cell.isNode())
      {
        const int firstOld = cell.first();
        const int firstNew = moved_to_idx[firstOld];
        cell.update_first(firstNew);
      }
    if (cell.parent() > 0)
      cell.update_parent(moved_to_idx[cell.parent()]);

    cellListOut[idx] = cell;

    if (threadIdx.x == 0 && blockIdx.x == 0)
      leafIdx_counter = 0;
  }

  template<int NTHREAD2>
  static __global__ 
  void collect_leaves(const int n, const CellData *cellList, int *leafList)
  {
    const int gidx = blockDim.x*blockIdx.x + threadIdx.x;

    const CellData cell = cellList[min(gidx,n-1)];

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
      leafList[shdata[0] + scatter] = gidx;
  }
}


void Treecode::buildTree(const int nLeaf)
{
  this->nLeaf = nLeaf;
  assert(nLeaf == 16 || nLeaf == 24 || nLeaf == 32 || nLeaf == 48 || nLeaf == 64);
  /* compute bounding box */

  {
    const int NTHREAD2 = 8;
    const int NTHREAD  = 1<<NTHREAD2;
    const int NBLOCK   = NTHREAD;

    assert(2*NBLOCK <= 2048);  /* see Treecode constructor for d_minmax allocation */
    cudaDeviceSynchronize();
    const double t0 = get_time();
    treeBuild::computeBoundingBox<NTHREAD2><<<NBLOCK,NTHREAD,NTHREAD*sizeof(float2)>>>
      (nPtcl, d_minmax, d_domain, d_ptclPos);
    kernelSuccess("cudaDomainSize");
    const double dt = get_time() - t0;
    fprintf(stdout,"Get bounds           : %.7f s\n",  dt);
  }

  /*** build tree ***/

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_node_max, &node_max, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_cell_max, &cell_max, sizeof(int), 0, cudaMemcpyHostToDevice));

  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,16384);

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<16,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<16,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<24,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<24,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<32,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<32,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<48,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<48,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<64,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<64,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  {
    CUDA_SAFE_CALL(cudaMemset(d_stack_memory_pool,0,stack_size*sizeof(int)));
    cudaDeviceSynchronize();
    const double t0 = get_time();
    switch(nLeaf)
      {
      case 16:
        treeBuild::buildOctree<16><<<1,1>>>(
					    nPtcl, d_domain, d_sourceCells, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
        break;
      case 24:
        treeBuild::buildOctree<24><<<1,1>>>(
					    nPtcl, d_domain, d_sourceCells, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
        break;
      case 32:
        treeBuild::buildOctree<32><<<1,1>>>(
					    nPtcl, d_domain, d_sourceCells, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
        break;
      case 48:
        treeBuild::buildOctree<48><<<1,1>>>(
					    nPtcl, d_domain, d_sourceCells, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
        break;
      case 64:
        treeBuild::buildOctree<64><<<1,1>>>(
					    nPtcl, d_domain, d_sourceCells, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel);
        break;
      default:
        assert(0);
      }
    kernelSuccess("buildOctree");
    const double dt = get_time() - t0;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLevels, treeBuild::nlevels, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numSources,  treeBuild::ncells, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLeaves, treeBuild::nleaves, sizeof(int)));
    fprintf(stdout,"Grow tree            : %.7f s\n",  dt);
  }

  /* sort nodes by level */
  {
    cudaDeviceSynchronize();
    const double t0 = get_time();
    const int nthread = 256;
    const int nblock  = (numSources-1)/nthread  + 1;
    treeBuild::get_cell_levels<<<nblock,nthread>>>(numSources, d_sourceCells, d_sourceCells_tmp, d_key, d_value);

    thrust::device_ptr<int> keys_beg(d_key.ptr);
    thrust::device_ptr<int> keys_end(d_key.ptr + numSources);
    thrust::device_ptr<int> vals_beg(d_value.ptr);

    thrust::stable_sort_by_key(keys_beg, keys_end, vals_beg); 

    /* compute begining & end of each level */
    treeBuild::getLevelRange<<<nblock,nthread,(nthread+2)*sizeof(int)>>>(numSources, d_key, d_levelRange);

    treeBuild::write_newIdx <<<nblock,nthread>>>(numSources, d_value, d_key);
    treeBuild::shuffle_cells<<<nblock,nthread>>>(numSources, d_value, d_key, d_sourceCells_tmp, d_sourceCells);

    /* group leaves */

    d_leafList.realloc(numLeaves);
    const int NTHREAD2 = 8;
    const int NTHREAD  = 256;
    const int nblock1 = (numSources-1)/NTHREAD+1;
    treeBuild::collect_leaves<NTHREAD2><<<nblock1,NTHREAD>>>(numSources, d_sourceCells, d_leafList);

    kernelSuccess("shuffle");
    const double dt = get_time() - t0;
    fprintf(stdout,"Link tree            : %.7f s\n", dt);
  }
}
