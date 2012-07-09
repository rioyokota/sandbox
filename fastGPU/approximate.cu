#include "octree.h"
#define laneId (threadIdx.x & (WARP_SIZE - 1))
#define warpId (threadIdx.x >> WARP_SIZE2)
#define IF(x) (-(int)(x))
#define ABS(a) ((int(a) < 0) ? -(a) : (a))

__device__ __forceinline__ int inclusiveScanInt(int* prefix, int value)
{
  prefix[laneId] = value;
  for( int i=0; i<WARP_SIZE2; i++ ) {
    const int offset = 1 << i;
    const int laneOffset = ABS(laneId-offset);
    prefix[laneId] += prefix[laneOffset] & IF(laneId >= offset);
  }
  return prefix[WARP_SIZE-1];
}

__device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}

__device__ int exclusiveScanBit(const bool flag)
{
  const uint flags = __ballot(flag);
  return __popc(flags & lanemask_lt());
}

__device__ int reduceBit(const bool flag)
{
  const uint flags = __ballot(flag);
  return __popc(flags);
}

__device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

__device__ __forceinline__ int inclusive_segscan_warp(
    int *shmem, const int packed_value, int &dist_block, int &nseg)
{
  const int  flag = packed_value < 0;
  const int  mask = IF(flag);
  const int value = (mask & (-1-packed_value)) + (~mask & 1);
  const int flags = __ballot(flag);

  nseg += __popc(flags) ;
  dist_block = __clz(__brev(flags));

  const int distance = min(__clz(flags & lanemask_le()) + laneId - 31, laneId);
  shmem[laneId] = value;
  for( int i=0; i<WARP_SIZE2; i++ ) {
    const int offset = 1 << i;
    const int laneOffset = ABS(laneId-offset);
    shmem[laneId] += shmem[laneOffset] & IF(offset <= distance);
  }
  return shmem[WARP_SIZE - 1];
}

__device__ __forceinline__ int inclusive_segscan_array(int *shmem_in, const int N)
{
  int dist, nseg = 0;
  int y = inclusive_segscan_warp(shmem_in, shmem_in[laneId], dist, nseg);
  for( int p=WARP_SIZE; p<N; p+=WARP_SIZE ) {
    int *shmem = shmem_in + p;
    int y1 = inclusive_segscan_warp(shmem, shmem[laneId], dist, nseg);
    shmem[laneId] += y & IF(laneId < dist);
    y = y1;
  }
  return nseg;
}

__device__ __forceinline__ int ACCESS(const int i) {
  return (i & (LMEM_STACK_SIZE - 1)) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ void P2P(vec4 &acc, 
                                    const vec3 targetX,
                                    const vec3 sourceX,
                                    const float sourceM) {
  vec3 dist = sourceX - targetX;
  const float R2 = norm(dist) + EPS2;
  float invR = rsqrtf(R2);
  const float invR2 = invR * invR;
  invR *= sourceM;
  dist *= invR * invR2;
  acc[3] -= invR;
  acc[0] += dist[0];
  acc[1] += dist[1];
  acc[2] += dist[2];
}

__device__ __forceinline__ void M2P(vec4 &acc,
                                    const vec3 targetX,
                                    const vec3 sourceX,
                                    const vecM M) {
  vec3 dist = sourceX - targetX;
  const float R2 = norm(dist) + EPS2;
  float invR = rsqrtf(R2);
  const float invR2 = invR * invR;
  invR *= M[0];
  dist *= invR * invR2;
  acc[3] -= invR;
  acc[0] += dist[0];
  acc[1] += dist[1];
  acc[2] += dist[2];
}

__device__ bool applyMAC(const vec3 sourceX,
                         const vec3 targetX,
                         const float Cell_RCRIT) {
  vec3 dist = targetX - sourceX;
  const float R2 = norm(dist);
  return R2 <= fabsf(Cell_RCRIT);
}

__device__ void traverse(vec3 *Body_X,
                         float *Body_SRC,
                         vec3 &X,
                         vec4 &TRG,
                         uint *Cell_BEGIN,
                         uint *Cell_SIZE,
                         float *Cell_RCRIT,
                         vec3 *Cell_X,
                         vecM *Multipole,
                         vec3 targetX,
                         int *blockPtr) {
  const int stackSize = LMEM_STACK_SIZE * NTHREAD;
  int *approxNodes = blockPtr + stackSize + 2 * WARP_SIZE * warpId;
  const uint numShrd = 10 + MTERM;
  __shared__ int shmem[numShrd * NTHREAD];
  int *numDirect = shmem + numShrd * WARP_SIZE * warpId;
  int *prefix = &numDirect[WARP_SIZE];
  int *stackShrd = &prefix[WARP_SIZE];
  int *directNodes = &stackShrd[WARP_SIZE];
  vec3 *sourceX = (vec3*)&directNodes[3*WARP_SIZE];
  vecM *sourceM = (vecM*)&sourceX[WARP_SIZE];

  // stack
  int *stackGlob = blockPtr;
  // begin tree-walk
  int warpOffsetApprox = 0;
  int warpOffsetDirect = 0;
  int numNodes = 1;
  int beginStack = 0;
  int endStack = 1;
  stackGlob[threadIdx.x] = laneId;
  // walk each level
  while( numNodes > 0 ) {
    int numNodesNew = 0;
    int warpOffsetSplit = 0;
    int numStack = endStack;
    // walk a level
    for( int iStack=beginStack; iStack<endStack; iStack++ ) {
      bool valid = laneId < numNodes;
      int node = stackGlob[ACCESS(iStack)] & IF(valid);
      numNodes -= WARP_SIZE;
      float RCRIT = Cell_RCRIT[node];
      bool split = applyMAC(Cell_X[node], targetX, RCRIT);
      bool leaf = RCRIT <= 0;
      bool flag = split && !leaf && valid;
      int begin = Cell_BEGIN[node];
      int size = Cell_SIZE[node];
      int numChild = size & IF(flag);
      int sumChild = inclusiveScanInt(prefix, numChild);
      int laneOffset = prefix[laneId];
      laneOffset += warpOffsetSplit - numChild;
      for( int i=0; i<numChild; i++ ) {
        stackShrd[laneOffset+i] = begin+i;
      }
      warpOffsetSplit += sumChild;
      while( warpOffsetSplit >= WARP_SIZE ) {
        warpOffsetSplit -= WARP_SIZE;
        stackGlob[ACCESS(numStack)] = stackShrd[warpOffsetSplit+laneId];
        numStack++;
        numNodesNew += WARP_SIZE;
        if( (numStack - iStack) > LMEM_STACK_SIZE ) {
          printf("overflow\n");
          return;
        }
      }
#if 1   // APPROX
      flag = !split && valid;
      laneOffset = exclusiveScanBit(flag);
      if( flag ) approxNodes[warpOffsetApprox+laneOffset] = node;
      warpOffsetApprox += reduceBit(flag);
      if( warpOffsetApprox >= WARP_SIZE ) {
        warpOffsetApprox -= WARP_SIZE;
        node = approxNodes[warpOffsetApprox+laneId];
        sourceX[laneId] = Cell_X[node];
        sourceM[laneId] = Multipole[node];
        for( int i=0; i<WARP_SIZE; i++ )
          M2P(TRG, X, sourceX[i], sourceM[i]);
      }
#endif
#if 1   // DIRECT
      flag = split && leaf && valid;
      int numBodies = size & IF(flag);
      directNodes[laneId] = numDirect[laneId];

      int sumBodies = inclusiveScanInt(prefix, numBodies);
      laneOffset = prefix[laneId];
      if( flag ) prefix[exclusiveScanBit(flag)] = laneId;
      numDirect[laneId] = laneOffset;
      laneOffset -= numBodies;
      int numFinished = 0;
      while( sumBodies > 0 ) {
        numBodies = min(sumBodies, 3*WARP_SIZE-warpOffsetDirect);
        for( int i=warpOffsetDirect; i<warpOffsetDirect+numBodies; i+=WARP_SIZE )
          directNodes[i+laneId] = 0;
        if( flag && (numDirect[laneId] <= numBodies) && (laneOffset >= 0) )
          directNodes[warpOffsetDirect+laneOffset] = -1-begin;
        numFinished += inclusive_segscan_array(&directNodes[warpOffsetDirect], numBodies);
        numBodies = numDirect[prefix[numFinished-1]];
        sumBodies -= numBodies;
        numDirect[laneId] -= numBodies;
        laneOffset -= numBodies;
        warpOffsetDirect += numBodies;
        while( warpOffsetDirect >= WARP_SIZE ) {
          warpOffsetDirect -= WARP_SIZE;
          const int j = directNodes[warpOffsetDirect+laneId];
          sourceX[laneId] = Body_X[j];
          sourceM[laneId][0] = Body_SRC[j];
          for( int i=0; i<WARP_SIZE; i++ )
            P2P(TRG, X, sourceX[i], sourceM[i][0]);
        }
      }
      numDirect[laneId] = directNodes[laneId];
#endif
    }

    if( warpOffsetSplit > 0 ) {
      stackGlob[ACCESS(numStack)] = stackShrd[laneId];
      numStack++;
      numNodesNew += warpOffsetSplit;
    }
    numNodes = numNodesNew;
    beginStack = endStack;
    endStack = numStack;
  }

  if( warpOffsetApprox > 0 ) {
    if( laneId < warpOffsetApprox )  {
      const int node = approxNodes[laneId];
      sourceX[laneId] = Cell_X[node];
      sourceM[laneId] = Multipole[node];
    } else {
      sourceX[laneId] = 1.0e10f;
      sourceM[laneId] = 0.0f;
    }
    for( int i=0; i<WARP_SIZE; i++ )
      M2P(TRG, X, sourceX[i], sourceM[i]);
  }

  if( warpOffsetDirect > 0 ) {
    if( laneId < warpOffsetDirect ) {
      const int j = numDirect[laneId];
      sourceX[laneId] = Body_X[j];
      sourceM[laneId][0] = Body_SRC[j];
    } else {
      sourceX[laneId] = 1.0e10f;
      sourceM[laneId][0] = 0.0f;
    }
    for( int i=0; i<WARP_SIZE; i++ )
      P2P(TRG, X, sourceX[i], sourceM[i][0]);
  }
}

extern "C" __global__ void traverseKernel(const int numLeafs,
                                          uint *leafNodes,
                                          uint *Cell_BEGIN,
                                          uint *Cell_SIZE,
                                          float *Cell_RCRIT,
                                          vec3 *Cell_X,
                                          vecM *Multipole,
                                          vec3 *Body_X,
                                          float *Body_SRC,
                                          vec4 *Body_TRG,
                                          int *buffer,
                                          uint *workToDo) {
  __shared__ int warpLeaf[4];
  int *leaf = warpLeaf + warpId;
  int *blockPtr = buffer + blockIdx.x * (LMEM_STACK_SIZE * NTHREAD + 2 * NTHREAD);
  while(true) {
    if( laneId == 0 )
      *leaf = atomicAdd(workToDo,1);
    if( *leaf >= numLeafs ) return;
    int node = leafNodes[*leaf];
    const uint begin = Cell_BEGIN[node];
    const uint size  = Cell_SIZE[node];
    vec3 targetX = Cell_X[node];
    bool valid = laneId < size;
    int body = (begin + laneId) & IF(valid);
    vec3 X = Body_X[body];
    vec4 TRG = 0.0f;
    traverse(Body_X, Body_SRC, X, TRG, Cell_BEGIN, Cell_SIZE, Cell_RCRIT, Cell_X, Multipole, targetX, blockPtr);
    if( valid ) Body_TRG[body] = TRG;
  }
}

extern "C" __global__ void directKernel(vec3 *Body_X, float *Body_SRC, vec4 *Body_TRG, const int N) {
  uint idx = min(blockIdx.x * blockDim.x + threadIdx.x, N-1);
  vec3 targetX = Body_X[idx];
  vec4 TRG = 0.0f;
  __shared__ vec3 shrdX[NTHREAD];
  __shared__ float shrdM[NTHREAD];
  vec3 *sourceX = shrdX + WARP_SIZE * warpId;
  float *sourceM = shrdM + WARP_SIZE * warpId;
  const int numWarp = ALIGN(N, WARP_SIZE);
  for( int jwarp=0; jwarp<numWarp; jwarp++ ) {
    int jGlob = jwarp*WARP_SIZE+laneId;
    int j = min(jGlob,N-1);
    sourceX[laneId] = Body_X[j];
    sourceM[laneId] = Body_SRC[j] * (jGlob < N);
    for( int i=0; i<WARP_SIZE; i++ )
      P2P(TRG, targetX, sourceX[i], sourceM[i]);
  }
  Body_TRG[idx] = TRG;
}

void octree::traverse() {
  workToDo.zeros();
  traverseKernel<<<NBLOCK,NTHREAD,0,execStream>>>(
    numLeafs,
    leafNodes.devc(),
    Cell_BEGIN.devc(),
    Cell_SIZE.devc(),
    Cell_RCRIT.devc(),
    Cell_X.devc(),
    Multipole.devc(),
    Body_X.devc(),
    Body_SRC.devc(),
    Body_TRG.devc(),
    buffer.devc(),
    workToDo.devc()
  );
}

void octree::iterate() {
  CU_SAFE_CALL(cudaStreamCreate(&execStream));
  double t1 = get_time();
  getKeys();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("INDEX : %lf\n",get_time() - t1);;
  t1 = get_time();
  sortKeys();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("SORT  : %lf\n",get_time() - t1);;
  t1 = get_time();
  sortBodies();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("PERM  : %lf\n",get_time() - t1);;
  t1 = get_time();
  buildTree();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("BUILD : %lf\n",get_time() - t1);;
  t1 = get_time();
  allocateTreePropMemory();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("ALLOC : %lf\n",get_time() - t1);;
  t1 = get_time();
  linkTree();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("LINK  : %lf\n",get_time() - t1);;
  t1 = get_time();
  upward();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("UPWARD: %lf\n",get_time() - t1);;
  t1 = get_time();
  traverse();
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  printf("TRAV  : %lf\n",get_time() - t1);;
}

void octree::direct() {
  int blocks = ALIGN(numBodies/100, NTHREAD);
  directKernel<<<blocks,NTHREAD,0,execStream>>>(Body_X.devc(),Body_SRC.devc(),Body2_TRG.devc(),numBodies);
  CU_SAFE_CALL(cudaStreamSynchronize(execStream));
  CU_SAFE_CALL(cudaStreamDestroy(execStream));
}
