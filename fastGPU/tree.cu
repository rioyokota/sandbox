#include "octree.h"

static __device__ void pairMinMax(vec3 &xmin, vec3 &xmax,
                                  vec3 Xmin, vec3 Xmax) {
  xmin = fminf(xmin, Xmin);
  xmax = fmaxf(xmax, Xmax);
}

static __device__ uint4 getKey(int4 index3) {
  const int bits = 30;
  uint4 key4 = {0, 0, 0, 0};
  int mask = 1 << (bits - 1);
  int key = 0;
  for( int i=0; i<bits; i++, mask >>= 1) {
    int xi = (index3.x & mask) ? 1 : 0;
    int yi = (index3.y & mask) ? 1 : 0;
    int zi = (index3.z & mask) ? 1 : 0;        
    key = (key << 3) + (xi << 2) + (yi << 1) + zi;
    if(i == 19) {
      key4.y = key;
      key = 0;
    }
    if(i == 9) {
      key4.x = key;
      key = 0;
    }
  }
  key4.z = key;
  return key4;
}

static __device__ uint3 getMask(int level) {
  int mask_levels = 3 * (MAXLEVELS - level);
  uint3 mask = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
  if (mask_levels > 60) {
    mask.z = 0;
    mask.y = 0;
    mask.x = (mask.x >> (mask_levels - 60)) << (mask_levels - 60);
  } else if (mask_levels > 30) {
    mask.z = 0;
    mask.y = (mask.y >> (mask_levels - 30)) << (mask_levels - 30);
  } else {
    mask.z = (mask.z >> mask_levels) << mask_levels;
  }
  return mask;
}

static __device__ int compareKey(uint3 a, uint3 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    else {
      if       (a.z < b.z) return -1;
      else  if (a.z > b.z) return +1;
      return 0;
    }
  }
}

//Binary search of the key within certain bounds (cij.x, cij.y)
static __device__ int findKey(uint3 key, uint2 cij, uint3 *keys) {
  int l = cij.x;
  int r = cij.y - 1;
  while (r - l > 1) {
    int m = (r + l) >> 1;
    int cmp = compareKey(keys[m], key);
    if (cmp == -1)
      l = m;
    else 
      r = m;
  }
  if (compareKey(keys[l], key) >= 0) return l;
  return r;
}

extern "C" __global__ void getKeyKernel(int numBodies,
                                        vec3 *Body_X,
                                        uint4 *Body_ICELL) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBodies) return;
  vec3 pos = Body_X[idx];
  int4 index3;
  float size = 1 << MAXLEVELS;
  index3.x = int(pos[0] * size);
  index3.y = int(pos[1] * size);
  index3.z = int(pos[2] * size);
  uint4 key = getKey(index3);
  key.w = idx;
  Body_ICELL[idx] = key;
}

extern "C" __global__ void getValidRange(int numBodies,
                                         int level,
                                         uint4 *Body_ICELL,
                                         uint *validRange,
                                         const uint *workToDo) {
  if (*workToDo == 0) return;
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBodies) return;
  const uint3 key_F = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
  uint3 mask = getMask(level);
  uint3 key_c = make_uint3(Body_ICELL[idx].x,Body_ICELL[idx].y,Body_ICELL[idx].z);
  uint3 key_m;
  if( idx == 0 )
    key_m = key_F;
  else
    key_m = make_uint3(Body_ICELL[idx-1].x,Body_ICELL[idx-1].y,Body_ICELL[idx-1].z);

  uint3 key_p;
  if( idx == numBodies-1 )
    key_p = key_F;
  else
    key_p = make_uint3(Body_ICELL[idx+1].x,Body_ICELL[idx+1].y,Body_ICELL[idx+1].z);

  int valid0 = 0;
  int valid1 = 0;
  if (compareKey(key_c, key_F) != 0) {
    key_c.x = key_c.x & mask.x;
    key_c.y = key_c.y & mask.y;
    key_c.z = key_c.z & mask.z;
    key_p.x = key_p.x & mask.x;
    key_p.y = key_p.y & mask.y;
    key_p.z = key_p.z & mask.z;
    key_m.x = key_m.x & mask.x;
    key_m.y = key_m.y & mask.y;
    key_m.z = key_m.z & mask.z;
    valid0 = abs(compareKey(key_c, key_m));
    valid1 = abs(compareKey(key_c, key_p));
  }
  validRange[idx*2]   = idx | ((valid0) << 31);
  validRange[idx*2+1] = idx | ((valid1) << 31);
}

extern "C" __global__ void buildNodes(uint level,
                                      uint *workToDo,
                                      uint *maxLevel,
                                      uint2 *levelRange,
                                      uint *bodyRange,
                                      uint4 *Body_ICELL,
                                      uint3 *Cell_ICELL,
                                      uint *Cell_LEVEL,
                                      uint *Cell_LEAF,
                                      uint *Cell_NLEAF) {
  if( *workToDo == 0 ) return;
  uint idx  = blockIdx.x * blockDim.x + threadIdx.x;
  const uint stride = gridDim.x * blockDim.x;
  uint n = (*workToDo) / 2;
  uint offset;
  if( level == 0 )
    offset = 0;
  else
    offset = levelRange[level-1].y;

  while( idx < n ) {
    uint begin = bodyRange[idx*2];
    uint end = bodyRange[idx*2+1]+1;
    uint4 key  = Body_ICELL[begin];
    uint3 mask = getMask(level);
    Cell_ICELL[offset+idx] = make_uint3(key.x & mask.x, key.y & mask.y, key.z & mask.z);
    Cell_LEVEL[offset+idx] = level;
    Cell_LEAF[offset+idx] = begin;
    Cell_NLEAF[offset+idx] = end - begin;
    if( end - begin <= NCRIT )
      for( int i=begin; i<end; i++ )
        Body_ICELL[i] = make_uint4(0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF);
    idx += stride;
  }

  if( threadIdx.x == 0 && blockIdx.x == 0 ) {
    levelRange[level] = make_uint2(offset, offset + n);
    *maxLevel = level;
  }
}

extern "C" __global__ void linkNodes(int numNodes,
                                     uint3 *Cell_ICELL,
                                     uint *Cell_LEVEL,
                                     uint *Cell_NLEAF,
                                     uint *Cell_BEGIN,
                                     uint *Cell_SIZE,
                                     uint2 *levelRange,
                                     uint* validRange) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numNodes) return;
  uint3 key = Cell_ICELL[idx];
  uint level = Cell_LEVEL[idx];
  uint size = Cell_NLEAF[idx];

  uint3 mask = getMask(level-1);
  key = make_uint3(key.x & mask.x, key.y & mask.y,  key.z & mask.z); 
  if(idx > 0) {
    int ci = findKey(key,levelRange[level-1],Cell_ICELL);
    atomicAdd(&Cell_SIZE[ci], 1);
  }

  key = Cell_ICELL[idx];
  mask = getMask(level);
  key = make_uint3(key.x & mask.x, key.y & mask.y, key.z & mask.z); 
  int cj = findKey(key,levelRange[level+1],Cell_ICELL);
  atomicOr(&Cell_BEGIN[idx], cj);

  uint valid = idx; 
  if( size <= NCRIT )    
    valid = idx | (uint)(1 << 31);
  validRange[idx] = valid;
}

extern "C" __global__ void getLevelRange(const int numNodes,
                                         const int numLeafs,
                                         uint *leafNodes,
                                         uint *Cell_LEVEL,
                                         uint* validRange) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x + numLeafs;
  if (idx >= numNodes) return;
  const int node = leafNodes[idx];
  int level_c, level_m, level_p;
  level_c = Cell_LEVEL[node];
  if( idx+1 < numNodes )
    level_p = Cell_LEVEL[leafNodes[idx+1]];
  else
    level_p = MAXLEVELS+5;
  if(node == 0)
    level_m = -1;    
  else
    level_m = Cell_LEVEL[leafNodes[idx-1]];
  validRange[(idx-numLeafs)*2]   = idx | (level_c != level_m) << 31;
  validRange[(idx-numLeafs)*2+1] = idx | (level_c != level_p) << 31;
}

extern "C" __global__ void setNodeRange(int numBodies,
                                        uint *nodeRange,
                                        int treeDepth) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBodies) return;
  __shared__ int shmem[128];
  if(blockIdx.x == 0) {
    if(threadIdx.x < (MAXLEVELS*2))
      shmem[threadIdx.x] = nodeRange[threadIdx.x];
    __syncthreads();
    if(threadIdx.x < MAXLEVELS) {
      nodeRange[threadIdx.x]  = shmem[threadIdx.x*2];
      if(threadIdx.x == treeDepth-1)
        nodeRange[threadIdx.x] = shmem[threadIdx.x*2-1]+1;
    }
  }
}

extern "C" __global__ void permuteBodies(const int numBodies, uint4 *index, vec3 *Body_X, float *Body_SRC, vec4* output) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBodies) return;
  int newIndex = index[idx].w;
  output[idx] = make_vec4(Body_X[newIndex][0],Body_X[newIndex][1],Body_X[newIndex][2],Body_SRC[newIndex]);
}

extern "C" __global__ void copyBodies(const int numBodies, uint4 *index, vec3 *Body_X, float *Body_SRC, vec4* input) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numBodies) return;
  Body_X[idx] = make_vec3(input[idx][0],input[idx][1],input[idx][2]);
  Body_SRC[idx] = input[idx][3];
}

extern "C" __global__ void P2M(const int numLeafs,
                               uint *leafNodes,
                               uint *Cell_LEAF,
                               uint *Cell_NLEAF,
                               vec3 *Cell_X,
                               vec3 *Body_X,
                               float *Body_SRC,
                               vecM *Multipole,
                               vec4 *nodeLowerBounds,
                               vec4 *nodeUpperBounds) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numLeafs) return;
  int node = leafNodes[idx];
  const uint begin = Cell_LEAF[node];
  const uint size = Cell_NLEAF[node];
  vecM Mi = 0.0f;
  vec3 Xi = 0.0f;
  vec3 xmin =  1e10f;
  vec3 xmax = -1e10f;
  for( int i=begin; i<begin+size; i++ ) {
    float Mj = Body_SRC[i];
    vec3 Xj = Body_X[i];
    Mi[0] += Mj;
    Xi[0] += Mj * Xj[0];
    Xi[1] += Mj * Xj[1];
    Xi[2] += Mj * Xj[2];
    pairMinMax(xmin, xmax, Xj, Xj);
  }
  float invM = 1.0 / Mi[0];
  if(Mi[0] == 0) invM = 0;
  Xi[0] *= invM;
  Xi[1] *= invM;
  Xi[2] *= invM;
  Multipole[node] = Mi;
  Cell_X[node] = Xi;
  nodeLowerBounds[node] = make_vec4(xmin[0], xmin[1], xmin[2], 0.0f);
  nodeUpperBounds[node] = make_vec4(xmax[0], xmax[1], xmax[2], 1.0f);
  return;
}

extern "C" __global__ void M2M(const int level,
                               uint *leafNodes,
                               uint *nodeRange,
                               uint *Cell_BEGIN,
                               uint *Cell_SIZE,
                               vec3 *Cell_X,
                               vecM *Multipole,
                               vec4 *nodeLowerBounds,
                               vec4 *nodeUpperBounds) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x + nodeRange[level-1];
  if(idx >= nodeRange[level]) return;
  const int node = leafNodes[idx];
  const uint begin = Cell_BEGIN[node];
  const uint end = begin + Cell_SIZE[node];
  vecM Mi = 0.0f;
  vec3 Xi = 0.0f;
  vec3 xmin =  1e10f;
  vec3 xmax = -1e10f;
  for( int i=begin; i<end; i++ ) {
    vecM Mj = Multipole[i];
    vec3 Xj = Cell_X[i];
    Mi[0] += Mj[0];
    Xi[0] += Mj[0] * Xj[0];
    Xi[1] += Mj[0] * Xj[1];
    Xi[2] += Mj[0] * Xj[2];
    pairMinMax(xmin, xmax, make_vec3(nodeLowerBounds[i]), make_vec3(nodeUpperBounds[i]));
  }
  float invM = 1.0 / Mi[0];
  if(Mi[0] == 0) invM = 0;
  Xi[0] *= invM;
  Xi[1] *= invM;
  Xi[2] *= invM;
  Multipole[node] = Mi;
  Cell_X[node] = Xi;
  nodeLowerBounds[node] = make_vec4(xmin[0], xmin[1], xmin[2], 0.0f);
  nodeUpperBounds[node] = make_vec4(xmax[0], xmax[1], xmax[2], 0.0f);
  return;
}

extern "C" __global__ void rescale(const int numNodes,
                                   vecM *Multipole,
                                   vec4 *nodeLowerBounds,
                                   vec4 *nodeUpperBounds,
                                   uint  *Cell_BEGIN,
                                   uint  *Cell_SIZE,
                                   float *Cell_RCRIT,
                                   vec3 *Cell_X,
                                   uint *Cell_LEAF,
                                   uint *Cell_NLEAF){
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= numNodes) return;
  vecM M = Multipole[idx];
  vec3 X = Cell_X[idx];
  vec4 xmin = nodeLowerBounds[idx];
  vec4 xmax = nodeUpperBounds[idx];
  vec3 boxCenter = make_vec3(xmin + xmax) * 0.5;
  vec3 boxSize = fmaxf(fabsf(boxCenter-make_vec3(xmin)), fabsf(boxCenter-make_vec3(xmax)));
  vec3 dist = boxCenter - X;
  float R = norm(dist);
  if(fabsf(M[0]) < 1e-10) R = 0;

  float length = 2 * fmaxf(boxSize);
  if(length < 1e-6) length = 1e-6;
  float RCRIT = length / THETA + R;
  RCRIT = RCRIT * RCRIT;
  uint begin = Cell_LEAF[idx];
  uint size = Cell_NLEAF[idx];
  bool leaf = (size <= NCRIT);

  if( leaf ) {
    RCRIT = -RCRIT;
    Cell_BEGIN[idx] = begin;
    Cell_SIZE[idx] = size;
  }
  Cell_RCRIT[idx] = RCRIT;
  return;
}

void octree::getKeys() {
  int threads = 128;
  int blocks = ALIGN(numBodies,threads);
  getKeyKernel<<<blocks,threads,0,execStream>>>(numBodies,Body_X.devc(),uint4buffer);
}

void octree::sortKeys() {
  sorter->sort(uint4buffer,Body_ICELL,numBodies);
}

void octree::sortBodies() {
  int threads = 512;
  int blocks = ALIGN(numBodies,threads);
  permuteBodies<<<blocks,threads,0,execStream>>>(numBodies,Body_ICELL.devc(),Body_X.devc(),Body_SRC.devc(),vec4buffer);
  copyBodies<<<blocks,threads,0,execStream>>>(numBodies,Body_ICELL.devc(),Body_X.devc(),Body_SRC.devc(),vec4buffer);
}

void octree::buildTree() {
  cudaVec<uint> maxLevel;
  maxLevel.alloc(1);
  validRange.zeros();
  levelRange.zeros();
  workToDo.ones();
  int threads = 128;
  int blocks = ALIGN(numBodies,threads);
  for( int level=0; level<MAXLEVELS; level++ ) {
    getValidRange<<<blocks,threads,0,execStream>>>(numBodies,level,Body_ICELL.devc(),validRange.devc(),workToDo.devc());
    gpuCompact(validRange,compactRange,2*numBodies);
    buildNodes<<<64,threads,0,execStream>>>(level,workToDo.devc(),maxLevel.devc(),levelRange.devc(),compactRange.devc(),Body_ICELL.devc(),Cell_ICELL.devc(),Cell_LEVEL.devc(),Cell_LEAF.devc(),Cell_NLEAF.devc());
  }
  maxLevel.d2h();
  numLevels = maxLevel[0];
  levelRange.d2h();
  numNodes = levelRange[numLevels].y;
}

void octree::allocateTreePropMemory()
{
  Multipole.alloc(numNodes);
  Cell_BEGIN.alloc(numNodes);
  Cell_SIZE.alloc(numNodes);
  Cell_RCRIT.alloc(numNodes);
  Cell_X.alloc(numNodes);
}

void octree::linkTree() {
  // leafNodes
  Cell_BEGIN.zeros();
  Cell_SIZE.zeros();
  int threads = 128;
  int blocks = ALIGN(numNodes,threads);
  linkNodes<<<blocks,threads,0,execStream>>>(numNodes,Cell_ICELL.devc(),Cell_LEVEL.devc(),Cell_NLEAF.devc(),Cell_BEGIN.devc(),Cell_SIZE.devc(),levelRange.devc(),validRange.devc());
  leafNodes.alloc(numNodes);
  workToDo.ones();
  gpuSplit(validRange, leafNodes, numNodes);
  workToDo.d2h();
  numLeafs = workToDo[0];
  // nodeRange
  validRange.zeros();
  blocks = ALIGN(numNodes-numLeafs,threads);
  getLevelRange<<<blocks,threads,0,execStream>>>(numNodes,numLeafs,leafNodes.devc(),Cell_LEVEL.devc(),validRange.devc());
  gpuCompact(validRange, nodeRange, 2*(numNodes-numLeafs));
  blocks = ALIGN(numBodies,threads);
  setNodeRange<<<blocks,threads,0,execStream>>>(numBodies,nodeRange.devc(),numLevels+1);
}

void octree::upward() {
  cudaVec<vec4> nodeLowerBounds;
  cudaVec<vec4> nodeUpperBounds;
  nodeLowerBounds.alloc(numNodes);
  nodeUpperBounds.alloc(numNodes);

  int threads = 128;
  int blocks = ALIGN(numLeafs,threads);
  P2M<<<blocks,threads,0,execStream>>>(numLeafs,leafNodes.devc(),Cell_LEAF.devc(),Cell_NLEAF.devc(),Cell_X.devc(),Body_X.devc(),Body_SRC.devc(),Multipole.devc(),nodeLowerBounds.devc(),nodeUpperBounds.devc());

  nodeRange.d2h();
  for( int level=numLevels; level>=1; level-- ) {
    int totalOnThisLevel = nodeRange[level] - nodeRange[level-1];
    blocks = ALIGN(totalOnThisLevel,threads);
    M2M<<<blocks,threads,0,execStream>>>(level,leafNodes.devc(),nodeRange.devc(),Cell_BEGIN.devc(),Cell_SIZE.devc(),Cell_X.devc(),Multipole.devc(),nodeLowerBounds.devc(),nodeUpperBounds.devc());
  }

  blocks = ALIGN(numNodes,threads);
  rescale<<<blocks,threads,0,execStream>>>(numNodes,Multipole.devc(),nodeLowerBounds.devc(),nodeUpperBounds.devc(),Cell_BEGIN.devc(),Cell_SIZE.devc(),Cell_RCRIT.devc(),Cell_X.devc(),Cell_LEAF.devc(),Cell_NLEAF.devc());
}
