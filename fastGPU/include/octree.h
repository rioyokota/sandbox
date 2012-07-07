#ifndef _OCTREE_H_
#define _OCTREE_H_

#include "types.h"

typedef unsigned int uint;

struct setupParams {
  int jobs;
  int blocksWithExtraJobs;
  int extraElements;
  int extraOffset;
};

namespace b40c {
  namespace util {
    template <typename T1, typename T2> class DoubleBuffer;
  }
  namespace radix_sort {
    class Enactor;
  }
}

class Sort90 {
private:
  b40c::util::DoubleBuffer<uint, uint> *double_buffer;
  b40c::radix_sort::Enactor *sort_enactor;

public:
  Sort90(uint size, uint *generalBuffer);
  ~Sort90();
  void sort(uint4 *input, cudaVec<uint4> &output, int size);
};

class octree {
private:
  cudaStream_t execStream;
  Sort90 *sorter;

  int numBodies;
  int numLeafs;
  int numNodes;
  int numLevels;
  union {
    uint4 *uint4buffer;
    vec4 *vec4buffer;
  };
  cudaVec<uint4>  Body_ICELL;
  cudaVec<uint>   Cell_BEGIN;
  cudaVec<uint>   Cell_SIZE;
  cudaVec<uint4>  nodeKeys;
  cudaVec<uint>   nodeChild;
  cudaVec<uint>   nodeRange;
  cudaVec<uint2>  levelRange;
  cudaVec<uint>   validRange;
  cudaVec<uint>   compactRange;

  cudaVec<uint>   leafNodes;
  cudaVec<uint2>  groupRange;
  cudaVec<vecM>   multipole;      

  cudaVec<float>  openingAngle;
  cudaVec<uint>   generalBuffer1;
  vec4 corner;

  cudaVec<vec3>   XMIN;
  cudaVec<vec3>   XMAX;
  cudaVec<uint>   offset;
  cudaVec<uint>   workToDo;
  
public:
  cudaVec<vec4>   bodyPos;
  cudaVec<vec4>   bodyAcc;
  cudaVec<vec4>   bodyAcc2;

private:
  bool isPowerOfTwo(const int n) {
    return ((n != 0) && !(n & (n - 1)));
  }
  void gpuCompact(cudaVec<uint> &input, cudaVec<uint> &output, int size);
  void gpuSplit(cudaVec<uint> &input, cudaVec<uint> &output, int size);
  void getBoundaries();
  void getKeys();
  void sortKeys();
  void sortBodies();
  void allocateTreePropMemory();
  void buildTree();
  void linkTree();
  void upward();
  void traverse();

public:
  octree(const int _n) : numBodies(_n) {
    assert(isPowerOfTwo(NCRIT));
    cudaSetDevice(2);
    bodyPos.alloc(numBodies+1);
    Body_ICELL.alloc(numBodies+1);
    bodyAcc.alloc(numBodies);
    bodyAcc2.alloc(numBodies);
    Cell_BEGIN.alloc(numBodies);
    Cell_SIZE.alloc(numBodies);
    nodeKeys.alloc(numBodies);
    nodeChild.alloc(numBodies);
    nodeRange.alloc(MAXLEVELS*2);
    levelRange.alloc(MAXLEVELS);
    validRange.alloc(2*numBodies);
    compactRange.alloc(2*numBodies);
    bodyAcc.zeros();
    CU_SAFE_CALL(cudaMalloc((uint4**)&uint4buffer, numBodies*sizeof(uint4)));

    int treeWalkStackSize = (LMEM_STACK_SIZE * NTHREAD + 2 * NTHREAD) * NBLOCK;
    int sortBufferSize = 4 * ALIGN(numBodies,128) * 128;
    generalBuffer1.alloc(max(treeWalkStackSize,sortBufferSize));
    sorter = new Sort90(numBodies, generalBuffer1.devc());

    XMIN.alloc(64);
    XMAX.alloc(64);
    offset.alloc(NBLOCK);
    workToDo.alloc(1);

  }
  ~octree() {
    delete sorter;
  }

  double get_time() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return double(tv.tv_sec +1.e-6*tv.tv_usec);
  }

  void iterate();
  void direct();
};

#endif
