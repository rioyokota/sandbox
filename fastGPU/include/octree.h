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
  cudaVec<uint3>  Cell_ICELL;
  cudaVec<uint>   Cell_LEVEL;
  cudaVec<uint>   Cell_CHILD;
  cudaVec<uint>   Cell_NCHILD;
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
  cudaVec<vec3>   Body_X;
  cudaVec<float>  Body_SRC;
  cudaVec<vec4>   Body_TRG;
  cudaVec<vec4>   Body2_TRG;

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
    Body_X.alloc(numBodies+1);
    Body_SRC.alloc(numBodies+1);
    Body_ICELL.alloc(numBodies+1);
    Body_TRG.alloc(numBodies);
    Body2_TRG.alloc(numBodies);
    Cell_BEGIN.alloc(numBodies);
    Cell_SIZE.alloc(numBodies);
    Cell_ICELL.alloc(numBodies);
    Cell_LEVEL.alloc(numBodies);
    Cell_CHILD.alloc(numBodies);
    Cell_NCHILD.alloc(numBodies);
    nodeRange.alloc(MAXLEVELS*2);
    levelRange.alloc(MAXLEVELS);
    validRange.alloc(2*numBodies);
    compactRange.alloc(2*numBodies);
    Body_TRG.zeros();
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
