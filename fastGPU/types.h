#pragma once

#define WARP_SIZE2 5
#define WARP_SIZE 32
#define NTHREAD2 8
#define NTHREAD 256
#define CUDA_SAFE_CALL(err) cudaSafeCall(err, __FILE__, __LINE__)

#include <assert.h>
#include <stdio.h>
#include "cudavec.h"
#include "kahan.h"
#include "logger.h"
#include "vec.h"

typedef vec<3,float> fvec3;
typedef vec<4,float> fvec4;
typedef vec<4,kahan<float> > kvec4;

__host__ __device__
static fvec3 make_fvec3(float x, float y, float z) {
  fvec3 data;
  data[0] = x;
  data[1] = y;
  data[2] = z;
  return data;
}

__host__ __device__
static fvec4 make_fvec4(float x, float y, float z, float w) {
  fvec4 data;
  data[0] = x;
  data[1] = y;
  data[2] = z;
  data[3] = w;
  return data;
}

__host__ __device__
static fvec3 make_fvec3(fvec4 v) {
  fvec3 data;
  data[0] = v[0];
  data[1] = v[1];
  data[2] = v[2];
  return data;
}

static void kernelSuccess(const char kernel[] = "kernel") {
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,"%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

inline void cudaSafeCall(cudaError err, const char *file, const int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
            file, line, cudaGetErrorString(err) );
    exit(EXIT_FAILURE);
  }
}

class CellData {
 private:
  static const int CHILD_SHIFT = 29;
  static const int CHILD_MASK  = ~(0x7U << CHILD_SHIFT);
  static const int LEVEL_SHIFT = 27;
  static const int LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT);
  uint4 data;
 public:
  __host__ __device__
    CellData(const unsigned int level,
	     const unsigned int parent,
	     const unsigned int body,
	     const unsigned int nbody,
	     const unsigned int child = 0,
	     const unsigned int nchild = 1) {
    const unsigned int parentPack = parent | (level << LEVEL_SHIFT);
    const unsigned int childPack = child | ((nchild-1) << CHILD_SHIFT);
    data = make_uint4(parentPack, childPack, body, nbody);
  }
  __host__ __device__
    CellData(const uint4 data) : data(data) {}
  __host__ __device__
    int level() const { return data.x >> LEVEL_SHIFT; }
  __host__ __device__
    int parent() const { return data.x & LEVEL_MASK; }
  __host__ __device__
    int child() const { return data.y & CHILD_MASK; }
  __host__ __device__
    int nchild() const { return (data.y >> CHILD_SHIFT)+1; }
  __host__ __device__
    int body() const { return data.z; }
  __host__ __device__
    int nbody() const { return data.w; }
  __host__ __device__
    bool isLeaf() const { return data.y == 0; }
  __host__ __device__
    bool isNode() const { return !isLeaf(); }
  __host__ __device__
    void setParent(const unsigned int parent) {
    data.x = parent | (level() << LEVEL_SHIFT);
  }
  __host__ __device__
    void setChild(const unsigned int child) {
    data.y = child | (nchild()-1 << CHILD_SHIFT);
  }
};
