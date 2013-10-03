#pragma once

#include <assert.h>
#include "cudamem.h"
#include "cudavec.h"
#include "plummer.h"
#include <string>
#include <sstream>
#include <sys/time.h>

#define WARP_SIZE2 5
#define WARP_SIZE 32
#define NTHREAD2 8
#define NTHREAD 256

struct float6 {
  float xx;
  float yy;
  float zz;
  float xy;
  float xz;
  float yz;
};

struct double6 {
  double xx;
  double yy;
  double zz;
  double xy;
  double xz;
  double yz;
};

static inline double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec + tv.tv_usec * 1e-6);
}

static void kernelSuccess(const char kernel[] = "kernel") {
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,"%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

inline void CUDA_SAFE_CALL(cudaError err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",
            __FILE__, __LINE__, cudaGetErrorString(err) );
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
  __host__ __device__ CellData(const unsigned int level,
			       const unsigned int parent,
			       const unsigned int body,
			       const unsigned int nbody,
			       const unsigned int child = 0,
			       const unsigned int nchild = 0)
  {
    const unsigned int parentPack = parent | (level << LEVEL_SHIFT);
    const unsigned int childPack = child | (nchild << CHILD_SHIFT);
    data = make_uint4(parentPack, childPack, body, nbody);
  }

  __host__ __device__ CellData(const uint4 data) : data(data) {}

  __host__ __device__ int level()  const { return data.x >> LEVEL_SHIFT; }
  __host__ __device__ int parent() const { return data.x & LEVEL_MASK; }
  __host__ __device__ int child()  const { return data.y & CHILD_MASK; }
  __host__ __device__ int nchild() const { return (data.y >> CHILD_SHIFT)+1; }
  __host__ __device__ int body()   const { return data.z; }
  __host__ __device__ int nbody()  const { return data.w; }

  __host__ __device__ bool isLeaf() const { return data.y == 0; }
  __host__ __device__ bool isNode() const { return !isLeaf(); }

  __host__ __device__ void setParent(const unsigned int parent) {
    data.x = parent | (level() << LEVEL_SHIFT);
  }
  __host__ __device__ void setChild(const unsigned int child) {
    data.y = child | (nchild()-1 << CHILD_SHIFT);
  }
};
