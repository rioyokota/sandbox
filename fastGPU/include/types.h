#ifndef _TYPES_H_
#define _TYPES_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <vector>

#define NCRIT 32
#define NTHREAD 128
#define NBLOCK 512
#define WARP_SIZE 32
#define MAXLEVELS 30
#define LMEM_STACK_SIZE 2048
#define NWARP (NTHREAD / WARP_SIZE)

#if NCRIT == 8
#define NCRIT2 3
#define CRITBIT 29
#define CRITMASK 0x1FFFFFFF
#define INVCMASK 0xE0000000
#elif NCRIT == 16
#define NCRIT2 4
#define CRITBIT 28
#define CRITMASK 0x0FFFFFFF
#define INVCMASK 0xF0000000
#elif NCRIT == 32
#define NCRIT2 5
#define CRITBIT 27
#define CRITMASK 0x07FFFFFF
#define INVCMASK 0xF8000000
#elif NCRIT == 64
#define NCRIT2 6
#define CRITBIT 26
#define CRITMASK 0x03FFFFFF
#define INVCMASK 0xFC000000
#elif NCRIT == 128
#define NCRIT2 7
#define CRITBIT 25
#define CRITMASK 0x01FFFFFF
#define INVCMASK 0xFE000000
#else
#error "Please choose correct NCRIT available in node_specs.h"
#endif

#if WARP_SIZE == 16
#define WARP_SIZE2 4
#elif WARP_SIZE == 32
#define WARP_SIZE2 5
#else
#error "Please choose correct WARP_SIZE available in node_specs.h"
#endif

#if NCRIT > 2*WARP_SIZE
#error "NCRIT in include/node_specs.h must be <= WARP_SIZE"
#endif

#if NCRIT < NLEAF
#error "Fatal, NCRIT < NLEAF. Please check that NCRIT >= NLEAF"
#endif

#define ALIGN(a, b) ((a - 1) / b + 1)

#define CU_SAFE_CALL(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line ) {
  if(cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
    exit(-1);
  }
}

template<class T>
class cudaVec {
private:
  int size;
  T *devcPtr;
  T *hostPtr;

public:
  cudaVec() : size(0), devcPtr(NULL), hostPtr(NULL) {}

  ~cudaVec() {
    cudaFree(devcPtr);
#if PINNED
    cudaFreeHost((void*)hostPtr);
#else
    free(hostPtr);
#endif
  }
  
  void alloc(int n) {
    assert(size == 0);
    size = n;
#if PINNED
    CU_SAFE_CALL(cudaMallocHost((T**)&hostPtr, size*sizeof(T)));
#else
    hostPtr = (T*)malloc(size*sizeof(T));
#endif
    CU_SAFE_CALL(cudaMalloc((T**)&devcPtr, size*sizeof(T)));
  }

  void zeros() {
    CU_SAFE_CALL(cudaMemset((void*)devcPtr, 0, size*sizeof(T)));     
  }

  void ones() {
    CU_SAFE_CALL(cudaMemset((void*)devcPtr, 1, size*sizeof(T)));
  }

  void d2h() {
    CU_SAFE_CALL(cudaMemcpy(hostPtr, devcPtr, size*sizeof(T), cudaMemcpyDeviceToHost));
  }
  
  void d2h(int n) {
    CU_SAFE_CALL(cudaMemcpy(hostPtr, devcPtr, n*sizeof(T),cudaMemcpyDeviceToHost));
  }    
  
  void h2d() {
    CU_SAFE_CALL(cudaMemcpy(devcPtr, hostPtr, size*sizeof(T),cudaMemcpyHostToDevice ));
  }
  
  void h2d(int n) {
    CU_SAFE_CALL(cudaMemcpy(devcPtr, hostPtr, n*sizeof(T),cudaMemcpyHostToDevice));
  }        

  void tex(const char *symbol) {
    const textureReference *texref;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    CU_SAFE_CALL(cudaGetTextureReference(&texref,symbol));
    CU_SAFE_CALL(cudaBindTexture(0,texref,(void*)devcPtr,&channelDesc,sizeof(T)*size));
  }

  T& operator[] (int i){ return hostPtr[i]; }
  T* devc() {return devcPtr;}
};

#endif
