#pragma once
  
template<typename Tex, typename T>
  static void bindTexture(Tex &tex, const T *ptr, const int size) {
  tex.addressMode[0] = cudaAddressModeWrap;
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode     = cudaFilterModePoint;
  tex.normalized     = false;
  CUDA_SAFE_CALL(cudaBindTexture(0, tex, ptr, size*sizeof(T)));
}

template<typename Tex>
static void unbindTexture(Tex &tex) {
  CUDA_SAFE_CALL(cudaUnbindTexture(tex));
}

static __device__ __forceinline__ 
void getMinMax(float3 &_rmin, float3 &_rmax, const float3 pos) {
  float3 rmin = pos;
  float3 rmax = rmin;
#pragma unroll
  for (int i=0; i<WARP_SIZE2; i++) {
    rmin.x = min(rmin.x, __shfl_xor(rmin.x, 1<<i));
    rmin.y = min(rmin.y, __shfl_xor(rmin.y, 1<<i));
    rmin.z = min(rmin.z, __shfl_xor(rmin.z, 1<<i));
    rmax.x = max(rmax.x, __shfl_xor(rmax.x, 1<<i));
    rmax.y = max(rmax.y, __shfl_xor(rmax.y, 1<<i));
    rmax.z = max(rmax.z, __shfl_xor(rmax.z, 1<<i));
  }
  _rmin.x = min(_rmin.x, rmin.x);
  _rmin.y = min(_rmin.y, rmin.y);
  _rmin.z = min(_rmin.z, rmin.z);
  _rmax.x = max(_rmax.x, rmax.x);
  _rmax.y = max(_rmax.y, rmax.y);
  _rmax.z = max(_rmax.z, rmax.z);
}

// Int scan

static __device__ __forceinline__
uint shflScan(uint partial, uint offset) {
  uint result;
  asm("{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(offset), "r"(partial));
  return result;
}

template<int SIZE2>
static __device__ __forceinline__
uint inclusiveScan(const int value) {
  uint sum = value;
#pragma unroll
  for (int i=0; i<SIZE2; ++i)
    sum = shflScan(sum, 1 << i);
  return sum;
}

// Bool scan

static __device__ __forceinline__
int lanemask_lt() {
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__
int warpBinExclusiveScan1(const bool p) {
  const uint b = __ballot(p);
  return __popc(b & lanemask_lt());
}

static __device__ __forceinline__
int2 warpBinExclusiveScan(const bool p) {
  const uint b = __ballot(p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}

static __device__ __forceinline__
int warpBinReduce(const bool p) {
  const uint b = __ballot(p);
  return __popc(b);
}

// Int seg scan

static __device__ __forceinline__
int lanemask_le() {
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__
int shflSegScan(int partial, uint offset, uint distance) {
  asm("{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0, %1, %2, 0;"
      "setp.le.u32 p, %2, %3;"
      "@p add.u32 %1, r0, %1;"
      "mov.u32 %0, %1;}"
      : "=r"(partial) : "r"(partial), "r"(offset), "r"(distance));
  return partial;
}

template<const int SIZE2>
static __device__ __forceinline__
int inclusiveSegscan(int value, const int distance) {
  for (int i=0; i<SIZE2; i++)
    value = shflSegScan(value, 1<<i, distance);
  return value;
}

static __device__ __forceinline__
int inclusive_segscan_warp(const int packed_value, const int carryValue) {
  const int flag = packed_value < 0;
  const int mask = -flag;
  const int value = (~mask & packed_value) + (mask & (-1-packed_value));
  const int flags = __ballot(flag);
  const int dist_block = __clz(__brev(flags));
  const int laneIdx = threadIdx.x & (WARP_SIZE - 1);
  const int distance = __clz(flags & lanemask_le()) + laneIdx - 31;
  const int val = inclusiveSegscan<WARP_SIZE2>(value, min(distance, laneIdx)) +
    (carryValue & (-(laneIdx < dist_block)));
  return val;
}
