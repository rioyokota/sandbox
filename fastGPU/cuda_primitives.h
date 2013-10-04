#pragma once
  
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

static __device__ __forceinline__
uint inclusiveScanInt(const int value) {
  uint sum = value;
#pragma unroll
  for (int i=0; i<WARP_SIZE2; ++i)
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
int exclusiveScanBool(const bool p) {
  const uint b = __ballot(p);
  return __popc(b & lanemask_lt());
}

static __device__ __forceinline__
int reduceBool(const bool p) {
  const uint b = __ballot(p);
  return __popc(b);
}