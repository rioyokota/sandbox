#pragma once

#define WARP_SIZE2 5
#define WARP_SIZE 32

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

template<int N, typename T> struct vec;

template<> struct vec<4,float>  { typedef float4  type;  __host__ __device__ static float4 null() {return make_float4(0.0f, 0.0f, 0.0f, 0.0f);} };
template<> struct vec<4,double> { typedef double4 type;  __host__ __device__ static double4 null() {return make_double4(0.0, 0.0, 0.0, 0.0);} };

template<> struct vec<3,float>  { typedef float3  type;  __host__ __device__ static float3 null() {return make_float3(0.0f, 0.0f, 0.0f);} };
template<> struct vec<3,double> { typedef double3 type;  __host__ __device__ static double3 null() {return make_double3(0.0, 0.0, 0.0);} };

template<> struct vec<2,float>  { typedef float2  type;  __host__ __device__ static float2 null() {return make_float2(0.0f, 0.0f);} };
template<> struct vec<2,double> { typedef double2 type;  __host__ __device__ static double2 null() {return make_double2(0.0, 0.0);} };

struct Box
{
  float3 centre;
  float hsize;
  __device__ Box() {}
  __device__ Box(const float3 &c, float hs) : centre(c), hsize(hs) {}
};

static __host__ __device__ __forceinline__ int Octant(const float3 &lhs, const float3 &rhs)
{
  return 
    ((lhs.x <= rhs.x) << 0) +
    ((lhs.y <= rhs.y) << 1) +
    ((lhs.z <= rhs.z) << 2);
};

static __device__ __forceinline__ Box ChildBox(const Box &box, const int oct)
{
  const float s = 0.5f * box.hsize;
  return Box(make_float3(
        box.centre.x + s * ((oct&1) ? 1.0f : -1.0f),
        box.centre.y + s * ((oct&2) ? 1.0f : -1.0f),
        box.centre.z + s * ((oct&4) ? 1.0f : -1.0f)
        ), 
      s);
}
