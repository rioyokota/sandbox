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
#include "cudavec.h"
#include "macros.h"
#include "vec.h"

typedef vec<3,float> vec3;
typedef vec<4,float> vec4;

const int  P     = 3;
const float EPS2  = 0.0001;
const float THETA = .6;

const int MTERM = P*(P+1)*(P+2)/6;
const int LTERM = (P+1)*(P+2)*(P+3)/6;
typedef vec<MTERM,float> vecM;
typedef vec<LTERM,float> vecL;

namespace {
__host__ __device__
inline vec3 make_vec3(float x, float y, float z) {
  vec3 output;
  output[0] = x;
  output[1] = y;
  output[2] = z;
  return output;
}

__host__ __device__
inline vec3 make_vec3(vec4 input) {
  vec3 output;
  output[0] = input[0];
  output[1] = input[1];
  output[2] = input[2];
  return output;
}

__host__ __device__
inline vec4 make_vec4(float x, float y, float z, float w) {
  vec4 output;
  output[0] = x;
  output[1] = y;
  output[2] = z;
  output[3] = w;
  return output;
}
}
#endif
