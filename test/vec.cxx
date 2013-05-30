#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
#include <xmmintrin.h>
#include "vec.h"

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

const int N = 1 << 15;
const float OPS = 20. * N * N * 1e-9;
const float EPS2 = 1e-6;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void P2P(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for (int i=0; i<ni; i++) {
    float ax = 0;
    float ay = 0;
    float az = 0;
    float phi = 0;
    float xi = source[i].x;
    float yi = source[i].y;
    float zi = source[i].z;
    for (int j=0; j<nj; j++) {
      float dx = source[j].x - xi;
      float dy = source[j].y - yi;
      float dz = source[j].z - zi;
      float R2 = dx * dx + dy * dy + dz * dz + eps2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = source[j].w * invR * invR * invR;
      phi += source[j].w * invR;
      ax += dx * invR3;
      ay += dy * invR3;
      az += dz * invR3;
    }
    target[i].w = phi;
    target[i].x = ax;
    target[i].y = ay;
    target[i].z = az;
  }
}

typedef vec<4,float> vec4;

void P2Psse(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=4 ) {
    vec4 ax = 0;
    vec4 ay = 0;
    vec4 az = 0;
    vec4 phi = 0;

    vec4 xi(source[i].x,source[i+1].x,source[i+2].x,source[i+3].x);
    vec4 yi(source[i].y,source[i+1].y,source[i+2].y,source[i+3].y);
    vec4 zi(source[i].z,source[i+1].z,source[i+2].z,source[i+3].z);
    vec4 R2 = eps2;

    vec4 x2 = source[0].x;
    x2 -= xi;
    vec4 y2 = source[0].y;
    y2 -= yi;
    vec4 z2 = source[0].z;
    z2 -= zi;
    vec4 mj = source[0].w;

    vec4 xj = x2;
    R2 += x2 * x2;
    vec4 yj = y2;
    R2 += y2 * y2;
    vec4 zj = z2;
    R2 += z2 * z2;
    vec4 invR;

    x2 = source[1].x;
    y2 = source[1].y;
    z2 = source[1].z;
    for( int j=0; j<nj-2; j++ ) {
      invR = rsqrt(R2);
      R2 = eps2;
      x2 -= xi;
      y2 -= yi;
      z2 -= zi;

      mj *= invR;
      phi += mj;
      invR = invR * invR * mj;
      mj = source[j+1].w;

      ax += xj * invR;
      xj = x2;
      R2 += x2 * x2;
      x2 = source[j+2].x;

      ay += yj * invR;
      yj = y2;
      R2 += y2 * y2;
      y2 = source[j+2].y;

      az += zj * invR;
      zj = z2;
      R2 += z2 * z2;
      z2 = source[j+2].z;
    }
    invR = rsqrt(R2);
    R2 = eps2;
    x2 -= xi;
    y2 -= yi;
    z2 -= zi;
    mj *= invR;
    phi += mj;
    invR = invR * invR * mj;
    mj = source[nj-1].w;
    ax += xj * invR;
    xj = x2;
    R2 += x2 * x2;
    ay += yj * invR;
    yj = y2;
    R2 += y2 * y2;
    az += zj * invR;
    zj = z2;
    R2 += z2 * z2;

    invR = rsqrt(R2);
    mj *= invR;
    phi += mj;
    invR = invR * invR * mj;
    ax += xj * invR;
    ay += yj * invR;
    az += zj * invR;
    for( int k=0; k<4; k++ ) {
      target[i+k].x = ax[k];
      target[i+k].y = ay[k];
      target[i+k].z = az[k];
      target[i+k].w = phi[k];
    }
  }
}


int main() {
// ALLOCATE
  float4 *sourceHost = new float4 [N];
  float4 *targetHost = new float4 [N];
  float4 *targetSSE = new float4 [N];
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = drand48();
    sourceHost[i].y = drand48();
    sourceHost[i].z = drand48();
    sourceHost[i].w = drand48() / N;
  }
  std::cout << std::scientific << "N      : " << N << std::endl;

// Host P2P
  double tic = get_time();
  P2P(targetHost,sourceHost,N,N,EPS2);
  double toc = get_time();
  std::cout << std::scientific << "No SSE : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// SSE P2P
  tic = get_time();
  P2Psse(targetSSE,sourceHost,N,N,EPS2);
  toc = get_time();
  std::cout << std::scientific << "SSE    : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// COMPARE RESULTS
  float pd = 0, pn = 0, fd = 0, fn = 0;
  for( int i=0; i<N; i++ ) {
    targetHost[i].w -= sourceHost[i].w / sqrtf(EPS2);
    targetSSE[i].w -= sourceHost[i].w / sqrtf(EPS2);
    pd += (targetHost[i].w - targetSSE[i].w) * (targetHost[i].w - targetSSE[i].w);
    pn += targetHost[i].w * targetHost[i].w;
    fd += (targetHost[i].x - targetSSE[i].x) * (targetHost[i].x - targetSSE[i].x)
        + (targetHost[i].y - targetSSE[i].y) * (targetHost[i].y - targetSSE[i].y)
        + (targetHost[i].z - targetSSE[i].z) * (targetHost[i].z - targetSSE[i].z);
    fn += targetHost[i].x * targetHost[i].x + targetHost[i].y * targetHost[i].y + targetHost[i].z * targetHost[i].z;
  }
  std::cout << std::scientific << "P ERR  : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR  : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  delete[] sourceHost;
  delete[] targetHost;
  delete[] targetSSE;
}
