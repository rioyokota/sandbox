#include <assert.h>
#include <stdio.h>
#include "simd.h"

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2Psse(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=4 ) {
    fvec4 pot = 0;
    fvec4 ax = 0;
    fvec4 ay = 0;
    fvec4 az = 0;

    fvec4 xi(source[i+0].x,source[i+1].x,source[i+2].x,source[i+3].x);
    fvec4 yi(source[i+0].y,source[i+1].y,source[i+2].y,source[i+3].y);
    fvec4 zi(source[i+0].z,source[i+1].z,source[i+2].z,source[i+3].z);
    fvec4 R2 = eps2;

    fvec4 x2 = source[0].x;
    x2 -= xi;
    fvec4 y2 = source[0].y;
    y2 -= yi;
    fvec4 z2 = source[0].z;
    z2 -= zi;
    fvec4 mj = source[0].w;

    fvec4 xj = x2;
    R2 += x2 * x2;
    fvec4 yj = y2;
    R2 += y2 * y2;
    fvec4 zj = z2;
    R2 += z2 * z2;
    fvec4 invR;

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
      pot += mj;
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
    pot += mj;
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
    pot += mj;
    invR = invR * invR * mj;

    ax += xj * invR;
    ay += yj * invR;
    az += zj * invR;
    for( int k=0; k<4; k++ ) {
      target[i+k].x = ax[k];
      target[i+k].y = ay[k];
      target[i+k].z = az[k];
      target[i+k].w = pot[k];
    }
  }
}
