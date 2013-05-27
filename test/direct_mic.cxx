#include <assert.h>
#include <stdio.h>
#include "simd.h"

struct float4 {
  float x;
  float y;
  float z;
  float w;
}; 

void P2Pmic(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=16 ) {
    fvec16 pot = 0;
    fvec16 ax = 0;
    fvec16 ay = 0;
    fvec16 az = 0;
  
    fvec16 xi(source[i   ].x,source[i+1 ].x,source[i+2 ].x,source[i+3 ].x,
             source[i+4 ].x,source[i+5 ].x,source[i+6 ].x,source[i+7 ].x,
             source[i+8 ].x,source[i+9 ].x,source[i+10].x,source[i+11].x,
	     source[i+12].x,source[i+13].x,source[i+14].x,source[i+15].x);
    fvec16 yi(source[i   ].y,source[i+1 ].y,source[i+2 ].y,source[i+3 ].y,
             source[i+4 ].y,source[i+5 ].y,source[i+6 ].y,source[i+7 ].y,
             source[i+8 ].y,source[i+9 ].y,source[i+10].y,source[i+11].y,
	     source[i+12].y,source[i+13].y,source[i+14].y,source[i+15].y);
    fvec16 zi(source[i   ].z,source[i+1 ].z,source[i+2 ].z,source[i+3 ].z,
             source[i+4 ].z,source[i+5 ].z,source[i+6 ].z,source[i+7 ].z,
             source[i+8 ].z,source[i+9 ].z,source[i+10].z,source[i+11].z,
	     source[i+12].z,source[i+13].z,source[i+14].z,source[i+15].z);
    fvec16 R2 = eps2;
  
    fvec16 x2 = source[0].x;
    x2 -= xi;
    fvec16 y2 = source[0].y;
    y2 -= yi;
    fvec16 z2 = source[0].z;
    z2 -= zi;
    fvec16 mj = source[0].w;
  
    fvec16 xj = x2;
    R2 += x2 * x2;
    fvec16 yj = y2;
    R2 += y2 * y2;
    fvec16 zj = z2;
    R2 += z2 * z2;
    fvec16 invR;
  
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
    for( int k=0; k<16; k++ ) {
      target[i+k].x = ax[k];
      target[i+k].y = ay[k];
      target[i+k].z = az[k];
      target[i+k].w = pot[k];
    }
  }
}
