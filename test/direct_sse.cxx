#include <assert.h>
#include <stdio.h>
#include <xmmintrin.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2Psse(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=4 ) {
    __m128 ax = _mm_setzero_ps();         // ax = 0
    __m128 ay = _mm_setzero_ps();         // ay = 0
    __m128 az = _mm_setzero_ps();         // az = 0
    __m128 phi = _mm_setzero_ps();        // phi = 0
  
    __m128 xi = _mm_setr_ps(source[i].x,source[i+1].x,source[i+2].x,source[i+3].x);   // xi = target->x
    __m128 yi = _mm_setr_ps(source[i].y,source[i+1].y,source[i+2].y,source[i+3].y);   // yi = target->y
    __m128 zi = _mm_setr_ps(source[i].z,source[i+1].z,source[i+2].z,source[i+3].z);   // zi = target->z
    __m128 R2 = _mm_set1_ps(eps2);      // R2 = eps2
  
    __m128 x2 = _mm_set1_ps(source[0].x); // x2 = source->x
    x2 = _mm_sub_ps(x2, xi);              // x2 = x2 - xi
    __m128 y2 = _mm_set1_ps(source[0].y); // y2 = source->y
    y2 = _mm_sub_ps(y2, yi);              // y2 = y2 - yi
    __m128 z2 = _mm_set1_ps(source[0].z); // z2 = source->z
    z2 = _mm_sub_ps(z2, zi);              // z2 = z2 - zi
    __m128 mj = _mm_set1_ps(source[0].w); // mj = source->w
  
    __m128 xj = x2;                       // xj = x2
    x2 = _mm_mul_ps(x2, x2);              // x2 = x2 * x2
    R2 = _mm_add_ps(R2, x2);              // R2 = R2 + x2
    __m128 yj = y2;                       // yj = y2
    y2 = _mm_mul_ps(y2, y2);              // y2 = y2 * y2
    R2 = _mm_add_ps(R2, y2);              // R2 = R2 + y2
    __m128 zj = z2;                       // zj = z2
    z2 = _mm_mul_ps(z2, z2);              // z2 = z2 * z2
    R2 = _mm_add_ps(R2, z2);              // R2 = R2 + z2
    __m128 invR;  

    x2 = _mm_set1_ps(source[1].x);      // x2 = source->x 
    y2 = _mm_set1_ps(source[1].y);      // y2 = source->y
    z2 = _mm_set1_ps(source[1].z);      // z2 = source->z
    for( int j=0; j<nj-2; j++ ) {
      invR = _mm_rsqrt_ps(R2);            // invR = rsqrt(R2)       1
      R2 = _mm_set1_ps(eps2);           // R2 = eps2
      x2 = _mm_sub_ps(x2, xi);            // x2 = x2 - xi           2
      y2 = _mm_sub_ps(y2, yi);            // y2 = y2 - yi           3
      z2 = _mm_sub_ps(z2, zi);            // z2 = z2 - zi           4
  
      mj = _mm_mul_ps(mj, invR);          // mj = mj * invR         5
      phi = _mm_add_ps(phi, mj);          // phi = phi + mj         6
      invR = _mm_mul_ps(invR, invR);      // invR = invR * invR     7
      invR = _mm_mul_ps(invR, mj);        // invR = invR * mj       8
      mj = _mm_set1_ps(source[j+1].w);  // mj = source->w
  
      xj = _mm_mul_ps(xj, invR);          // xj = xj * invR         9
      ax = _mm_add_ps(ax, xj);            // ax = ax + xj          10
      xj = x2;                            // xj = x2
      x2 = _mm_mul_ps(x2, x2);            // x2 = x2 * x2          11
      R2 = _mm_add_ps(R2, x2);            // R2 = R2 + x2          12
      x2 = _mm_set1_ps(source[j+2].x);  // x2 = source->x
  
      yj = _mm_mul_ps(yj, invR);          // yj = yj * invR        13
      ay = _mm_add_ps(ay, yj);            // ay = ay + yj          14
      yj = y2;                            // yj = y2
      y2 = _mm_mul_ps(y2, y2);            // y2 = y2 * y2          15
      R2 = _mm_add_ps(R2, y2);            // R2 = R2 + y2          16
      y2 = _mm_set1_ps(source[j+2].y);  // y2 = source->y
  
      zj = _mm_mul_ps(zj, invR);          // zj = zj * invR        17
      az = _mm_add_ps(az, zj);            // az = az + zj          18
      zj = z2;                            // zj = z2
      z2 = _mm_mul_ps(z2, z2);            // z2 = z2 * z2          19
      R2 = _mm_add_ps(R2, z2);            // R2 = R2 + z2          20
      z2 = _mm_set1_ps(source[j+2].z);  // z2 = source->z
    }
    invR = _mm_rsqrt_ps(R2);              // invR = rsqrt(R2)
    R2 = _mm_set1_ps(eps2);             // R2 = eps2
    x2 = _mm_sub_ps(x2, xi);              // x2 = x2 - xi
    y2 = _mm_sub_ps(y2, yi);              // y2 = y2 - yi
    z2 = _mm_sub_ps(z2, zi);              // z2 = z2 - zi
  
    mj = _mm_mul_ps(mj, invR);            // mj = mj * invR
    phi = _mm_add_ps(phi, mj);            // phi = phi + mj
    invR = _mm_mul_ps(invR, invR);        // invR = invR * invR
    invR = _mm_mul_ps(invR, mj);          // invR = invR * mj
    mj = _mm_set1_ps(source[nj-1].w);   // mj = source->w
  
    xj = _mm_mul_ps(xj, invR);            // xj = xj * invR
    ax = _mm_add_ps(ax, xj);              // ax = ax + xj
    xj = x2;                              // xj = x2
    x2 = _mm_mul_ps(x2, x2);              // x2 = x2 * x2
    R2 = _mm_add_ps(R2, x2);              // R2 = R2 + x2
  
    yj = _mm_mul_ps(yj, invR);            // yj = yj * invR
    ay = _mm_add_ps(ay, yj);              // ay = ay + yj
    yj = y2;                              // yj = y2
    y2 = _mm_mul_ps(y2, y2);              // y2 = y2 * y2
    R2 = _mm_add_ps(R2, y2);              // R2 = R2 + y2
  
    zj = _mm_mul_ps(zj, invR);            // zj = zj * invR
    az = _mm_add_ps(az, zj);              // az = az + zj
    zj = z2;                              // zj = z2
    z2 = _mm_mul_ps(z2, z2);              // z2 = z2 * z2
    R2 = _mm_add_ps(R2, z2);              // R2 = R2 + z2

    invR = _mm_rsqrt_ps(R2);              // invR = rsqrt(R2)
    mj = _mm_mul_ps(mj, invR);            // mj = mj * invR
    phi = _mm_add_ps(phi, mj);            // phi = phi + mj
    invR = _mm_mul_ps(invR, invR);        // invR = invR * invR
    invR = _mm_mul_ps(invR, mj);          // invR = invR * mj

    xj = _mm_mul_ps(xj, invR);            // xj = xj * invR
    ax = _mm_add_ps(ax, xj);              // ax = ax + xj
    yj = _mm_mul_ps(yj, invR);            // yj = yj * invR
    ay = _mm_add_ps(ay, yj);              // ay = ay + yj
    zj = _mm_mul_ps(zj, invR);            // zj = zj * invR
    az = _mm_add_ps(az, zj);              // az = az + zj
    for( int k=0; k<4; k++ ) {
      target[i+k].x = ((float*)&ax)[k];   // target->x = ax
      target[i+k].y = ((float*)&ay)[k];   // target->y = ay
      target[i+k].z = ((float*)&az)[k];   // target->z = az
      target[i+k].w = ((float*)&phi)[k];  // target->w = phi
    }
  }
}
