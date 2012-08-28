#include <assert.h>
#include <stdio.h>
#include <immintrin.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2Pavx(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=8 ) {
    __m256 pot = _mm256_setzero_ps();                           // pot = 0
    __m256 ax = _mm256_setzero_ps();                            // ax = 0
    __m256 ay = _mm256_setzero_ps();                            // ay = 0
    __m256 az = _mm256_setzero_ps();                            // az = 0
  
    __m256 xi = _mm256_setr_ps(source[i].x,source[i+1].x,source[i+2].x,source[i+3].x,
                               source[i+4].x,source[i+5].x,source[i+6].x,source[i+7].x);
    __m256 yi = _mm256_setr_ps(source[i].y,source[i+1].y,source[i+2].y,source[i+3].y,
                               source[i+4].y,source[i+5].y,source[i+6].y,source[i+7].y);
    __m256 zi = _mm256_setr_ps(source[i].z,source[i+1].z,source[i+2].z,source[i+3].z,
                               source[i+4].z,source[i+5].z,source[i+6].z,source[i+7].z);
    __m256 R2 = _mm256_set1_ps(eps2);                           // R2 = eps2
  
    __m256 x2 = _mm256_set1_ps(source[0].x);                    // x2 = source->x
    x2 = _mm256_sub_ps(x2, xi);                                 // x2 = x2 - xi
    __m256 y2 = _mm256_set1_ps(source[0].y);                    // y2 = source->y
    y2 = _mm256_sub_ps(y2, yi);                                 // y2 = y2 - yi
    __m256 z2 = _mm256_set1_ps(source[0].z);                    // z2 = source->z
    z2 = _mm256_sub_ps(z2, zi);                                 // z2 = z2 - zi
    __m256 mj = _mm256_set1_ps(source[0].w);                    // mj = source->w
  
    __m256 xj = x2;                                             // xj = x2
    x2 = _mm256_mul_ps(x2, x2);                                 // x2 = x2 * x2
    R2 = _mm256_add_ps(R2, x2);                                 // R2 = R2 + x2
    __m256 yj = y2;                                             // yj = y2
    y2 = _mm256_mul_ps(y2, y2);                                 // y2 = y2 * y2
    R2 = _mm256_add_ps(R2, y2);                                 // R2 = R2 + y2
    __m256 zj = z2;                                             // zj = z2
    z2 = _mm256_mul_ps(z2, z2);                                 // z2 = z2 * z2
    R2 = _mm256_add_ps(R2, z2);                                 // R2 = R2 + z2
  
    x2 = _mm256_set1_ps(source[1].x);
    y2 = _mm256_set1_ps(source[1].y);
    z2 = _mm256_set1_ps(source[1].z);
    for( int j=0; j<nj; j++ ) {
      __m256 invR = _mm256_rsqrt_ps(R2);                        // invR = rsqrt(R2)       1
      R2 = _mm256_set1_ps(eps2);                                // R2 = eps2
      x2 = _mm256_sub_ps(x2, xi);                               // x2 = x2 - xi           2
      y2 = _mm256_sub_ps(y2, yi);                               // y2 = y2 - yi           3
      z2 = _mm256_sub_ps(z2, zi);                               // z2 = z2 - zi           4
  
      mj = _mm256_mul_ps(mj, invR);                             // mj = mj * invR         5
      pot = _mm256_add_ps(pot, mj);                             // pot = pot + mj         6
      invR = _mm256_mul_ps(invR, invR);                         // invR = invR * invR     7
      invR = _mm256_mul_ps(invR, mj);                           // invR = invR * mj       8
      mj = _mm256_set1_ps(source[j+1].w);
  
      xj = _mm256_mul_ps(xj, invR);                             // xj = xj * invR         9
      ax = _mm256_add_ps(ax, xj);                               // ax = ax + xj          10
      xj = x2;                                                  // xj = x2
      x2 = _mm256_mul_ps(x2, x2);                               // x2 = x2 * x2          11
      R2 = _mm256_add_ps(R2, x2);                               // R2 = R2 + x2          12
      x2 = _mm256_set1_ps(source[j+2].x);
  
      yj = _mm256_mul_ps(yj, invR);                             // yj = yj * invR        13
      ay = _mm256_add_ps(ay, yj);                               // ay = ay + yj          14
      yj = y2;                                                  // yj = y2
      y2 = _mm256_mul_ps(y2, y2);                               // y2 = y2 * y2          15
      R2 = _mm256_add_ps(R2, y2);                               // R2 = R2 + y2          16
      y2 = _mm256_set1_ps(source[j+2].y);
  
      zj = _mm256_mul_ps(zj, invR);                             // zj = zj * invR        17
      az = _mm256_add_ps(az, zj);                               // az = az + zj          18
      zj = z2;                                                  // zj = z2
      z2 = _mm256_mul_ps(z2, z2);                               // z2 = z2 * z2          19
      R2 = _mm256_add_ps(R2, z2);                               // R2 = R2 + z2          20
      z2 = _mm256_set1_ps(source[j+2].z);
    }
    for( int k=0; k<8; k++ ) {
      target[i+k].w = ((float*)&pot)[k];  // target->w = pot
      target[i+k].x = ((float*)&ax)[k];   // target->x = ax
      target[i+k].y = ((float*)&ay)[k];   // target->y = ay
      target[i+k].z = ((float*)&az)[k];   // target->z = az
    }
  }
}
