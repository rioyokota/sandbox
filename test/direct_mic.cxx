#include <assert.h>
#include <stdio.h>
#include <immintrin.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

void P2Pmic(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int i=0; i<ni; i+=16 ) {
    __m512 pot = _mm512_setzero_ps();                           // pot = 0
    __m512 ax = _mm512_setzero_ps();                            // ax = 0
    __m512 ay = _mm512_setzero_ps();                            // ay = 0
    __m512 az = _mm512_setzero_ps();                            // az = 0
  
    __m512 xi = _mm512_setr_ps(source[i].x,source[i+1].x,source[i+2].x,source[i+3].x,
                               source[i+4].x,source[i+5].x,source[i+6].x,source[i+7].x,
                               source[i+8].x,source[i+9].x,source[i+10].x,source[i+11].x,
                               source[i+12].x,source[i+13].x,source[i+14].x,source[i+15].x);
    __m512 yi = _mm512_setr_ps(source[i].y,source[i+1].y,source[i+2].y,source[i+3].y,
                               source[i+4].y,source[i+5].y,source[i+6].y,source[i+7].y,
                               source[i+8].y,source[i+9].y,source[i+10].y,source[i+11].y,
                               source[i+12].y,source[i+13].y,source[i+14].y,source[i+15].y);
    __m512 zi = _mm512_setr_ps(source[i].z,source[i+1].z,source[i+2].z,source[i+3].z,
                               source[i+4].z,source[i+5].z,source[i+6].z,source[i+7].z,
                               source[i+8].z,source[i+9].z,source[i+10].z,source[i+11].z,
                               source[i+12].z,source[i+13].z,source[i+14].z,source[i+15].z);
    __m512 R2 = _mm512_set1_ps(eps2);                           // R2 = eps2
  
    __m512 x2 = _mm512_set1_ps(source[0].x);                    // x2 = source->x
    x2 = _mm512_sub_ps(x2, xi);                                 // x2 = x2 - xi
    __m512 y2 = _mm512_set1_ps(source[0].y);                    // y2 = source->y
    y2 = _mm512_sub_ps(y2, yi);                                 // y2 = y2 - yi
    __m512 z2 = _mm512_set1_ps(source[0].z);                    // z2 = source->z
    z2 = _mm512_sub_ps(z2, zi);                                 // z2 = z2 - zi
    __m512 mj = _mm512_set1_ps(source[0].w);                    // mj = source->w
  
    __m512 xj = x2;                                             // xj = x2
    R2 = _mm512_fmadd_ps(x2, x2, R2);                           // R2 = R2 + x2 * x2
    __m512 yj = y2;                                             // yj = y2
    R2 = _mm512_fmadd_ps(y2, y2, R2);                           // R2 = R2 + y2 * y2
    __m512 zj = z2;                                             // zj = z2
    R2 = _mm512_fmadd_ps(z2, z2, R2);                           // R2 = R2 + z2 * z2
    __m512 invR;
  
    x2 = _mm512_set1_ps(source[1].x);
    y2 = _mm512_set1_ps(source[1].y);
    z2 = _mm512_set1_ps(source[1].z);
    for( int j=0; j<nj-2; j++ ) {
      invR = _mm512_rsqrt23_ps(R2);                             // invR = rsqrt(R2)       1
      R2 = _mm512_set1_ps(eps2);                                // R2 = eps2
      x2 = _mm512_sub_ps(x2, xi);                               // x2 = x2 - xi           2
      y2 = _mm512_sub_ps(y2, yi);                               // y2 = y2 - yi           3
      z2 = _mm512_sub_ps(z2, zi);                               // z2 = z2 - zi           4
  
      mj = _mm512_mul_ps(mj, invR);                             // mj = mj * invR         5
      pot = _mm512_add_ps(pot, mj);                             // pot = pot + mj         6
      invR = _mm512_mul_ps(invR, invR);                         // invR = invR * invR     7
      invR = _mm512_mul_ps(invR, mj);                           // invR = invR * mj       8
      mj = _mm512_set1_ps(source[j+1].w);
  
      ax = _mm512_fmadd_ps(invR, xj, ax);                       // ax = ax + xj *invR     9, 10
      xj = x2;                                                  // xj = x2
      R2 = _mm512_fmadd_ps(x2, x2, R2);                         // R2 = R2 + x2 * x2     11, 12
      x2 = _mm512_set1_ps(source[j+2].x);
  
      ay = _mm512_fmadd_ps(invR, yj, ay);                       // ay = ay + yj * invR   13, 14
      yj = y2;                                                  // yj = y2
      R2 = _mm512_fmadd_ps(y2, y2, R2);                         // R2 = R2 + y2 * y2     15, 16
      y2 = _mm512_set1_ps(source[j+2].y);
  
      az = _mm512_fmadd_ps(invR, zj, az);                       // az = az + zj * invR   17, 18
      zj = z2;                                                  // zj = z2
      R2 = _mm512_fmadd_ps(z2, z2, R2);                         // R2 = R2 + z2 * z2     19, 20
      z2 = _mm512_set1_ps(source[j+2].z);
    }
    invR = _mm512_rsqrt23_ps(R2);                               // invR = rsqrt(R2)
    R2 = _mm512_set1_ps(eps2);                                  // R2 = eps2
    x2 = _mm512_sub_ps(x2, xi);                                 // x2 = x2 - xi
    y2 = _mm512_sub_ps(y2, yi);                                 // y2 = y2 - yi
    z2 = _mm512_sub_ps(z2, zi);                                 // z2 = z2 - zi

    mj = _mm512_mul_ps(mj, invR);                               // mj = mj * invR
    pot = _mm512_add_ps(pot, mj);                               // pot = pot + mj
    invR = _mm512_mul_ps(invR, invR);                           // invR = invR * invR
    invR = _mm512_mul_ps(invR, mj);                             // invR = invR * mj
    mj = _mm512_set1_ps(source[nj-1].w);

    ax = _mm512_fmadd_ps(invR, xj, ax);                         // ax = ax + xj * invR
    xj = x2;                                                    // xj = x2
    R2 = _mm512_fmadd_ps(x2, x2, R2);                           // R2 = R2 + x2 * x2

    ay = _mm512_fmadd_ps(invR, yj, ay);                         // ay = ay + yj * invR
    yj = y2;                                                    // yj = y2
    R2 = _mm512_fmadd_ps(y2, y2, R2);                           // R2 = R2 + y2 * y2

    az = _mm512_fmadd_ps(invR, zj, az);                         // az = az + zj * invR
    zj = z2;                                                    // zj = z2
    R2 = _mm512_fmadd_ps(z2, z2, R2);                           // R2 = R2 + z2 * z2

    invR = _mm512_rsqrt23_ps(R2);                               // invR = rsqrt(R2)
    mj = _mm512_mul_ps(mj, invR);                               // mj = mj * invR
    pot = _mm512_add_ps(pot, mj);                               // pot = pot + mj
    invR = _mm512_mul_ps(invR, invR);                           // invR = invR * invR
    invR = _mm512_mul_ps(invR, mj);                             // invR = invR * mj

    ax = _mm512_fmadd_ps(invR, xj, ax);                         // ax = ax + xj * invR
    ay = _mm512_fmadd_ps(invR, yj, ay);                         // ay = ay + yj * invR
    az = _mm512_fmadd_ps(invR, zj, az);                         // az = az + zj * invR
    for( int k=0; k<16; k++ ) {
      target[i+k].w = ((float*)&pot)[k];  // target->w = pot
      target[i+k].x = ((float*)&ax)[k];   // target->x = ax
      target[i+k].y = ((float*)&ay)[k];   // target->y = ay
      target[i+k].z = ((float*)&az)[k];   // target->z = az
    }
  }
}
