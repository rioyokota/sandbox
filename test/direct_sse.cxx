#include <assert.h>
#include <xmmintrin.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

struct float16 {
  float x[4];
  float y[4];
  float z[4];
  float w[4];
};

void P2PsseKernel(float16 *target, const float4 *source, const int &n){
  assert(((unsigned long)source & 15) == 0);
  assert(((unsigned long)target & 15) == 0);

  __m128 ax = _mm_setzero_ps();         // ax = 0
  __m128 ay = _mm_setzero_ps();         // ay = 0
  __m128 az = _mm_setzero_ps();         // az = 0
  __m128 phi = _mm_setzero_ps();        // phi = 0

  __m128 xi = _mm_load_ps(target->x);   // xi = target->x
  __m128 yi = _mm_load_ps(target->y);   // yi = target->y
  __m128 zi = _mm_load_ps(target->z);   // zi = target->z
  __m128 R2 = _mm_load_ps(target->w);   // R2 = target->w

  __m128 x2 = _mm_load1_ps(&source->x); // x2 = source->x
  x2 = _mm_sub_ps(x2, xi);              // x2 = x2 - xi
  __m128 y2 = _mm_load1_ps(&source->y); // y2 = source->y
  y2 = _mm_sub_ps(y2, yi);              // y2 = y2 - yi
  __m128 z2 = _mm_load1_ps(&source->z); // z2 = source->z
  z2 = _mm_sub_ps(z2, zi);              // z2 = z2 - zi
  __m128 mj = _mm_load1_ps(&source->w); // mj = source->w

  __m128 xj = x2;                       // xj = x2
  x2 = _mm_mul_ps(x2, x2);              // x2 = x2 * x2
  R2 = _mm_add_ps(R2, x2);              // R2 = R2 + x2
  __m128 yj = y2;                       // yj = y2
  y2 = _mm_mul_ps(y2, y2);              // y2 = y2 * y2
  R2 = _mm_add_ps(R2, y2);              // R2 = R2 + y2
  __m128 zj = z2;                       // zj = z2
  z2 = _mm_mul_ps(z2, z2);              // z2 = z2 * z2
  R2 = _mm_add_ps(R2, z2);              // R2 = R2 + z2

  x2 = _mm_load_ps(&source[1].x);       // x2 = next source->x,y,z,w 
  y2 = x2;                              // y2 = x2;
  z2 = x2;                              // z2 = x2;
  for( int j=0; j<n; j++ ) {
    __m128 invR = _mm_rsqrt_ps(R2);     // invR = rsqrt(R2)       1
    source++;
    R2 = _mm_load_ps(target->w);        // R2 = target->w
    x2 = _mm_shuffle_ps(x2, x2, 0x00);  // x2 = source->x
    x2 = _mm_sub_ps(x2, xi);            // x2 = x2 - xi           2
    y2 = _mm_shuffle_ps(y2, y2, 0x55);  // y2 = source->y
    y2 = _mm_sub_ps(y2, yi);            // y2 = y2 - yi           3
    z2 = _mm_shuffle_ps(z2, z2, 0xaa);  // z2 = source->z
    z2 = _mm_sub_ps(z2, zi);            // z2 = z2 - zi           4

    mj = _mm_mul_ps(mj, invR);          // mj = mj * invR         5
    phi = _mm_add_ps(phi, mj);          // phi = phi + mj         6
    invR = _mm_mul_ps(invR, invR);      // invR = invR * invR     7
    invR = _mm_mul_ps(invR, mj);        // invR = invR * mj       8
    mj = _mm_load_ps(&source->x);       // mj = source->x,y,z,w
    mj = _mm_shuffle_ps(mj, mj, 0xff);  // mj = source->w

    xj = _mm_mul_ps(xj, invR);          // xj = xj * invR         9
    ax = _mm_add_ps(ax, xj);            // ax = ax + xj          10
    xj = x2;                            // xj = x2
    x2 = _mm_mul_ps(x2, x2);            // x2 = x2 * x2          11
    R2 = _mm_add_ps(R2, x2);            // R2 = R2 + x2          12
    x2 = _mm_load_ps(&source[1].x);     // x2 = next source->x,y,z,w

    yj = _mm_mul_ps(yj, invR);          // yj = yj * invR        13
    ay = _mm_add_ps(ay, yj);            // ay = ay + yj          14
    yj = y2;                            // yj = y2
    y2 = _mm_mul_ps(y2, y2);            // y2 = y2 * y2          15
    R2 = _mm_add_ps(R2, y2);            // R2 = R2 + y2          16
    y2 = x2;                            // y2 = x2

    zj = _mm_mul_ps(zj, invR);          // zj = zj * invR        17
    az = _mm_add_ps(az, zj);            // az = az + zj          18
    zj = z2;                            // zj = z2
    z2 = _mm_mul_ps(z2, z2);            // z2 = z2 * z2          19
    R2 = _mm_add_ps(R2, z2);            // R2 = R2 + z2          20
    z2 = x2;                            // z2 = x2
  }
  _mm_store_ps(target->x, ax);          // target->x = ax
  _mm_store_ps(target->y, ay);          // target->y = ay
  _mm_store_ps(target->z, az);          // target->z = az
  _mm_store_ps(target->w, phi);         // target->w = phi
}

void P2Psse(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int base=0; base<ni; base+=4 ) {
    float16 target4;
    for( int i=0; i<4; i++ ) {
      target4.x[i] = source[base+i].x;
      target4.y[i] = source[base+i].y;
      target4.z[i] = source[base+i].z;
      target4.w[i] = eps2;
    }
    P2PsseKernel(&target4, source, nj);
    for( int i=0; i<4; i++ ) {
      target[base+i].x = target4.x[i];
      target[base+i].y = target4.y[i];
      target[base+i].z = target4.z[i];
      target[base+i].w = target4.w[i];
    }
  }
}
