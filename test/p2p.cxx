#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <papi.h>
#include <sys/time.h>
#include <xmmintrin.h>

const int N = 10001;
const float OPS = 20. * N * N * 1e-9;
const float EPS2 = 1e-6;

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

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void P2Psse(float16 &target4, const float4 *source, const int &n){
  __m128 ax = _mm_setzero_ps();         // ax = 0
  __m128 ay = _mm_setzero_ps();         // ay = 0
  __m128 az = _mm_setzero_ps();         // az = 0
  __m128 phi = _mm_setzero_ps();        // phi = 0

  __m128 xi = _mm_load_ps(target4.x);   // xi = target4.x
  __m128 yi = _mm_load_ps(target4.y);   // yi = target4.y
  __m128 zi = _mm_load_ps(target4.z);   // zi = target4.z
  __m128 R2 = _mm_load_ps(target4.w);   // R2 = target4.w

  __m128 x2 = _mm_load1_ps(&source[0].x);// x2 = source->x
  x2 = _mm_sub_ps(x2, xi);              // x2 = x2 - xi
  __m128 y2 = _mm_load1_ps(&source[0].y);// y2 = source->y
  y2 = _mm_sub_ps(y2, yi);              // y2 = y2 - yi
  __m128 z2 = _mm_load1_ps(&source[0].z);// z2 = source->z
  z2 = _mm_sub_ps(z2, zi);              // z2 = z2 - zi
  __m128 mj = _mm_load1_ps(&source[0].w);// mj = source->w

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
    R2 = _mm_load_ps(target4.w);        // R2 = target4.w
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
    mj = _mm_load_ps(&source[0].x);     // mj = source->x,y,z,w
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
  _mm_store_ps(target4.x, ax);          // target4.x = ax
  _mm_store_ps(target4.y, ay);          // target4.y = ay
  _mm_store_ps(target4.z, az);          // target4.z = az
  _mm_store_ps(target4.w, phi);         // target4.w = phi
}

void P2Phost(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int base=0; base<ni; base+=4 ) {
    int nvec = std::min(ni-base,4);
    float16 target4;
    for( int i=0; i<nvec; i++ ) {
      target4.x[i] = source[base+i].x;
      target4.y[i] = source[base+i].y;
      target4.z[i] = source[base+i].z;
      target4.w[i] = eps2;
    }
    P2Psse(target4, source, nj);
    for( int i=0; i<nvec; i++ ) {
      target[base+i].x = target4.x[i];
      target[base+i].y = target4.y[i];
      target[base+i].z = target4.z[i];
      target[base+i].w = target4.w[i];
    }
  }
}

void P2P(float4 *target, float4 *source, int ni, int nj, float eps2) {
  for( int i=0; i<ni; i++ ) {
    float4 t = {0};
    for( int j=0; j<nj; j++ ) {
      float dx = source[j].x - source[i].x;
      float dy = source[j].y - source[i].y;
      float dz = source[j].z - source[i].z;
      float R2 = dx * dx + dy * dy + dz * dz + eps2;
      float invR = 1. / sqrtf(R2);
      t.w += source[j].w * invR;
      float invR3 = invR * invR * invR * source[j].w;
      t.x += dx * invR3;
      t.y += dy * invR3;
      t.z += dz * invR3;
    }
    target[i] = t;
  }
}

int main() {
// ALLOCATE
  float4 *hostS = new float4 [N]; // source
  float4 *hostT = new float4 [N]; // target
  float4 *hostR = new float4 [N]; // reference
  for( int i=0; i<N; i++ ) {
    hostS[i].x = drand48();
    hostS[i].y = drand48();
    hostS[i].z = drand48();
    hostS[i].w = drand48() / N;
  }
  std::cout << std::scientific << "N     : " << N << std::endl;

  int Events[3] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM };
  int EventSet = PAPI_NULL;
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  PAPI_start(EventSet);

  double tic = get_time();
  P2Phost(hostT,hostS,N,N,EPS2);
  double toc = get_time();

  long long values[3];
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;

  std::cout << std::scientific << "SSE   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

  tic = get_time();
  P2P(hostR,hostS,N,N,EPS2);
  toc = get_time();
  std::cout << std::scientific << "CPU   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// COMPARE RESULTS
  float pd = 0, pn = 0, fd = 0, fn = 0;
  for( int i=0; i<N; i++ ) {
    hostR[i].w -= hostS[i].w / sqrtf(EPS2);
    hostT[i].w -= hostS[i].w / sqrtf(EPS2);
    pd += (hostR[i].w - hostT[i].w) * (hostR[i].w - hostT[i].w);
    pn += hostR[i].w * hostR[i].w;
    fd += (hostR[i].x - hostT[i].x) * (hostR[i].x - hostT[i].x)
        + (hostR[i].y - hostT[i].y) * (hostR[i].y - hostT[i].y)
        + (hostR[i].z - hostT[i].z) * (hostR[i].z - hostT[i].z);
    fn += hostR[i].x * hostR[i].x + hostR[i].y * hostR[i].y + hostR[i].z * hostR[i].z;
  }
  std::cout << std::scientific << "P ERR : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  delete[] hostT;
  delete[] hostS;
  delete[] hostR;
}
