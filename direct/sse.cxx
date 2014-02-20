#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <papi.h>
#include <sys/time.h>
#include <xmmintrin.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
// Initialize
  const int N = 1 << 16;
  const float OPS = 20. * N * N * 1e-9;
  const float EPS2 = 1e-6;
  float * x = (float*) malloc(N * sizeof(float));
  float * y = (float*) malloc(N * sizeof(float));
  float * z = (float*) malloc(N * sizeof(float));
  float * m = (float*) malloc(N * sizeof(float));
  float * p = (float*) malloc(N * sizeof(float));
  float * ax = (float*) malloc(N * sizeof(float));
  float * ay = (float*) malloc(N * sizeof(float));
  float * az = (float*) malloc(N * sizeof(float));
  float4 *source = new float4 [N];
  float4 *target = new float4 [N];
  for( int i=0; i<N; i++ ) {
    x[i] = source[i].x = drand48();
    y[i] = source[i].y = drand48();
    z[i] = source[i].z = drand48();
    m[i] = source[i].w = drand48() / N;
  }
  double tic, toc;
  int Events[3] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM };
  int EventSet = PAPI_NULL;
  long long values[3];
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  std::cout << std::scientific << "N      : " << N << std::endl;

// SSE
  PAPI_start(EventSet);
  tic = get_time();
#pragma omp parallel for
  for( int i=0; i<N; i+=4 ) {
    __m128 ax = _mm_setzero_ps();
    __m128 ay = _mm_setzero_ps();
    __m128 az = _mm_setzero_ps();
    __m128 phi = _mm_setzero_ps();
    __m128 xi = _mm_setr_ps(source[i].x,source[i+1].x,source[i+2].x,source[i+3].x);
    __m128 yi = _mm_setr_ps(source[i].y,source[i+1].y,source[i+2].y,source[i+3].y);
    __m128 zi = _mm_setr_ps(source[i].z,source[i+1].z,source[i+2].z,source[i+3].z);
    __m128 R2 = _mm_load1_ps(&EPS2);
    __m128 x2 = _mm_load1_ps(&source[0].x);
    x2 = _mm_sub_ps(x2, xi);
    __m128 y2 = _mm_load1_ps(&source[0].y);
    y2 = _mm_sub_ps(y2, yi);
    __m128 z2 = _mm_load1_ps(&source[0].z);
    z2 = _mm_sub_ps(z2, zi);
    __m128 mj = _mm_load1_ps(&source[0].w);
    __m128 xj = x2;
    x2 = _mm_mul_ps(x2, x2);
    R2 = _mm_add_ps(R2, x2);
    __m128 yj = y2;
    y2 = _mm_mul_ps(y2, y2);
    R2 = _mm_add_ps(R2, y2);
    __m128 zj = z2;
    z2 = _mm_mul_ps(z2, z2);
    R2 = _mm_add_ps(R2, z2);
    __m128 invR;
    x2 = _mm_load1_ps(&x[1]);
    y2 = _mm_load1_ps(&y[1]);
    z2 = _mm_load1_ps(&z[1]);
    for( int j=0; j<N-2; j++ ) {
      invR = _mm_rsqrt_ps(R2);
      R2 = _mm_load1_ps(&EPS2);
      x2 = _mm_sub_ps(x2, xi);
      y2 = _mm_sub_ps(y2, yi);
      z2 = _mm_sub_ps(z2, zi);
      mj = _mm_mul_ps(mj, invR);
      phi = _mm_add_ps(phi, mj);
      invR = _mm_mul_ps(invR, invR);
      invR = _mm_mul_ps(invR, mj);
      mj = _mm_load1_ps(&m[j+1]);
      xj = _mm_mul_ps(xj, invR);
      ax = _mm_add_ps(ax, xj);
      xj = x2;
      x2 = _mm_mul_ps(x2, x2);
      R2 = _mm_add_ps(R2, x2);
      x2 = _mm_load1_ps(&x[j+2]);
      yj = _mm_mul_ps(yj, invR);
      ay = _mm_add_ps(ay, yj);
      yj = y2;
      y2 = _mm_mul_ps(y2, y2);
      R2 = _mm_add_ps(R2, y2);
      y2 = _mm_load1_ps(&y[j+2]);
      zj = _mm_mul_ps(zj, invR);
      az = _mm_add_ps(az, zj);
      zj = z2;
      z2 = _mm_mul_ps(z2, z2);
      R2 = _mm_add_ps(R2, z2);
      z2 = _mm_load1_ps(&z[j+2]);
    }
    invR = _mm_rsqrt_ps(R2);
    R2 = _mm_load1_ps(&EPS2);
    x2 = _mm_sub_ps(x2, xi);
    y2 = _mm_sub_ps(y2, yi);
    z2 = _mm_sub_ps(z2, zi);
    mj = _mm_mul_ps(mj, invR);
    phi = _mm_add_ps(phi, mj);
    invR = _mm_mul_ps(invR, invR);
    invR = _mm_mul_ps(invR, mj);
    mj = _mm_load1_ps(&m[N-1]);
    xj = _mm_mul_ps(xj, invR);
    ax = _mm_add_ps(ax, xj);
    xj = x2;
    x2 = _mm_mul_ps(x2, x2);
    R2 = _mm_add_ps(R2, x2);
    yj = _mm_mul_ps(yj, invR);
    ay = _mm_add_ps(ay, yj);
    yj = y2;
    y2 = _mm_mul_ps(y2, y2);
    R2 = _mm_add_ps(R2, y2);
    zj = _mm_mul_ps(zj, invR);
    az = _mm_add_ps(az, zj);
    zj = z2;
    z2 = _mm_mul_ps(z2, z2);
    R2 = _mm_add_ps(R2, z2);
    invR = _mm_rsqrt_ps(R2);
    mj = _mm_mul_ps(mj, invR);
    phi = _mm_add_ps(phi, mj);
    invR = _mm_mul_ps(invR, invR);
    invR = _mm_mul_ps(invR, mj);
    xj = _mm_mul_ps(xj, invR);
    ax = _mm_add_ps(ax, xj);
    yj = _mm_mul_ps(yj, invR);
    ay = _mm_add_ps(ay, yj);
    zj = _mm_mul_ps(zj, invR);
    az = _mm_add_ps(az, zj);
    for( int k=0; k<4; k++ ) {
      target[i+k].x = ((float*)&ax)[k];
      target[i+k].y = ((float*)&ay)[k];
      target[i+k].z = ((float*)&az)[k];
      target[i+k].w = ((float*)&phi)[k];
    }
  }
  toc = get_time();
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;
  std::cout << std::scientific << "SSE    : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;
  for (int i=0; i<3; i++) values[i] = 0;

// No SSE
  float pd = 0, pn = 0, fd = 0, fn = 0;
  PAPI_start(EventSet);
  tic = get_time();
#pragma omp parallel for
  for (int i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    float xi = source[i].x;
    float yi = source[i].y;
    float zi = source[i].z;
    for (int j=0; j<N; j++) {
      float dx = source[j].x - xi;
      float dy = source[j].y - yi;
      float dz = source[j].z - zi;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = source[j].w * invR * invR * invR;
      pi += source[j].w * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    pd += (target[i].w - pi) * (target[i].w - pi);
    pn += pi * pi;
    fd += (target[i].x - axi) * (target[i].x - axi)
        + (target[i].y - ayi) * (target[i].y - ayi)
        + (target[i].z - azi) * (target[i].z - azi);
    fn += axi * axi + ayi * ayi + azi * azi;    
  }
  toc = get_time();
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;
  std::cout << std::scientific << "No SSE : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;
  std::cout << std::scientific << "P ERR  : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR  : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  delete[] source;
  delete[] target;
  free(x);
  free(y);
  free(z);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(az);
}
