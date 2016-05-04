#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
// Initialize
  int N = 1 << 16;
  int i, j;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  double tic, toc;
  float * x = (float*) malloc(N * sizeof(float));
  float * y = (float*) malloc(N * sizeof(float));
  float * z = (float*) malloc(N * sizeof(float));
  float * m = (float*) malloc(N * sizeof(float));
  float * p = (float*) malloc(N * sizeof(float));
  float * ax = (float*) malloc(N * sizeof(float));
  float * ay = (float*) malloc(N * sizeof(float));
  float * az = (float*) malloc(N * sizeof(float));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// SSE
  tic = get_time();
#pragma omp parallel for private(j)
  for (i=0; i<N; i+=4) {
    __m128 pi = _mm_setzero_ps();
    __m128 axi = _mm_setzero_ps();
    __m128 ayi = _mm_setzero_ps();
    __m128 azi = _mm_setzero_ps();
    __m128 xi = _mm_load_ps(x+i);
    __m128 yi = _mm_load_ps(y+i);
    __m128 zi = _mm_load_ps(z+i);
    __m128 R2 = _mm_set1_ps(EPS2);
    __m128 x2 = _mm_set1_ps(x[0]);
    x2 = _mm_sub_ps(x2, xi);
    __m128 y2 = _mm_set1_ps(y[0]);
    y2 = _mm_sub_ps(y2, yi);
    __m128 z2 = _mm_set1_ps(z[0]);
    z2 = _mm_sub_ps(z2, zi);
    __m128 mj = _mm_set1_ps(m[0]);
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
    x2 = _mm_set1_ps(x[1]);
    y2 = _mm_set1_ps(y[1]);
    z2 = _mm_set1_ps(z[1]);
    for (j=0; j<N-2; j++) {
      invR = _mm_rsqrt_ps(R2);
      R2 = _mm_set1_ps(EPS2);
      x2 = _mm_sub_ps(x2, xi);
      y2 = _mm_sub_ps(y2, yi);
      z2 = _mm_sub_ps(z2, zi);
      mj = _mm_mul_ps(mj, invR);
      pi = _mm_add_ps(pi, mj);
      invR = _mm_mul_ps(invR, invR);
      invR = _mm_mul_ps(invR, mj);
      mj = _mm_set1_ps(m[j+1]);
      xj = _mm_mul_ps(xj, invR);
      axi = _mm_add_ps(axi, xj);
      xj = x2;
      x2 = _mm_mul_ps(x2, x2);
      R2 = _mm_add_ps(R2, x2);
      x2 = _mm_set1_ps(x[j+2]);
      yj = _mm_mul_ps(yj, invR);
      ayi = _mm_add_ps(ayi, yj);
      yj = y2;
      y2 = _mm_mul_ps(y2, y2);
      R2 = _mm_add_ps(R2, y2);
      y2 = _mm_set1_ps(y[j+2]);
      zj = _mm_mul_ps(zj, invR);
      azi = _mm_add_ps(azi, zj);
      zj = z2;
      z2 = _mm_mul_ps(z2, z2);
      R2 = _mm_add_ps(R2, z2);
      z2 = _mm_set1_ps(z[j+2]);
    }
    invR = _mm_rsqrt_ps(R2);
    R2 = _mm_set1_ps(EPS2);
    x2 = _mm_sub_ps(x2, xi);
    y2 = _mm_sub_ps(y2, yi);
    z2 = _mm_sub_ps(z2, zi);
    mj = _mm_mul_ps(mj, invR);
    pi = _mm_add_ps(pi, mj);
    invR = _mm_mul_ps(invR, invR);
    invR = _mm_mul_ps(invR, mj);
    mj = _mm_set1_ps(m[N-1]);
    xj = _mm_mul_ps(xj, invR);
    axi = _mm_add_ps(axi, xj);
    xj = x2;
    x2 = _mm_mul_ps(x2, x2);
    R2 = _mm_add_ps(R2, x2);
    yj = _mm_mul_ps(yj, invR);
    ayi = _mm_add_ps(ayi, yj);
    yj = y2;
    y2 = _mm_mul_ps(y2, y2);
    R2 = _mm_add_ps(R2, y2);
    zj = _mm_mul_ps(zj, invR);
    azi = _mm_add_ps(azi, zj);
    zj = z2;
    z2 = _mm_mul_ps(z2, z2);
    R2 = _mm_add_ps(R2, z2);
    invR = _mm_rsqrt_ps(R2);
    mj = _mm_mul_ps(mj, invR);
    pi = _mm_add_ps(pi, mj);
    invR = _mm_mul_ps(invR, invR);
    invR = _mm_mul_ps(invR, mj);
    xj = _mm_mul_ps(xj, invR);
    axi = _mm_add_ps(axi, xj);
    yj = _mm_mul_ps(yj, invR);
    ayi = _mm_add_ps(ayi, yj);
    zj = _mm_mul_ps(zj, invR);
    azi = _mm_add_ps(azi, zj);
    _mm_store_ps(p+i, pi);
    _mm_store_ps(ax+i, axi);
    _mm_store_ps(ay+i, ayi);
    _mm_store_ps(az+i, azi);
  }
  toc = get_time();
  printf("SSE    : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

// No SSE
  float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  tic = get_time();
#pragma omp parallel for private(j) reduction(+: pdiff, pnorm, adiff, anorm)
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    for (j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float dz = z[j] - zi;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    pdiff += (p[i] - pi) * (p[i] - pi);
    pnorm += pi * pi;
    adiff += (ax[i] - axi) * (ax[i] - axi)
      + (ay[i] - ayi) * (ay[i] - ayi)
      + (az[i] - azi) * (az[i] - azi);
    anorm += axi * axi + ayi * ayi + azi * azi;    
  }
  toc = get_time();
  printf("No SIMD: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
  printf("A ERR  : %e\n",sqrt(adiff/anorm));

// DEALLOCATE
  free(x);
  free(y);
  free(z);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(az);
  return 0;
}
