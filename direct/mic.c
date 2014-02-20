#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <papi.h>
#include <sys/time.h>
#include <immintrin.h>

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
#pragma omp parallel for
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
    p[i] = ax[i] = ay[i] = az[i] = 0;
  }
  int Events[3] = {PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM};
  int EventSet = PAPI_NULL;
  long long values[3] = {0, 0, 0};
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  printf("N      : %d\n",N);

// MIC
  PAPI_start(EventSet);
  tic = get_time();
#pragma omp parallel for private(j)
  for (i=0; i<N; i+=16) {
    __m512 pi = _mm512_setzero_ps();
    __m512 axi = _mm512_setzero_ps();
    __m512 ayi = _mm512_setzero_ps();
    __m512 azi = _mm512_setzero_ps();
    __m512 xi = _mm512_setr_ps(x[i],x[i+1],x[i+2],x[i+3],x[i+4],x[i+5],x[i+6],x[i+7],
			       x[i+8],x[i+9],x[i+10],x[i+11],x[i+12],x[i+13],x[i+14],x[i+15]);
    __m512 yi = _mm512_setr_ps(y[i],y[i+1],y[i+2],y[i+3],y[i+4],y[i+5],y[i+6],y[i+7],
			       y[i+8],y[i+9],y[i+10],y[i+11],y[i+12],y[i+13],y[i+14],y[i+15]);
    __m512 zi = _mm512_setr_ps(z[i],z[i+1],z[i+2],z[i+3],z[i+4],z[i+5],z[i+6],z[i+7],
			       z[i+8],z[i+9],z[i+10],z[i+11],z[i+12],z[i+13],z[i+14],z[i+15]);
    __m512 R2 = _mm512_set1_ps(EPS2);
    __m512 x2 = _mm512_set1_ps(x[0]);
    x2 = _mm512_sub_ps(x2, xi);
    __m512 y2 = _mm512_set1_ps(y[0]);
    y2 = _mm512_sub_ps(y2, yi);
    __m512 z2 = _mm512_set1_ps(z[0]);
    z2 = _mm512_sub_ps(z2, zi);
    __m512 mj = _mm512_set1_ps(m[0]);
    __m512 xj = x2;
    x2 = _mm512_mul_ps(x2, x2);
    R2 = _mm512_add_ps(R2, x2);
    __m512 yj = y2;
    y2 = _mm512_mul_ps(y2, y2);
    R2 = _mm512_add_ps(R2, y2);
    __m512 zj = z2;
    z2 = _mm512_mul_ps(z2, z2);
    R2 = _mm512_add_ps(R2, z2);
    __m512 invR;
    x2 = _mm512_set1_ps(x[1]);
    y2 = _mm512_set1_ps(y[1]);
    z2 = _mm512_set1_ps(z[1]);
    for (j=0; j<N-2; j++) {
      invR = _mm512_rsqrt23_ps(R2);
      R2 = _mm512_set1_ps(EPS2);
      x2 = _mm512_sub_ps(x2, xi);
      y2 = _mm512_sub_ps(y2, yi);
      z2 = _mm512_sub_ps(z2, zi);
      mj = _mm512_mul_ps(mj, invR);
      pi = _mm512_add_ps(pi, mj);
      invR = _mm512_mul_ps(invR, invR);
      invR = _mm512_mul_ps(invR, mj);
      mj = _mm512_set1_ps(m[j+1]);
      xj = _mm512_mul_ps(xj, invR);
      axi = _mm512_add_ps(axi, xj);
      xj = x2;
      x2 = _mm512_mul_ps(x2, x2);
      R2 = _mm512_add_ps(R2, x2);
      x2 = _mm512_set1_ps(x[j+2]);
      yj = _mm512_mul_ps(yj, invR);
      ayi = _mm512_add_ps(ayi, yj);
      yj = y2;
      y2 = _mm512_mul_ps(y2, y2);
      R2 = _mm512_add_ps(R2, y2);
      y2 = _mm512_set1_ps(y[j+2]);
      zj = _mm512_mul_ps(zj, invR);
      azi = _mm512_add_ps(azi, zj);
      zj = z2;
      z2 = _mm512_mul_ps(z2, z2);
      R2 = _mm512_add_ps(R2, z2);
      z2 = _mm512_set1_ps(z[j+2]);
    }
    invR = _mm512_rsqrt23_ps(R2);
    R2 = _mm512_set1_ps(EPS2);
    x2 = _mm512_sub_ps(x2, xi);
    y2 = _mm512_sub_ps(y2, yi);
    z2 = _mm512_sub_ps(z2, zi);
    mj = _mm512_mul_ps(mj, invR);
    pi = _mm512_add_ps(pi, mj);
    invR = _mm512_mul_ps(invR, invR);
    invR = _mm512_mul_ps(invR, mj);
    mj = _mm512_set1_ps(m[N-1]);
    xj = _mm512_mul_ps(xj, invR);
    axi = _mm512_add_ps(axi, xj);
    xj = x2;
    x2 = _mm512_mul_ps(x2, x2);
    R2 = _mm512_add_ps(R2, x2);
    yj = _mm512_mul_ps(yj, invR);
    ayi = _mm512_add_ps(ayi, yj);
    yj = y2;
    y2 = _mm512_mul_ps(y2, y2);
    R2 = _mm512_add_ps(R2, y2);
    zj = _mm512_mul_ps(zj, invR);
    azi = _mm512_add_ps(azi, zj);
    zj = z2;
    z2 = _mm512_mul_ps(z2, z2);
    R2 = _mm512_add_ps(R2, z2);
    invR = _mm512_rsqrt23_ps(R2);
    mj = _mm512_mul_ps(mj, invR);
    pi = _mm512_add_ps(pi, mj);
    invR = _mm512_mul_ps(invR, invR);
    invR = _mm512_mul_ps(invR, mj);
    xj = _mm512_mul_ps(xj, invR);
    axi = _mm512_add_ps(axi, xj);
    yj = _mm512_mul_ps(yj, invR);
    ayi = _mm512_add_ps(ayi, yj);
    zj = _mm512_mul_ps(zj, invR);
    azi = _mm512_add_ps(azi, zj);
    for (j=0; j<16; j++) {
      p[i+j] = ((float*)&pi)[j];
      ax[i+j] = ((float*)&axi)[j];
      ay[i+j] = ((float*)&ayi)[j];
      az[i+j] = ((float*)&azi)[j];
    }
  }
  toc = get_time();
  PAPI_stop(EventSet,values);
  printf("L2 Miss: %lld L2 Access: %lld TLB Miss: %lld\n",values[0],values[1],values[2]);
  printf("MIC    : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  for (i=0; i<3; i++) values[i] = 0;

// No MIC
  float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  PAPI_start(EventSet);
  tic = get_time();
#pragma omp parallel for private(j)
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
  PAPI_stop(EventSet,values);
  printf("L2 Miss: %lld L2 Access: %lld TLB Miss: %lld\n",values[0],values[1],values[2]);
  printf("No MIC : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
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
