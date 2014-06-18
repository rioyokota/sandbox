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
  int NT = 32;
  int NALIGN = 64;
  int it, i, j;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  double tic, toc;
  float * x = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * y = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * z = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * m = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * p = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * ax = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * ay = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * az = (float*) _mm_malloc(N * sizeof(float), NALIGN);
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
  float flop1[NT], flop2[NT];
  for (it=0; it<NT; it++) {
    printf("it     : %d\n", it);
    tic = get_time();
#pragma omp parallel for private(j)
    for (i=0; i<N; i+=16) {
      __m512 pi = _mm512_setzero_ps();
      __m512 axi = _mm512_setzero_ps();
      __m512 ayi = _mm512_setzero_ps();
      __m512 azi = _mm512_setzero_ps();
      __m512 xi = _mm512_load_ps(x+i);
      __m512 yi = _mm512_load_ps(y+i);
      __m512 zi = _mm512_load_ps(z+i);
      for (j=0; j<N; j++) {
	__m512 xj = _mm512_set1_ps(x[j]);
	xj = _mm512_sub_ps(xj, xi);
	__m512 yj = _mm512_set1_ps(y[j]);
	yj = _mm512_sub_ps(yj, yi);
	__m512 zj = _mm512_set1_ps(z[j]);
	zj = _mm512_sub_ps(zj, zi);
	__m512 R2 = _mm512_set1_ps(EPS2);
	R2 = _mm512_fmadd_ps(xj, xj, R2);
	R2 = _mm512_fmadd_ps(yj, yj, R2);
	R2 = _mm512_fmadd_ps(zj, zj, R2);
	__m512 mj = _mm512_set1_ps(m[j]);
	__m512 invR = _mm512_rsqrt23_ps(R2);
	mj = _mm512_mul_ps(mj, invR);
	pi = _mm512_add_ps(pi, mj);
	invR = _mm512_mul_ps(invR, invR);
	invR = _mm512_mul_ps(invR, mj);
	axi = _mm512_fmadd_ps(xj, invR, axi);
	ayi = _mm512_fmadd_ps(yj, invR, ayi);
	azi = _mm512_fmadd_ps(zj, invR, azi);
      }
      _mm512_store_ps(p+i, pi);
      _mm512_store_ps(ax+i, axi);
      _mm512_store_ps(ay+i, ayi);
      _mm512_store_ps(az+i, azi);
    }
    toc = get_time();
    PAPI_stop(EventSet,values);
    printf("L2 Miss: %lld L2 Access: %lld TLB Miss: %lld\n",values[0],values[1],values[2]);
    printf("MIC    : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
    flop1[it] = OPS/(toc-tic);
    for (i=0; i<3; i++) values[i] = 0;

    // No MIC
    float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
    PAPI_start(EventSet);
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
    PAPI_stop(EventSet,values);
    printf("L2 Miss: %lld L2 Access: %lld TLB Miss: %lld\n",values[0],values[1],values[2]);
    printf("No MIC : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
    flop2[it] = OPS/(toc-tic);
    printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
    printf("A ERR  : %e\n",sqrt(adiff/anorm));
  }

  float fave1 = 0, fave2 = 0;
  for (it=1; it<NT; it++) {
    fave1 += flop1[it];
    fave2 += flop2[it];
  }
  fave1 /= NT - 1;
  fave2 /= NT - 1;
  float fstd1 = 0, fstd2 = 0;
  for (it=1; it<NT; it++) {
    fstd1 += (flop1[it] - fave1) * (flop1[it] - fave1);
    fstd2 += (flop2[it] - fave2) * (flop2[it] - fave2);
  }
  printf("MIC    : %f +- %f GFlops\n", fave1, sqrtf(fstd1/(NT-1)));
  printf("No MIC : %f +- %f GFlops\n", fave2, sqrtf(fstd2/(NT-1)));

  // DEALLOCATE
  _mm_free(x);
  _mm_free(y);
  _mm_free(z);
  _mm_free(m);
  _mm_free(p);
  _mm_free(ax);
  _mm_free(ay);
  _mm_free(az);
  return 0;
}
