// Step 2. AVX

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  int N = 1 << 12;
  int NALIGN = 32;
  int i, j;
  float EPS2 = 1e-6;
  float * x = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * y = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * u = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * q = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
  double tic = get_time();
#pragma omp parallel for private(j)
  for (i=0; i<N; i+=8) {
    __m256 xi = _mm256_load_ps(x+i);
    __m256 yi = _mm256_load_ps(y+i);
    __m256 ui = _mm256_setzero_ps();
    for (j=0; j<N; j++) {
      __m256 dx = _mm256_set1_ps(x[j]);
      dx = _mm256_sub_ps(dx, xi);
      __m256 dy = _mm256_set1_ps(y[j]);
      dy = _mm256_sub_ps(dy, yi);
      __m256 qj = _mm256_set1_ps(q[j]);
      __m256 R2 = _mm256_set1_ps(EPS2);
      dx = _mm256_mul_ps(dx, dx);
      R2 = _mm256_add_ps(R2, dx);
      dy = _mm256_mul_ps(dy, dy);
      R2 = _mm256_add_ps(R2, dy);
      __m256 invR = _mm256_rsqrt_ps(R2);
      qj = _mm256_mul_ps(qj,invR);
      ui = _mm256_add_ps(ui, qj);
    }
    _mm256_store_ps(u+i, ui);
  }
  double toc = get_time();
  printf("%lf\n",toc-tic);
  int u2[N];
#pragma omp parallel private(j)
  for (i=0; i<N; i++) {
    double ui = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx * dx + dy * dy + EPS2);
      ui += q[j] / r;
    }
    u2[i] = ui;
  }
  tic = get_time();
  printf("%lf\n",tic-toc);
  _mm_free(x);
  _mm_free(y);
  _mm_free(u);
  _mm_free(q);
}
