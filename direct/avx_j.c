#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

inline void rsqrt_newton(__m256& rinv, const __m256& r2, const float& nwtn_const){
  rinv = _mm256_mul_ps(rinv,_mm256_sub_ps(_mm256_set_ps(nwtn_const, nwtn_const, nwtn_const, nwtn_const,
                                                        nwtn_const, nwtn_const, nwtn_const, nwtn_const),
                                          _mm256_mul_ps(r2,_mm256_mul_ps(rinv,rinv))));
}

int main(int argc, char** argv) {
// Initialize
  long long N = 1 << atoi(argv[1]);
  int NALIGN = 32;
  int i, j, err = 1;
  float OPS = 20. * N * N * 1e-9 * err;
  float EPS2 = 1e-6;
  double tic, toc;
  float * x_i = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * y_i = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * z_i = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * x_j = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * y_j= (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * z_j= (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * m = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * p = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * ax = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * ay = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  float * az = (float*) _mm_malloc(N * sizeof(float), NALIGN);
  double * pd = (double*) _mm_malloc(N * sizeof(double), NALIGN);
  double * axd = (double*) _mm_malloc(N * sizeof(double), NALIGN);
  double * ayd = (double*) _mm_malloc(N * sizeof(double), NALIGN);
  double * azd = (double*) _mm_malloc(N * sizeof(double), NALIGN);

  FILE *file;
  //#define XWRITE
#ifdef XWRITE
  file = fopen("xj.dat","w");
#else
  file = fopen("xj.dat","r");
#endif
  for (i=0; i<N; i++) {
#ifdef XWRITE
    x_i[i] = drand48();
    y_i[i] = drand48();
    z_i[i] = drand48();
    x_j[i] = drand48()+5;
    y_j[i] = drand48()+5;
    z_j[i] = drand48()+5;
    m[i] = drand48() / N;
    fprintf(file,"%16.14f ", x_i[i]);
    fprintf(file,"%16.14f ", y_i[i]);
    fprintf(file,"%16.14f ", z_i[i]);
    fprintf(file,"%16.14f ", x_j[i]);
    fprintf(file,"%16.14f ", y_j[i]);
    fprintf(file,"%16.14f ", z_j[i]);
    fprintf(file,"%16.14f ", m[i]);
#else
    err = fscanf(file,"%f ", &x_i[i]);
    err = fscanf(file,"%f ", &y_i[i]);
    err = fscanf(file,"%f ", &z_i[i]);
    err = fscanf(file,"%f ", &x_j[i]);
    err = fscanf(file,"%f ", &y_j[i]);
    err = fscanf(file,"%f ", &z_j[i]);
    err = fscanf(file,"%f ", &m[i]);
#endif
    //printf("%16.14f %16.14f %16.14f %16.14f\n",x_i[i],y_i[i],z_i[i],m[i]);
  }
  fclose(file);
  printf("N      : %lld\n",N);

#ifndef XWRITE
// AVX
  tic = get_time();
#pragma omp parallel for private(j)
  for (i=0; i<N; i+=8) {
    __m256 pi = _mm256_setzero_ps();
    __m256 axi = _mm256_setzero_ps();
    __m256 ayi = _mm256_setzero_ps();
    __m256 azi = _mm256_setzero_ps();
    __m256 xi = _mm256_load_ps(x_i+i);
    __m256 yi = _mm256_load_ps(y_i+i);
    __m256 zi = _mm256_load_ps(z_i+i);
    __m256 R2 = _mm256_set1_ps(EPS2);
    __m256 x2 = _mm256_set1_ps(x_j[0]);
    x2 = _mm256_sub_ps(x2, xi);
    __m256 y2 = _mm256_set1_ps(y_j[0]);
    y2 = _mm256_sub_ps(y2, yi);
    __m256 z2 = _mm256_set1_ps(z_j[0]);
    z2 = _mm256_sub_ps(z2, zi);
    __m256 mj = _mm256_set1_ps(m[0]);
    __m256 xj = x2;
    x2 = _mm256_mul_ps(x2, x2);
    R2 = _mm256_add_ps(R2, x2);
    __m256 yj = y2;
    y2 = _mm256_mul_ps(y2, y2);
    R2 = _mm256_add_ps(R2, y2);
    __m256 zj = z2;
    z2 = _mm256_mul_ps(z2, z2);
    R2 = _mm256_add_ps(R2, z2);
    __m256 invR;
    x2 = _mm256_set1_ps(x_j[1]);
    y2 = _mm256_set1_ps(y_j[1]);
    z2 = _mm256_set1_ps(z_j[1]);
    for (j=0; j<N-2; j++) {
      invR = _mm256_rsqrt_ps(R2);
      rsqrt_newton(invR, R2, float(3));
      rsqrt_newton(invR, R2, float(12));
      R2 = _mm256_set1_ps(EPS2);
      x2 = _mm256_sub_ps(x2, xi);
      y2 = _mm256_sub_ps(y2, yi);
      z2 = _mm256_sub_ps(z2, zi);
      mj = _mm256_mul_ps(mj, invR);
      pi = _mm256_add_ps(pi, mj);
      invR = _mm256_mul_ps(invR, invR);
      invR = _mm256_mul_ps(invR, mj);
      mj = _mm256_set1_ps(m[j+1]);
      xj = _mm256_mul_ps(xj, invR);
      axi = _mm256_add_ps(axi, xj);
      xj = x2;
      x2 = _mm256_mul_ps(x2, x2);
      R2 = _mm256_add_ps(R2, x2);
      x2 = _mm256_set1_ps(x_j[j+2]);
      yj = _mm256_mul_ps(yj, invR);
      ayi = _mm256_add_ps(ayi, yj);
      yj = y2;
      y2 = _mm256_mul_ps(y2, y2);
      R2 = _mm256_add_ps(R2, y2);
      y2 = _mm256_set1_ps(y_j[j+2]);
      zj = _mm256_mul_ps(zj, invR);
      azi = _mm256_add_ps(azi, zj);
      zj = z2;
      z2 = _mm256_mul_ps(z2, z2);
      R2 = _mm256_add_ps(R2, z2);
      z2 = _mm256_set1_ps(z_j[j+2]);
    }
    invR = _mm256_rsqrt_ps(R2);
    rsqrt_newton(invR, R2, float(3));
    rsqrt_newton(invR, R2, float(12));
    R2 = _mm256_set1_ps(EPS2);
    x2 = _mm256_sub_ps(x2, xi);
    y2 = _mm256_sub_ps(y2, yi);
    z2 = _mm256_sub_ps(z2, zi);
    mj = _mm256_mul_ps(mj, invR);
    pi = _mm256_add_ps(pi, mj);
    invR = _mm256_mul_ps(invR, invR);
    invR = _mm256_mul_ps(invR, mj);
    mj = _mm256_set1_ps(m[N-1]);
    xj = _mm256_mul_ps(xj, invR);
    axi = _mm256_add_ps(axi, xj);
    xj = x2;
    x2 = _mm256_mul_ps(x2, x2);
    R2 = _mm256_add_ps(R2, x2);
    yj = _mm256_mul_ps(yj, invR);
    ayi = _mm256_add_ps(ayi, yj);
    yj = y2;
    y2 = _mm256_mul_ps(y2, y2);
    R2 = _mm256_add_ps(R2, y2);
    zj = _mm256_mul_ps(zj, invR);
    azi = _mm256_add_ps(azi, zj);
    zj = z2;
    z2 = _mm256_mul_ps(z2, z2);
    R2 = _mm256_add_ps(R2, z2);
    invR = _mm256_rsqrt_ps(R2);
    rsqrt_newton(invR, R2, float(3));
    rsqrt_newton(invR, R2, float(12));
    mj = _mm256_mul_ps(mj, invR);
    pi = _mm256_add_ps(pi, mj);
    invR = _mm256_mul_ps(invR, invR);
    invR = _mm256_mul_ps(invR, mj);
    xj = _mm256_mul_ps(xj, invR);
    axi = _mm256_add_ps(axi, xj);
    yj = _mm256_mul_ps(yj, invR);
    ayi = _mm256_add_ps(ayi, yj);
    zj = _mm256_mul_ps(zj, invR);
    azi = _mm256_add_ps(azi, zj);
    _mm256_store_ps(p+i, pi);
    _mm256_store_ps(ax+i, axi);
    _mm256_store_ps(ay+i, ayi);
    _mm256_store_ps(az+i, azi);
  }
  toc = get_time();
  printf("AVX    : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

// No AVX
  double pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  char filename[128];
  strcpy(filename,"xj");
  strcat(filename,argv[1]);
  strcat(filename,".dat");
#define PWRITE
#ifdef PWRITE
  file = fopen(filename,"w");
  tic = get_time();
#pragma omp parallel for private(j) reduction(+: pdiff, pnorm, adiff, anorm)
  for (i=0; i<N; i++) {
    double pi = 0;
    double axi = 0;
    double ayi = 0;
    double azi = 0;
    double xi = x_i[i];
    double yi = y_i[i];
    double zi = z_i[i];
    for (j=0; j<N; j++) {
      double dx = x_j[j] - xi;
      double dy = y_j[j] - yi;
      double dz = z_j[j] - zi;
      double R2 = dx * dx + dy * dy + dz * dz + EPS2;
      double invR = 16.0f / sqrtf(R2);
      double invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    pd[i] = pi;
    axd[i] = axi;
    ayd[i] = ayi;
    azd[i] = azi;
  }
#else
  file = fopen(filename,"r");
#endif
  for (i=0; i<N; i++) {
#ifdef PWRITE
    fprintf(file,"%le ", pd[i]);
    fprintf(file,"%le ", axd[i]);
    fprintf(file,"%le ", ayd[i]);
    fprintf(file,"%le ", azd[i]);
#else
    err = fscanf(file,"%le ", &pd[i]);
    err = fscanf(file,"%le ", &axd[i]);
    err = fscanf(file,"%le ", &ayd[i]);
    err = fscanf(file,"%le ", &azd[i]);
#endif
    //printf("%le %le %le %le\n",pd[i],axd[i],ayd[i],azd[i]);
    pdiff += (p[i] - pd[i]) * (p[i] - pd[i]);
    pnorm += pd[i] * pd[i];
    adiff += (ax[i] - axd[i]) * (ax[i] - axd[i])
      + (ay[i] - ayd[i]) * (ay[i] - ayd[i])
      + (az[i] - azd[i]) * (az[i] - azd[i]);
    anorm += axd[i] * axd[i] + ayd[i] * ayd[i] + azd[i] * azd[i];
  }
  fclose(file);
  toc = get_time();
  printf("No SIMD: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
  printf("A ERR  : %e\n",sqrt(adiff/anorm));
#endif

// DEALLOCATE
  _mm_free(x_i);
  _mm_free(y_i);
  _mm_free(z_i);
  _mm_free(x_j);
  _mm_free(y_j);
  _mm_free(z_j);
  _mm_free(m);
  _mm_free(p);
  _mm_free(ax);
  _mm_free(ay);
  _mm_free(az);
  _mm_free(pd);
  _mm_free(axd);
  _mm_free(ayd);
  _mm_free(azd);
  return 0;
}
