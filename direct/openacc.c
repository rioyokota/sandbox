#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <openacc.h>

#define N 12800

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double) tv.tv_sec+tv.tv_usec*1e-6;
}

int main() {
  double tic, toc;
  float *x  = (float*) malloc(N * sizeof(float));
  float *y  = (float*) malloc(N * sizeof(float));
  float *z  = (float*) malloc(N * sizeof(float));
  float *m  = (float*) malloc(N * sizeof(float));
  float *p  = (float*) malloc(N * sizeof(float));
  float *fx = (float*) malloc(N * sizeof(float));
  float *fy = (float*) malloc(N * sizeof(float));
  float *fz = (float*) malloc(N * sizeof(float));
  for( int i=0; i<N; i++ ) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N     : %d\n",N);

#pragma acc data copy(x,y,z,m,p,fx,fy,fz)
  tic = get_time();
#pragma acc kernels loop gang(100) vector(128)
  for( int i=0; i<N; i++ ) {
    float pi = 0;
    float fxi = 0;
    float fyi = 0;
    float fzi = 0;
    for( int j=0; j<N; j++ ) {
      float dx = x[i] - x[j];
      float dy = y[i] - y[j];
      float dz = z[i] - z[j];
      float R2 = dx * dx + dy * dy + dz * dz + 1e-6;
      float invR = 1 / sqrtf(R2);
      pi -= m[j] * invR;
      float invR3 = invR * invR * invR * m[j];
      fxi += dx * invR3;
      fyi += dy * invR3;
      fzi += dz * invR3;
    }
    p[i] = pi;
    fx[i] = fxi;
    fy[i] = fyi;
    fz[i] = fzi;
  }
  toc = get_time();

  printf("GPU   : %lf s : %lf GFlops\n",toc-tic,(20.*N*N*1e-9)/(toc-tic));

  tic = get_time();
  float pdiff = 0, pnorm = 0, fdiff = 0, fnorm = 0;
#pragma omp parallel for reduction(pdiff,pnorm,fdiff,fnorm)
  for( int i=0; i<N; i++ ) {
    float pi = 0;
    float fxi = 0;
    float fyi = 0;
    float fzi = 0;
    for( int j=0; j<N; j++ ) {
      float dx = x[i] - x[j];
      float dy = y[i] - y[j];
      float dz = z[i] - z[j];
      float R2 = dx * dx + dy * dy + dz * dz + 1e-6;
      float invR = 1 / sqrtf(R2);
      pi -= m[j] * invR;
      float invR3 = invR * invR * invR * m[j];
      fxi += dx * invR3;
      fyi += dy * invR3;
      fzi += dz * invR3;
    }
    pdiff += ( p[i] -  pi) * ( p[i] -  pi);
    fdiff += (fx[i] - fxi) * (fx[i] - fxi);
    fdiff += (fy[i] - fyi) * (fy[i] - fyi);
    fdiff += (fz[i] - fzi) * (fz[i] - fzi);
    pnorm +=  pi * pi;
    fnorm += fxi * fxi;
    fnorm += fyi * fyi;
    fnorm += fzi * fzi;
  }
  toc = get_time();
  printf("CPU   : %lf s : %lf GFlops\n",toc-tic,(20.*N*N*1e-9)/(toc-tic));
  printf("P ERR : %f\n",sqrtf(pdiff/pnorm));
  printf("F ERR : %f\n",sqrtf(fdiff/fnorm));

  free(x);
  free(y);
  free(z);
  free(m);
  free(p);
  free(fx);
  free(fy);
  free(fz);
}
