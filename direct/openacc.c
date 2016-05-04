#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <openacc.h>

#define THREADS 512

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
  float x[N];
  float y[N];
  float z[N];
  float m[N];
  float p[N];
  float ax[N];
  float ay[N];
  float az[N];
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);

// OpenACC
#pragma acc data copy(x,y,z,m,p,ax,ay,az)
  tic = get_time();
#pragma acc kernels loop gang(N/THREADS) vector(THREADS) private(i, j)
  for (i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    for (j=0; j<N; j++) {
      float dx = x[j] - x[i];
      float dy = y[j] - y[i];
      float dz = z[j] - z[i];
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = 1 / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
  toc = get_time();
  printf("ACC    : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

// No OpenACC
  float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  tic = get_time();
#pragma omp parallel for private(j) reduction(pdiff,pnorm,adiff,anorm)
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
  printf("No ACC : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
  printf("A ERR  : %e\n",sqrt(adiff/anorm));
  return 0;
}
