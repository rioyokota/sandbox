#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>


double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  const int N = 1 << 16;
  const int nalign = 64;
  float * x = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * y = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * z = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * m = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * p = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * ax = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * ay = (float*) _mm_malloc(sizeof(float) * N, nalign);
  float * az = (float*) _mm_malloc(sizeof(float) * N, nalign);
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = 1.0f / N;
  }
  double tic,toc;
  for (int it=0; it<2; it++) {
    tic = get_time();
#pragma omp parallel for
    for (int i=0; i<N; i++) {
      float pi = 0;
      float axi = 0;
      float ayi = 0;
      float azi = 0;
#pragma vector aligned
      for (int j=0; j<N; j++) {
	float dx = x[i] - x[j];
	float dy = y[i] - y[j];
	float dz = z[i] - z[j];
	float r2 = dx * dx + dy * dy + dz * dz + 1e-6;
	float invR = 1.0f / sqrtf(r2);
	float invR3 = invR * invR * invR * m[j];
	pi -= m[j] * invR;
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
  }
  printf("N      : %d\n",N);
  printf("Time   : %lf s\n",toc-tic);
  printf("GFlops : %d\n",int(2e-8*N*N/(toc-tic)));
  _mm_free(x);
  _mm_free(y);
  _mm_free(z);
  _mm_free(m);
  _mm_free(p);
  _mm_free(ax);
  _mm_free(ay);
  _mm_free(az);
}
