#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sys/time.h>

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

const int THREADS = 512;
const int N = THREADS * 128;
const float OPS = 20. * N * N * 1e-9;
const float EPS2 = 1e-6;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

extern void P2P(float4 *target, float4 *source, int ni, int nj, float eps2);

extern void P2Pmic(float4 *target, float4 *source, int ni, int nj, float eps2);

int main() {
// ALLOCATE
  float4 *sourceHost = new float4 [N];
  float4 *targetHost = new float4 [N];
  float4 *targetMIC = new float4 [N];
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = drand48();
    sourceHost[i].y = drand48();
    sourceHost[i].z = drand48();
    sourceHost[i].w = drand48() / N;
  }
  std::cout << std::scientific << "N     : " << N << std::endl;

// Host P2P
  for (int it=0; it<2; it++) {
  double tic = get_time();
  P2P(targetHost,sourceHost,N,N,EPS2);
  double toc = get_time();
  std::cout << std::scientific << "SISD  : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;
  }

// MIC P2P
  for (int it=0; it<2; it++) {
  double tic = get_time();
  P2Pmic(targetMIC,sourceHost,N,N,EPS2);
  double toc = get_time();
  std::cout << std::scientific << "SIMD  : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;
  }

// COMPARE RESULTS
  float pd = 0, pn = 0, fd = 0, fn = 0;
  for( int i=0; i<N; i++ ) {
    targetHost[i].w -= sourceHost[i].w / sqrtf(EPS2);
    targetMIC[i].w -= sourceHost[i].w / sqrtf(EPS2);
    pd += (targetHost[i].w - targetMIC[i].w) * (targetHost[i].w - targetMIC[i].w);
    pn += targetHost[i].w * targetHost[i].w;
    fd += (targetHost[i].x - targetMIC[i].x) * (targetHost[i].x - targetMIC[i].x)
        + (targetHost[i].y - targetMIC[i].y) * (targetHost[i].y - targetMIC[i].y)
        + (targetHost[i].z - targetMIC[i].z) * (targetHost[i].z - targetMIC[i].z);
    fn += targetHost[i].x * targetHost[i].x + targetHost[i].y * targetHost[i].y + targetHost[i].z * targetHost[i].z;
  }
  std::cout << std::scientific << "P ERR : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  delete[] sourceHost;
  delete[] targetHost;
  delete[] targetMIC;
}
