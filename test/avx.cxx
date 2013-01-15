#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <papi.h>
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

extern void P2Psse(float4 *target, float4 *source, int ni, int nj, float eps2);

extern void P2Pavx(float4 *target, float4 *source, int ni, int nj, float eps2);

int main() {
// ALLOCATE
  float4 *sourceHost = new float4 [N];
  float4 *targetSSE = new float4 [N];
  float4 *targetAVX = new float4 [N];
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = drand48();
    sourceHost[i].y = drand48();
    sourceHost[i].z = drand48();
    sourceHost[i].w = drand48() / N;
  }
  std::cout << std::scientific << "N     : " << N << std::endl;

// SSE P2P
  int Events[3] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM };
  int EventSet = PAPI_NULL;
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  PAPI_start(EventSet);

  double tic = get_time();
  P2Psse(targetSSE,sourceHost,N,N,EPS2);
  double toc = get_time();

  long long values[3];
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;

  std::cout << std::scientific << "SSE   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// AVX P2P
  PAPI_start(EventSet);

  tic = get_time();
  P2Pavx(targetAVX,sourceHost,N,N,EPS2);
  toc = get_time();

  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;

  std::cout << std::scientific << "AVX   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// COMPARE RESULTS
  float pd = 0, pn = 0, fd = 0, fn = 0;
  for( int i=0; i<N; i++ ) {
    targetSSE[i].w -= sourceHost[i].w / sqrtf(EPS2);
    targetAVX[i].w -= sourceHost[i].w / sqrtf(EPS2);
    pd += (targetSSE[i].w - targetAVX[i].w) * (targetSSE[i].w - targetAVX[i].w);
    pn += targetSSE[i].w * targetSSE[i].w;
    fd += (targetSSE[i].x - targetAVX[i].x) * (targetSSE[i].x - targetAVX[i].x)
        + (targetSSE[i].y - targetAVX[i].y) * (targetSSE[i].y - targetAVX[i].y)
        + (targetSSE[i].z - targetAVX[i].z) * (targetSSE[i].z - targetAVX[i].z);
    fn += targetSSE[i].x * targetSSE[i].x + targetSSE[i].y * targetSSE[i].y + targetSSE[i].z * targetSSE[i].z;
  }
  std::cout << std::scientific << "P ERR : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  delete[] sourceHost;
  delete[] targetSSE;
  delete[] targetAVX;
}
