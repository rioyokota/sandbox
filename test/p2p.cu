#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <papi.h>
#include <sys/time.h>

const int THREADS = 128;
const int N = THREADS * 512;
const int Nj = N / 256;
const float OPS = 20. * N * Nj * 1e-9;
const float EPS2 = 1e-6;

double get_time() {
  struct timeval tv;
  cudaThreadSynchronize();
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

void P2Phost(float4 *target, float4 *source, int ni, int nj, float eps2);

__global__ void P2Pdevice(float4 *target, float4 *source) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  float4 t = {0,0,0,0};
  __shared__ float4 s[THREADS];
  for ( int jb=0; jb<Nj/THREADS; jb++ ) {
    __syncthreads();
    s[threadIdx.x] = source[jb*THREADS+threadIdx.x];
    __syncthreads();
    for( int j=0; j<THREADS; j++ ) {
      float dx = s[j].x - source[i].x;
      float dy = s[j].y - source[i].y;
      float dz = s[j].z - source[i].z;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = rsqrtf(R2);
      t.w += s[j].w * invR;
      float invR3 = invR * invR * invR * s[j].w;
      t.x += dx * invR3;
      t.y += dy * invR3;
      t.z += dz * invR3;
    }
  }
  target[i] = t;
}

int main() {
// ALLOCATE
  float4 *hostS = new float4 [N];
  float4 *hostT = new float4 [N];
  float4 *hostR = new float4 [N];
  for( int i=0; i<N; i++ ) {
    hostS[i].x = drand48();
    hostS[i].y = drand48();
    hostS[i].z = drand48();
    hostS[i].w = drand48() / N;
  }
  float4 *deviceS, *deviceT;
  cudaMalloc((void**)&deviceS,N*sizeof(float4));
  cudaMalloc((void**)&deviceT,N*sizeof(float4));
  std::cout << std::scientific << "N     : " << N << std::endl;

// CPU P2P
  int Events[3] = { PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM };
  int EventSet = PAPI_NULL;
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  PAPI_start(EventSet);

  double tic = get_time();
  P2Phost(hostR,hostS,N,Nj,EPS2);
  double toc = get_time();

  long long values[3];
  PAPI_stop(EventSet,values);
  std::cout << "L2 Miss: " << values[0]
            << " L2 Access: " << values[1]
            << " TLB Miss: " << values[2] << std::endl;

  std::cout << std::scientific << "CPU   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;

// GPU P2P
  cudaMemcpy(deviceS,hostS,N*sizeof(float4),cudaMemcpyHostToDevice);
  tic = get_time();
  P2Pdevice<<<N/THREADS,THREADS>>>(deviceT,deviceS);
  toc = get_time();
  cudaMemcpy(hostT,deviceT,N*sizeof(float4),cudaMemcpyDeviceToHost);
  std::cout << std::scientific << "GPU   : " << toc-tic << " s : " << OPS / (toc-tic) << " GFlops" << std::endl;
  cudaDeviceReset();

// COMPARE RESULTS
  float pd = 0, pn = 0, fd = 0, fn = 0;
  for( int i=0; i<N; i++ ) {
    hostR[i].w -= hostS[i].w / sqrtf(EPS2);
    hostT[i].w -= hostS[i].w / sqrtf(EPS2);
    pd += (hostR[i].w - hostT[i].w) * (hostR[i].w - hostT[i].w);
    pn += hostR[i].w * hostR[i].w;
    fd += (hostR[i].x - hostT[i].x) * (hostR[i].x - hostT[i].x)
        + (hostR[i].y - hostT[i].y) * (hostR[i].y - hostT[i].y)
        + (hostR[i].z - hostT[i].z) * (hostR[i].z - hostT[i].z);
    fn += hostR[i].x * hostR[i].x + hostR[i].y * hostR[i].y + hostR[i].z * hostR[i].z;
  }
  std::cout << std::scientific << "P ERR : " << sqrtf(pd/pn) << std::endl;
  std::cout << std::scientific << "F ERR : " << sqrtf(fd/fn) << std::endl;

// DEALLOCATE
  cudaFree(deviceS);
  cudaFree(deviceT);
  delete[] hostT;
  delete[] hostS;
  delete[] hostR;
}
