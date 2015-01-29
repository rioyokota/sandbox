#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <papi.h>
#include <sys/time.h>
#include <xmmintrin.h>

#define THREADS 512
typedef float real_t;

double get_time() {
  struct timeval tv;
  cudaThreadSynchronize();
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

__global__ void GPUkernel(int N, real_t * x, real_t * y, real_t * z, real_t * m,
			  real_t * p, real_t * ax, real_t * ay, real_t * az, real_t eps2) {
  int i = blockIdx.x * THREADS + threadIdx.x;
  real_t pi = 0;
  real_t axi = 0;
  real_t ayi = 0;
  real_t azi = 0;
  real_t xi = x[i];
  real_t yi = y[i];
  real_t zi = z[i];
  __shared__ real_t xj[THREADS], yj[THREADS], zj[THREADS], mj[THREADS];
  for ( int jb=0; jb<N/THREADS; jb++ ) {
    __syncthreads();
    xj[threadIdx.x] = x[jb*THREADS+threadIdx.x];
    yj[threadIdx.x] = y[jb*THREADS+threadIdx.x];
    zj[threadIdx.x] = z[jb*THREADS+threadIdx.x];
    mj[threadIdx.x] = m[jb*THREADS+threadIdx.x];
    __syncthreads();
#pragma unroll
    for( int j=0; j<THREADS; j++ ) {
      real_t dx = xj[j] - xi;
      real_t dy = yj[j] - yi;
      real_t dz = zj[j] - zi;
      real_t R2 = dx * dx + dy * dy + dz * dz + eps2;
      real_t invR = rsqrtf(R2);
      pi += mj[j] * invR;
      real_t invR3 = mj[j] * invR * invR * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
  }
  p[i] = pi;
  ax[i] = axi;
  ay[i] = ayi;
  az[i] = azi;
}

int main() {
// Initialize
  int N = 1 << 16;
  int i, j;
  real_t OPS = 20. * N * N * 1e-9;
  real_t EPS2 = 1e-6;
  double tic, toc;
  real_t * x = (real_t*) malloc(N * sizeof(real_t));
  real_t * y = (real_t*) malloc(N * sizeof(real_t));
  real_t * z = (real_t*) malloc(N * sizeof(real_t));
  real_t * m = (real_t*) malloc(N * sizeof(real_t));
  real_t * p = (real_t*) malloc(N * sizeof(real_t));
  real_t * ax = (real_t*) malloc(N * sizeof(real_t));
  real_t * ay = (real_t*) malloc(N * sizeof(real_t));
  real_t * az = (real_t*) malloc(N * sizeof(real_t));
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  int Events[3] = {PAPI_L2_DCM, PAPI_L2_DCA, PAPI_TLB_DM};
  int EventSet = PAPI_NULL;
  long long values[3] = {0, 0, 0};
  PAPI_library_init(PAPI_VER_CURRENT);
  PAPI_create_eventset(&EventSet);
  PAPI_add_events(EventSet, Events, 3);
  printf("N      : %d\n",N);

// CUDA
  tic = get_time();
  real_t *x_d, *y_d, *z_d, *m_d, *p_d, *ax_d, *ay_d, *az_d;
  cudaMalloc((void**)&x_d, N * sizeof(real_t));
  cudaMalloc((void**)&y_d, N * sizeof(real_t));
  cudaMalloc((void**)&z_d, N * sizeof(real_t));
  cudaMalloc((void**)&m_d, N * sizeof(real_t));
  cudaMalloc((void**)&p_d, N * sizeof(real_t));
  cudaMalloc((void**)&ax_d, N * sizeof(real_t));
  cudaMalloc((void**)&ay_d, N * sizeof(real_t));
  cudaMalloc((void**)&az_d, N * sizeof(real_t));
  toc = get_time();
  //printf("malloc : %e s\n",toc-tic);
  tic = get_time();
  cudaMemcpy(x_d, x, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(m_d, m, N * sizeof(real_t), cudaMemcpyHostToDevice);
  toc = get_time();
  //printf("memcpy : %e s\n",toc-tic);
  PAPI_start(EventSet);
  tic = get_time();
  GPUkernel<<<N/THREADS,THREADS>>>(N, x_d, y_d, z_d, m_d, p_d, ax_d, ay_d, az_d, EPS2);
  toc = get_time();
  PAPI_stop(EventSet,values);
  printf("L2 Miss: %lld L2 Access: %lld TLB Miss: %lld\n",values[0],values[1],values[2]);
  printf("CUDA   : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  tic = get_time();
  cudaMemcpy(p, p_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ax, ax_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ay, ay_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(az, az_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  toc = get_time();
  //printf("memcpy : %e s\n",toc-tic);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(m_d);
  cudaFree(p_d);
  cudaFree(ax_d);
  cudaFree(ay_d);
  cudaFree(az_d);
  for (i=0; i<3; i++) values[i] = 0;

// No CUDA
  real_t pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  PAPI_start(EventSet);
  tic = get_time();
#pragma omp parallel for private(j) reduction(+: pdiff, pnorm, adiff, anorm)
  for (i=0; i<N; i++) {
    real_t pi = 0;
    real_t axi = 0;
    real_t ayi = 0;
    real_t azi = 0;
    real_t xi = x[i];
    real_t yi = y[i];
    real_t zi = z[i];
    for (j=0; j<N; j++) {
      real_t dx = x[j] - xi;
      real_t dy = y[j] - yi;
      real_t dz = z[j] - zi;
      real_t R2 = dx * dx + dy * dy + dz * dz + EPS2;
      real_t invR = 1.0f / sqrtf(R2);
      real_t invR3 = m[j] * invR * invR * invR;
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
  printf("No CUDA: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
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
