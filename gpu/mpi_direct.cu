#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
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

int main(int argc, char **argv) {
  // Device check
  int mpisize, mpirank, gpusize, gpurank;
  cudaGetDeviceCount(&gpusize);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  cudaSetDevice(mpirank % gpusize);
  cudaGetDevice(&gpurank);
  for (int irank=0; irank!=mpisize; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpirank == irank) {
      printf("MPI rank    : %d / %d  GPU device : %d / %d\n",
             mpirank, mpisize, gpurank, gpusize);
    }
  }

  // Initialize
  //int N = 1 << 16;
  int N = 1 << 24;
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
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  if(!mpirank) printf("N      : %d\n",N);
  MPI_Finalize();

  // CUDA
  real_t *x_d, *y_d, *z_d, *m_d, *p_d, *ax_d, *ay_d, *az_d;
  cudaMalloc((void**)&x_d, N * sizeof(real_t));
  cudaMalloc((void**)&y_d, N * sizeof(real_t));
  cudaMalloc((void**)&z_d, N * sizeof(real_t));
  cudaMalloc((void**)&m_d, N * sizeof(real_t));
  cudaMalloc((void**)&p_d, N * sizeof(real_t));
  cudaMalloc((void**)&ax_d, N * sizeof(real_t));
  cudaMalloc((void**)&ay_d, N * sizeof(real_t));
  cudaMalloc((void**)&az_d, N * sizeof(real_t));
  cudaMemcpy(x_d, x, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(z_d, z, N * sizeof(real_t), cudaMemcpyHostToDevice);
  cudaMemcpy(m_d, m, N * sizeof(real_t), cudaMemcpyHostToDevice);
  for (int i=0; i<20; i++) {
    tic = get_time();
    GPUkernel<<<N/THREADS,THREADS>>>(N, x_d, y_d, z_d, m_d, p_d, ax_d, ay_d, az_d, EPS2);
    toc = get_time();
    if(!mpirank) printf(" i     : %d, %lf s\n",i,toc-tic);
  }
  cudaMemcpy(p, p_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ax, ax_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ay, ay_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(az, az_d, N * sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(m_d);
  cudaFree(p_d);
  cudaFree(ax_d);
  cudaFree(ay_d);
  cudaFree(az_d);
  if(!mpirank) printf("CUDA   : %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));

#if 0
  // No CUDA
  real_t pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  tic = get_time();
#pragma omp parallel for reduction(+: pdiff, pnorm, adiff, anorm)
  for (int i=0; i<N; i++) {
    real_t pi = 0;
    real_t axi = 0;
    real_t ayi = 0;
    real_t azi = 0;
    real_t xi = x[i];
    real_t yi = y[i];
    real_t zi = z[i];
    for (int j=0; j<N; j++) {
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
  if(!mpirank) {
    printf("No CUDA: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
    printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
    printf("A ERR  : %e\n",sqrt(adiff/anorm));
  }
#endif
  // Deallocate
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
