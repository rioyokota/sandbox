#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char ** argv) {
// Initialize
  int N = 1 << 23; //atoi(argv[1]);
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  double tic, toc;
  float * x = (float*) malloc(N * sizeof(float));
  float * y = (float*) malloc(N * sizeof(float));
  float * z = (float*) malloc(N * sizeof(float));
  float * m = (float*) malloc(N * sizeof(float));
  float * p = (float*) malloc(N * sizeof(float));
  float * ax = (float*) malloc(N * sizeof(float));
  float * ay = (float*) malloc(N * sizeof(float));
  float * az = (float*) malloc(N * sizeof(float));
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  //printf("Rank   : %d/%d\n",mpirank,mpisize);
  if(mpirank==0) printf("N      : %d\n",N);

  tic = get_time();
#pragma omp parallel for reduction(+: pdiff, pnorm, adiff, anorm)
  for (int i=0; i<N; i++) {
    if (i % (N / 100) == 0 && mpirank == 0) printf("%d %\n",i/(N/100)); 
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    for (int j=0; j<N; j++) {
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
    p[i] = pi;
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
  }
  toc = get_time();
  if(mpirank==0) {
    printf("No SIMD: %e s : %lf GFlops\n",toc-tic, OPS/(toc-tic));
  }

// DEALLOCATE
  free(x);
  free(y);
  free(z);
  free(m);
  free(p);
  free(ax);
  free(ay);
  free(az);
  MPI_Finalize();
  return 0;
}
