#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  float ** A = new float * [N];
  float ** B = new float * [N];
  float ** C = new float * [N];
  for (int i=0; i<N; i++) {
    A[i] = new float [N];
    B[i] = new float [N];
    C[i] = new float [N];
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  double tic = get_time();
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  double toc = get_time();
  printf("N=%d: %lf s (%lf GFlops)\n",N,toc-tic,2.*N*N*N/(toc-tic)/1e9);
  for (int i=0; i<N; i++) {
    delete[] A[i];
    delete[] B[i];
    delete[] C[i];
  }
  delete[] A;
  delete[] B;
  delete[] C;
}
