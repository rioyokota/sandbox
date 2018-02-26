#include <cmath>
#include <cstdlib>
#include "mpi_utils.h"
#include "print.h"
#include "timer.h"

extern "C" {
  void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
}

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 8;
  int Nc = N/Nb;
  int Nrhs = 1;
  int info;
  int* ipiv = new int [N+1];
  double* x = new double [N];
  double* b = new double [N];
  double* A = new double [N*N];
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    b[i] = 0;
  }
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double r2 = (x[i] - x[j]) * (x[i] - x[j]) + 1e-6;
      A[N*i+j] = 1 / std::sqrt(r2);
      b[i] += A[N*i+j] * x[j];
    }
  }
  dgesv_(&N, &Nrhs, A, &N, ipiv, b, &N, &info);
  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x[i] - b[i]) * (x[i] - b[i]);
    norm += x[i] * x[i];
  }
  std::cout << std::sqrt(diff/norm) << std::endl;
  delete[] A;
  delete[] x;
  delete[] b;
  delete[] ipiv;
  return 0;
}
