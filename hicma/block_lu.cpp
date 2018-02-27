#include <cmath>
#include <cstdlib>
#include "mpi_utils.h"
#include "print.h"
#include "timer.h"

extern "C" {
  void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
  void dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);
  void dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* LDA, int* IPIV, double* B, int* LDB, int* INFO);
  double dlamch_(char* CMACH);
  double dlange_(char* NORM, int* M, int* N, double* A, int* LDA, double* WORK);
  void dlaswp_(int* N, double* A, int* LDA, int* K1, int* K2, int* IPIV, int* INCX);
  void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, double* ALPHA, double* A, int* LDA, double* B, int* LDB);
}

int main(int argc, char** argv) {
  int N = 64;
  int Nb = 32;
  int Nc = N/Nb;
  int one = 1;
  int info;
  int* ipiv = new int [N+1];
  double* x = new double [N];
  double* b = new double [N];
  double* A = new double [N*N];
  double* LU = new double [N*N];
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    b[i] = 0;
  }
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double r2 = (x[i] - x[j]) * (x[i] - x[j]) + 1e-6;
      LU[N*i+j] = A[N*i+j] = 1 / std::sqrt(r2);
      b[i] += A[N*i+j] * x[j];
    }
  }
  dgetrf_(&N, &N, A, &N, ipiv, &info);
  char c_side = 'l';
  char c_uplo = 'l';
  char c_transa = 'n';
  char c_diag = 'u';
  double alpha = 1;
  dtrsm_(&c_side, &c_uplo, &c_transa, &c_diag, &N, &one, &alpha, A, &N, b, &N);
  c_uplo = 'u';
  c_diag = 'n';
  dtrsm_(&c_side, &c_uplo, &c_transa, &c_diag, &N, &one, &alpha, A, &N, b, &N);

  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x[i] - b[i]) * (x[i] - b[i]);
    norm += x[i] * x[i];
  }
  std::cout << std::sqrt(diff/norm) << std::endl;
  delete[] LU;
  delete[] A;
  delete[] x;
  delete[] b;
  delete[] ipiv;
  return 0;
}
