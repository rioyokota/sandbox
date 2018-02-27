#include "mpi_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "print.h"
#include "timer.h"
#include <vector>

using namespace hicma;

extern "C" {
  void dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO);
  void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG, int* M, int* N, double* ALPHA, double* A, int* LDA, double* B, int* LDB);
}

int main(int argc, char** argv) {
  int N = 64;
  int info;
  std::vector<int> ipiv(N);
  std::vector<double> x(N);
  std::vector<double> b(N);
  std::vector<double> A(N*N);
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    b[i] = 0;
  }
  print("Time");
  start("Init matrix");
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      double r2 = (x[i] - x[j]) * (x[i] - x[j]) + 1e-6;
      A[N*i+j] = 1 / std::sqrt(r2);
      b[i] += A[N*i+j] * x[j];
    }
  }
  stop("Init matrix");
  start("LU decomposition");
  dgetrf_(&N, &N, &A[0], &N, &ipiv[0], &info);
  stop("LU decomposition");
  char c_side = 'l';
  char c_uplo = 'l';
  char c_transa = 'n';
  char c_diag = 'u';
  int one = 1;
  double alpha = 1;
  start("Forward substitution");
  dtrsm_(&c_side, &c_uplo, &c_transa, &c_diag, &N, &one, &alpha, &A[0], &N, &b[0], &N);
  stop("Forward substitution");
  c_uplo = 'u';
  c_diag = 'n';
  start("Backward substitution");
  dtrsm_(&c_side, &c_uplo, &c_transa, &c_diag, &N, &one, &alpha, &A[0], &N, &b[0], &N);
  stop("Backward substitution");

  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x[i] - b[i]) * (x[i] - b[i]);
    norm += x[i] * x[i];
  }
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
