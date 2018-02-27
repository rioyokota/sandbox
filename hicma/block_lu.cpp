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
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A, int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* LDA, double* X, int* INCX, double* BETA, double* Y, int* INCY);
}

int main(int argc, char** argv) {
  int N = 6;
  int Nb = 3;
  int Nc = N / Nb;
  int info;
  std::vector<int> ipiv(Nb);
  std::vector<double> x(N);
  std::vector<double> b(N);
  std::vector<double> A(N*N);
  for (int i=0; i<N; i++) {
    x[i] = i+1;
    b[i] = 0;
  }
  std::sort(x.begin(), x.end());
  print("Time");
  start("Init matrix");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      for (int ib=0; ib<Nb; ib++) {
        for (int jb=0; jb<Nb; jb++) {
          int i = Nb * ic + ib;
          int j = Nb * jc + jb;
          double r2 = (x[i] - x[j]) * (x[i] - x[j]);
          A[Nb*Nb*Nc*ic+Nb*Nb*jc+Nb*ib+jb] = 1 / (std::sqrt(r2) + 1e-3);
          b[i] += A[Nb*Nb*Nc*ic+Nb*Nb*jc+Nb*ib+jb] * x[j];
        }
      }
    }
  }
  stop("Init matrix");
  start("LU decomposition");
  char c_l='l';
  char c_r='r';
  char c_u='u';
  char c_n='n';
  char c_t='t';
  int i1 = 1;
  double p1 = 1;
  double m1 = -1;
  std::cout << "------------------------------------" << std::endl;
  for (int ic=0; ic<Nc; ic++) {
    dgetrf_(&Nb, &Nb, &A[Nb*Nb*Nc*ic+Nb*Nb*ic], &Nb, &ipiv[0], &info);
    for (int ib=0; ib<Nb; ib++) {
      for (int jb=0; jb<Nb; jb++) {
        if (ib > jb) A[Nb*Nb*Nc*ic+Nb*Nb*ic+Nb*ib+jb] /= 1000;
        if (ib < jb) A[Nb*Nb*Nc*ic+Nb*Nb*ic+Nb*ib+jb] *= 1000;
        if (ic==1 && ib >= jb) A[Nb*Nb*Nc*ic+Nb*Nb*ic+Nb*ib+jb] = - A[Nb*Nb*Nc*ic+Nb*Nb*ic+Nb*ib+jb];
        std::cout << A[Nb*Nb*Nc*ic+Nb*Nb*ic+Nb*ib+jb] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "------------------------------------" << std::endl;
    for (int jc=ic+1; jc<Nc; jc++) {
      dtrsm_(&c_r, &c_u, &c_t, &c_u, &Nb, &Nb, &p1, &A[Nb*Nb*Nc*ic+Nb*Nb*ic], &Nb, &A[Nb*Nb*Nc*ic+Nb*Nb*jc], &Nb);
      dtrsm_(&c_r, &c_u, &c_n, &c_n, &Nb, &Nb, &p1, &A[Nb*Nb*Nc*ic+Nb*Nb*ic], &Nb, &A[Nb*Nb*Nc*jc+Nb*Nb*ic], &Nb);
    }
    for (int jc=ic+1; jc<Nc; jc++) {
      for (int kc=ic+1; kc<Nc; kc++) {
        dgemm_(&c_n, &c_n, &Nb, &Nb, &Nb, &p1, &A[Nb*Nb*Nc*jc+Nb*Nb*ic], &Nb, &A[Nb*Nb*Nc*ic+Nb*Nb*kc], &Nb, &m1, &A[Nb*Nb*Nc*jc+Nb*Nb*kc], &Nb);
      }
    }
  }
  stop("LU decomposition");
  std::cout << "------------------------------------" << std::endl;
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      for (int ib=0; ib<Nb; ib++) {
        for (int jb=0; jb<Nb; jb++) {
          std::cout << A[Nb*Nb*Nc*ic+Nb*Nb*jc+Nb*ib+jb] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << "------------------------------------" << std::endl;
    }
  }
  start("Forward substitution");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<ic; jc++) {
      dgemv_(&c_t, &Nb, &Nb, &m1, &A[Nb*Nb*Nc*ic+Nb*Nb*jc], &Nb, &b[Nb*jc], &i1, &p1, &b[Nb*ic], &i1);
    }
    dtrsm_(&c_l, &c_u, &c_t, &c_u, &Nb, &i1, &p1, &A[Nb*Nb*Nc*ic+Nb*Nb*ic], &Nb, &b[Nb*ic], &Nb);
  }
  stop("Forward substitution");
  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      for (int ib=0; ib<Nb; ib++) std::cout << b[Nb*ic+ib] << " ";
      std::cout << std::endl;
      std::cout << "------------------------------------" << std::endl;
      dgemv_(&c_t, &Nb, &Nb, &m1, &A[Nb*Nb*Nc*ic+Nb*Nb*jc], &Nb, &b[Nb*jc], &i1, &p1, &b[Nb*ic], &i1);
      for (int ib=0; ib<Nb; ib++) std::cout << b[Nb*ic+ib] << " ";
      std::cout << std::endl;
      std::cout << "------------------------------------" << std::endl;
    }
    dtrsm_(&c_l, &c_l, &c_t, &c_n, &Nb, &i1, &p1, &A[Nb*Nb*Nc*ic+Nb*Nb*ic], &Nb, &b[Nb*ic], &Nb);
  }
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
