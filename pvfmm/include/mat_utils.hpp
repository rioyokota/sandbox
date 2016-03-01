#ifndef _PVFMM_MAT_UTILS_
#define _PVFMM_MAT_UTILS_
extern "C"
{
  void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA, float* A,
	      int* LDA, float* B, int* LDB, float* BETA, float* C, int* LDC);
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
	      int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  void sgesvd_(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
	       float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO);
  void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
	       double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
}

namespace pvfmm{
namespace mat{

  template <class T>
  void tgemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc) {
    sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  template<>
  void tgemm<double>(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc){
    dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
  }

  template <class T>
  void tsvd(char *JOBU, char *JOBVT, int *M, int *N, T *A, int *LDA,
	    T *S, T *U, int *LDU, T *VT, int *LDVT, T *WORK, int *LWORK,
	    int *INFO) {
    sgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
  }

  template<>
  void tsvd<double>(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
		    double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO){
    dgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
  }

  template <class T>
  void tpinv(T* M, int n1, int n2, T eps, T* M_);

}//end namespace
}//end namespace

#include <matrix.hpp>
#include <mat_utils.txx>

#endif //_PVFMM_MAT_UTILS_
