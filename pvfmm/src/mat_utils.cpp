#include <mat_utils.hpp>

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

template<>
void gemm<float>(char TransA, char TransB,  int M,  int N,  int K,  float alpha,  float *A,  int lda,  float *B,  int ldb,  float beta, float *C,  int ldc){
  sgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template<>
void gemm<double>(char TransA, char TransB,  int M,  int N,  int K,  double alpha,  double *A,  int lda,  double *B,  int ldb,  double beta, double *C,  int ldc){
  dgemm_(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

template<>
void svd<float>(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
    float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO){
  sgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
}

template<>
void svd<double>(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
    double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO){
  dgesvd_(JOBU,JOBVT,M,N,A,LDA,S,U,LDU,VT,LDVT,WORK,LWORK,INFO);
}

}//end namespace
}//end namespace
