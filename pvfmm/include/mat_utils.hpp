#ifndef _PVFMM_MAT_UTILS_
#define _PVFMM_MAT_UTILS_
namespace pvfmm{
namespace mat{

  template <class T>
  void gemm(char TransA, char TransB,  int M,  int N,  int K,  T alpha,  T *A,  int lda,  T *B,  int ldb,  T beta, T *C,  int ldc);

  template <class T>
  void svd(char *JOBU, char *JOBVT, int *M, int *N, T *A, int *LDA,
      T *S, T *U, int *LDU, T *VT, int *LDVT, T *WORK, int *LWORK,
      int *INFO);

  template <class T>
  void pinv(T* M, int n1, int n2, T eps, T* M_);

}//end namespace
}//end namespace

#include <matrix.hpp>
#include <mat_utils.txx>

#endif //_PVFMM_MAT_UTILS_
