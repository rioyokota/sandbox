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
  void tpinv(T* M, int n1, int n2, T eps, T* M_){
    if(n1*n2==0) return;
    int m = n2;
    int n = n1;
    int k = (m<n?m:n);
    T* tU =mem::aligned_new<T>(m*k);
    T* tS =mem::aligned_new<T>(k);
    T* tVT=mem::aligned_new<T>(k*n);
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);
    T* wsbuf = mem::aligned_new<T>(wssize);
    tsvd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k,
        wsbuf, &wssize, &INFO);
    if(INFO!=0)
      std::cout<<INFO<<'\n';
    assert(INFO==0);
    mem::aligned_delete<T>(wsbuf);
    T eps_=tS[0]*eps;
    for(int i=0;i<k;i++)
      if(tS[i]<eps_)
        tS[i]=0;
      else
        tS[i]=1.0/tS[i];
    for(int i=0;i<m;i++){
      for(int j=0;j<k;j++){
        tU[i+j*m]*=tS[j];
      }
    }
    tgemm<T>('T','T',n,m,k,1.0,&tVT[0],k,&tU[0],m,0.0,M_,n);
    mem::aligned_delete<T>(tU);
    mem::aligned_delete<T>(tS);
    mem::aligned_delete<T>(tVT);
  }

}//end namespace
}//end namespace

#include <matrix.hpp>

#endif //_PVFMM_MAT_UTILS_
