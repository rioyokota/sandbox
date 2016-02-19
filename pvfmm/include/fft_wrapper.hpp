#ifndef _PVFMM_FFT_WRAPPER_
#define _PVFMM_FFT_WRAPPER_

#include <fftw3.h>

namespace pvfmm{

template<class T>
struct FFTW_t{

  struct plan{
    std::vector<size_t> dim;
    std::vector<Matrix<T> > M;
    size_t howmany;
  };

  struct cplx{
    T real;
    T imag;
  };

  static plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      T *in, const int *inembed, int istride, int idist,
      cplx *out, const int *onembed, int ostride, int odist){
    assert(inembed==NULL);
    assert(onembed==NULL);
    assert(istride==1);
    assert(ostride==1);

    plan p;
    p.howmany=howmany;
    {
      p.dim.push_back(n[rank-1]);
      p.M.push_back(fft_r2c(n[rank-1]));
    }
    for(int i=rank-2;i>=0;i--){
      p.dim.push_back(n[i]);
      p.M.push_back(fft_c2c(n[i]));
    }

    size_t N1=1, N2=1;
    for(size_t i=0;i<p.dim.size();i++){
      N1*=p.dim[i];
      N2*=p.M[i].Dim(1)/2;
    }
    assert(idist==N1);
    assert(odist==N2);

    return p;
  }

  static plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      cplx *in, const int *inembed, int istride, int idist,
      T *out, const int *onembed, int ostride, int odist){
    assert(inembed==NULL);
    assert(onembed==NULL);
    assert(istride==1);
    assert(ostride==1);

    plan p;
    p.howmany=howmany;
    for(size_t i=0;i<rank-1;i++){
      p.dim.push_back(n[i]);
      p.M.push_back(fft_c2c(n[i]));
    }
    {
      p.dim.push_back(n[rank-1]);
      p.M.push_back(fft_c2r(n[rank-1]));
    }

    size_t N1=1, N2=1;
    for(size_t i=0;i<p.dim.size();i++){
      N1*=p.dim[i];
      N2*=p.M[i].Dim(0)/2;
    }
    assert(idist==N2);
    assert(odist==N1);

    return p;
  }

  static void fft_execute_dft_r2c(const plan p, T *in, cplx *out){
    size_t N1=p.howmany, N2=p.howmany;
    for(size_t i=0;i<p.dim.size();i++){
      N1*=p.dim[i];
      N2*=p.M[i].Dim(1)/2;
    }
    std::vector<T> buff_(N1+2*N2);
    T* buff=&buff_[0];

    {
      size_t i=0;
      const Matrix<T>& M=p.M[i];
      assert(2*N2/M.Dim(1)==N1/M.Dim(0));
      Matrix<T> x(  N1/M.Dim(0),M.Dim(0),  in,false);
      Matrix<T> y(2*N2/M.Dim(1),M.Dim(1),buff,false);
      Matrix<T>::GEMM(y, x, M);
      transpose<cplx>(2*N2/M.Dim(1), M.Dim(1)/2, (cplx*)buff);
    }
    for(size_t i=1;i<p.dim.size();i++){
      const Matrix<T>& M=p.M[i];
      assert(M.Dim(0)==M.Dim(1));
      Matrix<T> x(2*N2/M.Dim(0),M.Dim(0),buff);
      Matrix<T> y(2*N2/M.Dim(1),M.Dim(1),buff,false);
      Matrix<T>::GEMM(y, x, M);
      transpose<cplx>(2*N2/M.Dim(1), M.Dim(1)/2, (cplx*)buff);
    }
    {
      transpose<cplx>(N2/p.howmany, p.howmany, (cplx*)buff);
      mem::memcopy(out,buff,2*N2*sizeof(T));
    }
  }

  static void fft_execute_dft_c2r(const plan p, cplx *in, T *out){
    size_t N1=p.howmany, N2=p.howmany;
    for(size_t i=0;i<p.dim.size();i++){
      N1*=p.dim[i];
      N2*=p.M[i].Dim(0)/2;
    }
    std::vector<T> buff_(N1+2*N2);
    T* buff=&buff_[0];

    {
      mem::memcopy(buff,in,2*N2*sizeof(T));
      transpose<cplx>(p.howmany, N2/p.howmany, (cplx*)buff);
    }
    for(size_t i=0;i<p.dim.size()-1;i++){
      Matrix<T> M=p.M[i];
      assert(M.Dim(0)==M.Dim(1));
      transpose<cplx>(M.Dim(0)/2, 2*N2/M.Dim(0), (cplx*)buff);
      Matrix<T> y(2*N2/M.Dim(0),M.Dim(0),buff);
      Matrix<T> x(2*N2/M.Dim(1),M.Dim(1),buff,false);
      Matrix<T>::GEMM(x, y, M.Transpose());
    }
    {
      size_t i=p.dim.size()-1;
      const Matrix<T>& M=p.M[i];
      assert(2*N2/M.Dim(0)==N1/M.Dim(1));
      transpose<cplx>(M.Dim(0)/2, 2*N2/M.Dim(0), (cplx*)buff);
      Matrix<T> y(2*N2/M.Dim(0),M.Dim(0),buff,false);
      Matrix<T> x(  N1/M.Dim(1),M.Dim(1), out,false);
      Matrix<T>::GEMM(x, y, M);
    }
  }

  static void fft_destroy_plan(plan p){
    p.dim.clear();
    p.M.clear();
    p.howmany=0;
  }

  static void fftw_flops(const plan& p, double* add, double* mul, double* fma){
    *add=0;
    *mul=0;
    *fma=0;
  }

  static Matrix<T> fft_r2c(size_t N1){
    size_t N2=(N1/2+1);
    Matrix<T> M(N1,2*N2);
    for(size_t j=0;j<N1;j++)
    for(size_t i=0;i<N2;i++){
      M[j][2*i+0]=pvfmm::cos<T>(j*i*(1.0/N1)*2.0*M_PI);
      M[j][2*i+1]=pvfmm::sin<T>(j*i*(1.0/N1)*2.0*M_PI);
    }
    return M;
  }

  static Matrix<T> fft_c2c(size_t N1){
    Matrix<T> M(2*N1,2*N1);
    for(size_t i=0;i<N1;i++)
    for(size_t j=0;j<N1;j++){
      M[2*i+0][2*j+0]=pvfmm::cos<T>(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][2*j+0]=pvfmm::sin<T>(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+0][2*j+1]=-pvfmm::sin<T>(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][2*j+1]= pvfmm::cos<T>(j*i*(1.0/N1)*2.0*M_PI);
    }
    return M;
  }

  static Matrix<T> fft_c2r(size_t N1){
    size_t N2=(N1/2+1);
    Matrix<T> M(2*N2,N1);
    for(size_t i=0;i<N2;i++)
    for(size_t j=0;j<N1;j++){
      M[2*i+0][j]=2*pvfmm::cos<T>(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][j]=2*pvfmm::sin<T>(j*i*(1.0/N1)*2.0*M_PI);
    }
    if(N2>0){
      for(size_t j=0;j<N1;j++){
        M[0][j]=M[0][j]*0.5;
        M[1][j]=M[1][j]*0.5;
      }
    }
    if(N1%2==0){
      for(size_t j=0;j<N1;j++){
        M[2*N2-2][j]=M[2*N2-2][j]*0.5;
        M[2*N2-1][j]=M[2*N2-1][j]*0.5;
      }
    }
    return M;
  }

  template <class Y>
  static void transpose(size_t dim1, size_t dim2, Y* A){
    Matrix<Y> M(dim1, dim2, A);
    Matrix<Y> Mt(dim2, dim1, A, false);
    Mt=M.Transpose();
  }

};

template<>
struct FFTW_t<double>{
  typedef fftw_plan plan;
  typedef fftw_complex cplx;

  static plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      double *in, const int *inembed, int istride, int idist,
      fftw_complex *out, const int *onembed, int ostride, int odist){
    return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      cplx *in, const int *inembed, int istride, int idist,
      double *out, const int *onembed, int ostride, int odist){
    return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static void fft_execute_dft_r2c(const plan p, double *in, cplx *out){
    fftw_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const plan p, cplx *in, double *out){
    fftw_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(plan p){
    fftw_destroy_plan(p);
  }

  static void fftw_flops(const plan& p, double* add, double* mul, double* fma){
    ::fftw_flops(p, add, mul, fma);
  }

};

template<>
struct FFTW_t<float>{
  typedef fftwf_plan plan;
  typedef fftwf_complex cplx;

  static plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      float *in, const int *inembed, int istride, int idist,
      cplx *out, const int *onembed, int ostride, int odist){
    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      cplx *in, const int *inembed, int istride, int idist,
      float *out, const int *onembed, int ostride, int odist){
    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static void fft_execute_dft_r2c(const plan p, float *in, cplx *out){
    fftwf_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const plan p, cplx *in, float *out){
    fftwf_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(plan p){
    fftwf_destroy_plan(p);
  }

  static void fftw_flops(const plan& p, double* add, double* mul, double* fma){
    ::fftwf_flops(p, add, mul, fma);
  }

};

}//end namespace

#endif //_PVFMM_FFT_WRAPPER_

