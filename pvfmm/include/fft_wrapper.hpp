#ifndef _PVFMM_FFT_WRAPPER_
#define _PVFMM_FFT_WRAPPER_

#if FLOAT
typedef fftwf_complex fft_complex;
typedef fftwf_plan fft_plan;
#else
typedef fftw_complex fft_complex;
typedef fftw_plan fft_plan;
#endif

namespace pvfmm{

template<typename T>
struct FFTW_t{

  static fftw_plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      double *in, const int *inembed, int istride, int idist,
      fftw_complex *out, const int *onembed, int ostride, int odist){
    return fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static fftw_plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      fftw_complex *in, const int *inembed, int istride, int idist,
      double *out, const int *onembed, int ostride, int odist){
    return fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static void fft_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out){
    fftw_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out){
    fftw_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(fftw_plan p){
    fftw_destroy_plan(p);
  }

  static void fftw_flops(const fftw_plan& p, double* add, double* mul, double* fma){
    ::fftw_flops(p, add, mul, fma);
  }

  static fftwf_plan fft_plan_many_dft_r2c(int rank, const int *n, int howmany,
      float *in, const int *inembed, int istride, int idist,
      fftwf_complex *out, const int *onembed, int ostride, int odist){
    return fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
        idist, out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static fftwf_plan fft_plan_many_dft_c2r(int rank, const int *n, int howmany,
      fftwf_complex *in, const int *inembed, int istride, int idist,
      float *out, const int *onembed, int ostride, int odist){
    return fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
        out, onembed, ostride, odist, FFTW_ESTIMATE);
  }

  static void fft_execute_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out){
    fftwf_execute_dft_r2c(p, in, out);
  }

  static void fft_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out){
    fftwf_execute_dft_c2r(p, in, out);
  }

  static void fft_destroy_plan(fftwf_plan p){
    fftwf_destroy_plan(p);
  }

  static void fftw_flops(const fftwf_plan& p, double* add, double* mul, double* fma){
    ::fftwf_flops(p, add, mul, fma);
  }

  static Matrix<Real_t> fft_r2c(size_t N1){
    size_t N2=(N1/2+1);
    Matrix<Real_t> M(N1,2*N2);
    for(size_t j=0;j<N1;j++)
    for(size_t i=0;i<N2;i++){
      M[j][2*i+0]=cos(j*i*(1.0/N1)*2.0*M_PI);
      M[j][2*i+1]=sin(j*i*(1.0/N1)*2.0*M_PI);
    }
    return M;
  }

  static Matrix<Real_t> fft_c2c(size_t N1){
    Matrix<Real_t> M(2*N1,2*N1);
    for(size_t i=0;i<N1;i++)
    for(size_t j=0;j<N1;j++){
      M[2*i+0][2*j+0]=cos(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][2*j+0]=sin(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+0][2*j+1]=-sin(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][2*j+1]= cos(j*i*(1.0/N1)*2.0*M_PI);
    }
    return M;
  }

  static Matrix<Real_t> fft_c2r(size_t N1){
    size_t N2=(N1/2+1);
    Matrix<Real_t> M(2*N2,N1);
    for(size_t i=0;i<N2;i++)
    for(size_t j=0;j<N1;j++){
      M[2*i+0][j]=2*cos(j*i*(1.0/N1)*2.0*M_PI);
      M[2*i+1][j]=2*sin(j*i*(1.0/N1)*2.0*M_PI);
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

}//end namespace

#endif //_PVFMM_FFT_WRAPPER_

