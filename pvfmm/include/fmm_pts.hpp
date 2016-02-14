#ifndef _PVFMM_FMM_PTS_HPP_
#define _PVFMM_FMM_PTS_HPP_

#include <fmm_node.hpp>

namespace pvfmm{

template<class Real_t>
inline void matmult_8x8x2(Real_t*& M_, Real_t*& IN0, Real_t*& IN1, Real_t*& OUT0, Real_t*& OUT1){
  Real_t out_reg000, out_reg001, out_reg010, out_reg011;
  Real_t out_reg100, out_reg101, out_reg110, out_reg111;
  Real_t  in_reg000,  in_reg001,  in_reg010,  in_reg011;
  Real_t  in_reg100,  in_reg101,  in_reg110,  in_reg111;
  Real_t   m_reg000,   m_reg001,   m_reg010,   m_reg011;
  Real_t   m_reg100,   m_reg101,   m_reg110,   m_reg111;
  for(int i1=0;i1<8;i1+=2){
    Real_t* IN0_=IN0;
    Real_t* IN1_=IN1;
    out_reg000=OUT0[ 0]; out_reg001=OUT0[ 1];
    out_reg010=OUT0[ 2]; out_reg011=OUT0[ 3];
    out_reg100=OUT1[ 0]; out_reg101=OUT1[ 1];
    out_reg110=OUT1[ 2]; out_reg111=OUT1[ 3];
    for(int i2=0;i2<8;i2+=2){
      m_reg000=M_[ 0]; m_reg001=M_[ 1];
      m_reg010=M_[ 2]; m_reg011=M_[ 3];
      m_reg100=M_[16]; m_reg101=M_[17];
      m_reg110=M_[18]; m_reg111=M_[19];
      in_reg000=IN0_[0]; in_reg001=IN0_[1];
      in_reg010=IN0_[2]; in_reg011=IN0_[3];
      in_reg100=IN1_[0]; in_reg101=IN1_[1];
      in_reg110=IN1_[2]; in_reg111=IN1_[3];
      out_reg000 += m_reg000*in_reg000 - m_reg001*in_reg001;
      out_reg001 += m_reg000*in_reg001 + m_reg001*in_reg000;
      out_reg010 += m_reg010*in_reg000 - m_reg011*in_reg001;
      out_reg011 += m_reg010*in_reg001 + m_reg011*in_reg000;
      out_reg000 += m_reg100*in_reg010 - m_reg101*in_reg011;
      out_reg001 += m_reg100*in_reg011 + m_reg101*in_reg010;
      out_reg010 += m_reg110*in_reg010 - m_reg111*in_reg011;
      out_reg011 += m_reg110*in_reg011 + m_reg111*in_reg010;
      out_reg100 += m_reg000*in_reg100 - m_reg001*in_reg101;
      out_reg101 += m_reg000*in_reg101 + m_reg001*in_reg100;
      out_reg110 += m_reg010*in_reg100 - m_reg011*in_reg101;
      out_reg111 += m_reg010*in_reg101 + m_reg011*in_reg100;
      out_reg100 += m_reg100*in_reg110 - m_reg101*in_reg111;
      out_reg101 += m_reg100*in_reg111 + m_reg101*in_reg110;
      out_reg110 += m_reg110*in_reg110 - m_reg111*in_reg111;
      out_reg111 += m_reg110*in_reg111 + m_reg111*in_reg110;
      M_+=32;
      IN0_+=4;
      IN1_+=4;
    }
    OUT0[ 0]=out_reg000; OUT0[ 1]=out_reg001;
    OUT0[ 2]=out_reg010; OUT0[ 3]=out_reg011;
    OUT1[ 0]=out_reg100; OUT1[ 1]=out_reg101;
    OUT1[ 2]=out_reg110; OUT1[ 3]=out_reg111;
    M_+=4-64*2;
    OUT0+=4;
    OUT1+=4;
  }
}

#if defined(__AVX__) || defined(__SSE3__)
template<>
inline void matmult_8x8x2<double>(double*& M_, double*& IN0, double*& IN1, double*& OUT0, double*& OUT1){
#ifdef __AVX__
  __m256d out00,out01,out10,out11;
  __m256d out20,out21,out30,out31;
  double* in0__ = IN0;
  double* in1__ = IN1;
  out00 = _mm256_load_pd(OUT0);
  out01 = _mm256_load_pd(OUT1);
  out10 = _mm256_load_pd(OUT0+4);
  out11 = _mm256_load_pd(OUT1+4);
  out20 = _mm256_load_pd(OUT0+8);
  out21 = _mm256_load_pd(OUT1+8);
  out30 = _mm256_load_pd(OUT0+12);
  out31 = _mm256_load_pd(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m256d m00;
    __m256d ot00;
    __m256d mt0,mtt0;
    __m256d in00,in00_r,in01,in01_r;
    in00 = _mm256_broadcast_pd((const __m128d*)in0__);
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*)in1__);
    in01_r = _mm256_permute_pd(in01,5);
    m00 = _mm256_load_pd(M_);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+4);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+8);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+12);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    in00 = _mm256_broadcast_pd((const __m128d*) (in0__+2));
    in00_r = _mm256_permute_pd(in00,5);
    in01 = _mm256_broadcast_pd((const __m128d*) (in1__+2));
    in01_r = _mm256_permute_pd(in01,5);
    m00 = _mm256_load_pd(M_+16);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out00 = _mm256_add_pd(out00,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out01 = _mm256_add_pd(out01,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+20);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out10 = _mm256_add_pd(out10,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out11 = _mm256_add_pd(out11,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+24);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out20 = _mm256_add_pd(out20,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out21 = _mm256_add_pd(out21,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    m00 = _mm256_load_pd(M_+28);
    mt0 = _mm256_unpacklo_pd(m00,m00);
    ot00 = _mm256_mul_pd(mt0,in00);
    mtt0 = _mm256_unpackhi_pd(m00,m00);
    out30 = _mm256_add_pd(out30,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in00_r)));
    ot00 = _mm256_mul_pd(mt0,in01);
    out31 = _mm256_add_pd(out31,_mm256_addsub_pd(ot00,_mm256_mul_pd(mtt0,in01_r)));
    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm256_store_pd(OUT0,out00);
  _mm256_store_pd(OUT1,out01);
  _mm256_store_pd(OUT0+4,out10);
  _mm256_store_pd(OUT1+4,out11);
  _mm256_store_pd(OUT0+8,out20);
  _mm256_store_pd(OUT1+8,out21);
  _mm256_store_pd(OUT0+12,out30);
  _mm256_store_pd(OUT1+12,out31);
#elif defined __SSE3__
  __m128d out00, out01, out10, out11;
  __m128d in00, in01, in10, in11;
  __m128d m00, m01, m10, m11;
  for(int i1=0;i1<8;i1+=2){
    double* IN0_=IN0;
    double* IN1_=IN1;
    out00 =_mm_load_pd (OUT0  );
    out10 =_mm_load_pd (OUT0+2);
    out01 =_mm_load_pd (OUT1  );
    out11 =_mm_load_pd (OUT1+2);
    for(int i2=0;i2<8;i2+=2){
      m00 =_mm_load1_pd (M_   );
      m10 =_mm_load1_pd (M_+ 2);
      m01 =_mm_load1_pd (M_+16);
      m11 =_mm_load1_pd (M_+18);
      in00 =_mm_load_pd (IN0_  );
      in10 =_mm_load_pd (IN0_+2);
      in01 =_mm_load_pd (IN1_  );
      in11 =_mm_load_pd (IN1_+2);
      out00 = _mm_add_pd   (out00, _mm_mul_pd(m00 , in00 ));
      out00 = _mm_add_pd   (out00, _mm_mul_pd(m01 , in10 ));
      out01 = _mm_add_pd   (out01, _mm_mul_pd(m00 , in01 ));
      out01 = _mm_add_pd   (out01, _mm_mul_pd(m01 , in11 ));
      out10 = _mm_add_pd   (out10, _mm_mul_pd(m10 , in00 ));
      out10 = _mm_add_pd   (out10, _mm_mul_pd(m11 , in10 ));
      out11 = _mm_add_pd   (out11, _mm_mul_pd(m10 , in01 ));
      out11 = _mm_add_pd   (out11, _mm_mul_pd(m11 , in11 ));
      m00 =_mm_load1_pd (M_+   1);
      m10 =_mm_load1_pd (M_+ 2+1);
      m01 =_mm_load1_pd (M_+16+1);
      m11 =_mm_load1_pd (M_+18+1);
      in00 =_mm_shuffle_pd (in00,in00,_MM_SHUFFLE2(0,1));
      in01 =_mm_shuffle_pd (in01,in01,_MM_SHUFFLE2(0,1));
      in10 =_mm_shuffle_pd (in10,in10,_MM_SHUFFLE2(0,1));
      in11 =_mm_shuffle_pd (in11,in11,_MM_SHUFFLE2(0,1));
      out00 = _mm_addsub_pd(out00, _mm_mul_pd(m00, in00));
      out00 = _mm_addsub_pd(out00, _mm_mul_pd(m01, in10));
      out01 = _mm_addsub_pd(out01, _mm_mul_pd(m00, in01));
      out01 = _mm_addsub_pd(out01, _mm_mul_pd(m01, in11));
      out10 = _mm_addsub_pd(out10, _mm_mul_pd(m10, in00));
      out10 = _mm_addsub_pd(out10, _mm_mul_pd(m11, in10));
      out11 = _mm_addsub_pd(out11, _mm_mul_pd(m10, in01));
      out11 = _mm_addsub_pd(out11, _mm_mul_pd(m11, in11));
      M_+=32;
      IN0_+=4;
      IN1_+=4;
    }
    _mm_store_pd (OUT0  ,out00);
    _mm_store_pd (OUT0+2,out10);
    _mm_store_pd (OUT1  ,out01);
    _mm_store_pd (OUT1+2,out11);
    M_+=4-64*2;
    OUT0+=4;
    OUT1+=4;
  }
#endif
}
#endif

#if defined(__SSE3__)
template<>
inline void matmult_8x8x2<float>(float*& M_, float*& IN0, float*& IN1, float*& OUT0, float*& OUT1){
#if defined __SSE3__
  __m128 out00,out01,out10,out11;
  __m128 out20,out21,out30,out31;
  float* in0__ = IN0;
  float* in1__ = IN1;
  out00 = _mm_load_ps(OUT0);
  out01 = _mm_load_ps(OUT1);
  out10 = _mm_load_ps(OUT0+4);
  out11 = _mm_load_ps(OUT1+4);
  out20 = _mm_load_ps(OUT0+8);
  out21 = _mm_load_ps(OUT1+8);
  out30 = _mm_load_ps(OUT0+12);
  out31 = _mm_load_ps(OUT1+12);
  for(int i2=0;i2<8;i2+=2){
    __m128 m00;
    __m128 mt0,mtt0;
    __m128 in00,in00_r,in01,in01_r;
    in00 = _mm_castpd_ps(_mm_load_pd1((const double*)in0__));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*)in1__));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
    m00 = _mm_load_ps(M_);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+4);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01  ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+8);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+12);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,  in00));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
    in00 = _mm_castpd_ps(_mm_load_pd1((const double*) (in0__+2)));
    in00_r = _mm_shuffle_ps(in00,in00,_MM_SHUFFLE(2,3,0,1));
    in01 = _mm_castpd_ps(_mm_load_pd1((const double*) (in1__+2)));
    in01_r = _mm_shuffle_ps(in01,in01,_MM_SHUFFLE(2,3,0,1));
    m00 = _mm_load_ps(M_+16);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out00= _mm_add_ps   (out00,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out00= _mm_addsub_ps(out00,_mm_mul_ps(mtt0,in00_r));
    out01 = _mm_add_ps   (out01,_mm_mul_ps( mt0,in01  ));
    out01 = _mm_addsub_ps(out01,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+20);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out10= _mm_add_ps   (out10,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out10= _mm_addsub_ps(out10,_mm_mul_ps(mtt0,in00_r));
    out11 = _mm_add_ps   (out11,_mm_mul_ps( mt0,in01 ));
    out11 = _mm_addsub_ps(out11,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+24);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out20= _mm_add_ps   (out20,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out20= _mm_addsub_ps(out20,_mm_mul_ps(mtt0,in00_r));
    out21 = _mm_add_ps   (out21,_mm_mul_ps( mt0,in01  ));
    out21 = _mm_addsub_ps(out21,_mm_mul_ps(mtt0,in01_r));
    m00 = _mm_load_ps(M_+28);
    mt0  = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(2,2,0,0));
    out30= _mm_add_ps   (out30,_mm_mul_ps( mt0,in00  ));
    mtt0 = _mm_shuffle_ps(m00,m00,_MM_SHUFFLE(3,3,1,1));
    out30= _mm_addsub_ps(out30,_mm_mul_ps(mtt0,in00_r));
    out31 = _mm_add_ps   (out31,_mm_mul_ps( mt0,in01  ));
    out31 = _mm_addsub_ps(out31,_mm_mul_ps(mtt0,in01_r));
    M_ += 32;
    in0__ += 4;
    in1__ += 4;
  }
  _mm_store_ps(OUT0,out00);
  _mm_store_ps(OUT1,out01);
  _mm_store_ps(OUT0+4,out10);
  _mm_store_ps(OUT1+4,out11);
  _mm_store_ps(OUT0+8,out20);
  _mm_store_ps(OUT1+8,out21);
  _mm_store_ps(OUT0+12,out30);
  _mm_store_ps(OUT1+12,out31);
#endif
}
#endif

template <class Real_t, class FMMNode_t>
struct SetupData {
  int level;
  const Kernel<Real_t>* kernel;
  std::vector<Mat_Type> interac_type;
  std::vector<FMMNode_t*> nodes_in ;
  std::vector<FMMNode_t*> nodes_out;
  std::vector<Vector<Real_t>*>  input_vector;
  std::vector<Vector<Real_t>*> output_vector;
  Matrix< char>  interac_data;
  Matrix< char>* precomp_data;
  Matrix<Real_t>*  coord_data;
  Matrix<Real_t>*  input_data;
  Matrix<Real_t>* output_data;
};

template <class FMM_Mat_t>
class FMM_Tree;

template <class FMMNode>
class FMM_Pts {

 public:

  typedef FMMNode FMMNode_t;
  typedef FMM_Tree<FMMNode> FMMTree_t;

 protected:

  mem::MemoryManager* mem_mgr;
  InteracList<FMMNode_t> interac_list;
  const Kernel<Real_t>* kernel;
  PrecompMat<Real_t>* mat;
  std::string mat_fname;
  int multipole_order;
  typename FFTW_t<Real_t>::plan vprecomp_fftplan;
  bool vprecomp_fft_flag;
  typename FFTW_t<Real_t>::plan vlist_fftplan;
  bool vlist_fft_flag;
  typename FFTW_t<Real_t>::plan vlist_ifftplan;
  bool vlist_ifft_flag;

    
  template <class Real_t>
  std::vector<Real_t> surface(int p, Real_t* c, Real_t alpha, int depth){
    size_t n_=(6*(p-1)*(p-1)+2);
    std::vector<Real_t> coord(n_*3);
    coord[0]=coord[1]=coord[2]=-1.0;
    size_t cnt=1;
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=-1.0;
        coord[cnt*3+1]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+2]=(2.0*j-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*i-p+1)/(p-1);
        coord[cnt*3+1]=-1.0;
        coord[cnt*3+2]=(2.0*(j+1)-p+1)/(p-1);
        cnt++;
      }
    for(int i=0;i<p-1;i++)
      for(int j=0;j<p-1;j++){
        coord[cnt*3  ]=(2.0*(i+1)-p+1)/(p-1);
        coord[cnt*3+1]=(2.0*j-p+1)/(p-1);
        coord[cnt*3+2]=-1.0;
        cnt++;
      }
    for(size_t i=0;i<(n_/2)*3;i++)
      coord[cnt*3+i]=-coord[i];
    Real_t r = 0.5*pvfmm::pow<Real_t>(0.5,depth);
    Real_t b = alpha*r;
    for(size_t i=0;i<n_;i++){
      coord[i*3+0]=(coord[i*3+0]+1.0)*b+c[0];
      coord[i*3+1]=(coord[i*3+1]+1.0)*b+c[1];
      coord[i*3+2]=(coord[i*3+2]+1.0)*b+c[2];
    }
    return coord;
  }
  
  template <class Real_t>
  std::vector<Real_t> u_check_surf(int p, Real_t* c, int depth){
    Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
    Real_t coord[3]={(Real_t)(c[0]-r*(RAD1-1.0)),(Real_t)(c[1]-r*(RAD1-1.0)),(Real_t)(c[2]-r*(RAD1-1.0))};
    return surface(p,coord,(Real_t)RAD1,depth);
  }
  
  template <class Real_t>
  std::vector<Real_t> u_equiv_surf(int p, Real_t* c, int depth){
    Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
    Real_t coord[3]={(Real_t)(c[0]-r*(RAD0-1.0)),(Real_t)(c[1]-r*(RAD0-1.0)),(Real_t)(c[2]-r*(RAD0-1.0))};
    return surface(p,coord,(Real_t)RAD0,depth);
  }
  
  template <class Real_t>
  std::vector<Real_t> d_check_surf(int p, Real_t* c, int depth){
    Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
    Real_t coord[3]={(Real_t)(c[0]-r*(RAD0-1.0)),(Real_t)(c[1]-r*(RAD0-1.0)),(Real_t)(c[2]-r*(RAD0-1.0))};
    return surface(p,coord,(Real_t)RAD0,depth);
  }
  
  template <class Real_t>
  std::vector<Real_t> d_equiv_surf(int p, Real_t* c, int depth){
    Real_t r=0.5*pvfmm::pow<Real_t>(0.5,depth);
    Real_t coord[3]={(Real_t)(c[0]-r*(RAD1-1.0)),(Real_t)(c[1]-r*(RAD1-1.0)),(Real_t)(c[2]-r*(RAD1-1.0))};
    return surface(p,coord,(Real_t)RAD1,depth);
  }
  
  template <class Real_t>
  std::vector<Real_t> conv_grid(int p, Real_t* c, int depth){
    Real_t r=pvfmm::pow<Real_t>(0.5,depth);
    Real_t a=r*RAD0;
    Real_t coord[3]={c[0],c[1],c[2]};
    int n1=p*2;
    int n2=pvfmm::pow<int>((Real_t)n1,2);
    int n3=pvfmm::pow<int>((Real_t)n1,3);
    std::vector<Real_t> grid(n3*3);
    for(int i=0;i<n1;i++)
    for(int j=0;j<n1;j++)
    for(int k=0;k<n1;k++){
      grid[(i+n1*j+n2*k)*3+0]=(i-p)*a/(p-1)+coord[0];
      grid[(i+n1*j+n2*k)*3+1]=(j-p)*a/(p-1)+coord[1];
      grid[(i+n1*j+n2*k)*3+2]=(k-p)*a/(p-1)+coord[2];
    }
    return grid;
  }

  template <class Real_t>
  Permutation<Real_t> equiv_surf_perm(size_t m, size_t p_indx, const Permutation<Real_t>& ker_perm, const Vector<Real_t>* scal_exp=NULL){
    Real_t eps=1e-10;
    int dof=ker_perm.Dim();
  
    Real_t c[3]={-0.5,-0.5,-0.5};
    std::vector<Real_t> trg_coord=d_check_surf(m,c,0);
    int n_trg=trg_coord.size()/3;
  
    Permutation<Real_t> P=Permutation<Real_t>(n_trg*dof);
    if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ) {
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+0]-trg_coord[j*3+0]*(p_indx==ReflecX?-1.0:1.0))<eps)
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+1]-trg_coord[j*3+1]*(p_indx==ReflecY?-1.0:1.0))<eps)
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+2]-trg_coord[j*3+2]*(p_indx==ReflecZ?-1.0:1.0))<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else if(p_indx==SwapXY || p_indx==SwapXZ){
      for(int i=0;i<n_trg;i++)
      for(int j=0;j<n_trg;j++){
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+0]-trg_coord[j*3+(p_indx==SwapXY?1:2)])<eps)
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+1]-trg_coord[j*3+(p_indx==SwapXY?0:1)])<eps)
        if(pvfmm::fabs<Real_t>(trg_coord[i*3+2]-trg_coord[j*3+(p_indx==SwapXY?2:0)])<eps){
          for(int k=0;k<dof;k++){
            P.perm[j*dof+k]=i*dof+ker_perm.perm[k];
          }
        }
      }
    }else{
      for(int j=0;j<n_trg;j++){
        for(int k=0;k<dof;k++){
          P.perm[j*dof+k]=j*dof+ker_perm.perm[k];
        }
      }
    }
  
    if(scal_exp && p_indx==Scaling) {
      assert(dof==scal_exp->Dim());
      Vector<Real_t> scal(scal_exp->Dim());
      for(size_t i=0;i<scal.Dim();i++){
        scal[i]=pvfmm::pow<Real_t>(2.0,(*scal_exp)[i]);
      }
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=scal[i];
        }
      }
    }
    {
      for(int j=0;j<n_trg;j++){
        for(int i=0;i<dof;i++){
          P.scal[j*dof+i]*=ker_perm.scal[i];
        }
      }
    }
    return P;
  }
  
  void PrecompAll(Mat_Type type, int level=-1) {
    if(level==-1) {
      for(int l=0;l<MAX_DEPTH;l++) {
        PrecompAll(type, l);
      }
      return;
    }
    for(size_t i=0;i<Perm_Count;i++) {
      this->PrecompPerm(type, (Perm_Type) i);
    }
    size_t mat_cnt=interac_list.ListCount(type);
    mat->Mat(level, type, mat_cnt-1);
    std::vector<size_t> indx_lst;
    for(size_t i=0; i<mat_cnt; i++) {
      if(interac_list.InteracClass(type,i)==i) {
        indx_lst.push_back(i);
      }
    }
    for(size_t i=0; i<indx_lst.size(); i++){
      Precomp(level, type, indx_lst[i]);
    }
    for(size_t mat_indx=0;mat_indx<mat_cnt;mat_indx++){
      Matrix<Real_t>& M0=interac_list.ClassMat(level, type, mat_indx);
      Permutation<Real_t>& pr=interac_list.Perm_R(level, type, mat_indx);
      Permutation<Real_t>& pc=interac_list.Perm_C(level, type, mat_indx);
      if(pr.Dim()!=M0.Dim(0) || pc.Dim()!=M0.Dim(1)) Precomp(level, type, mat_indx);
    }
  }
  
  Permutation<Real_t>& PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<Real_t>& P_ = mat->Perm(type, perm_indx);
    if(P_.Dim()!=0) return P_;
    size_t m=this->MultipoleOrder();
    size_t p_indx=perm_indx % C_Perm;
    Permutation<Real_t> P;
    switch (type){
      case U2U_Type:
      {
        Vector<Real_t> scal_exp;
        Permutation<Real_t> ker_perm;
        if(perm_indx<C_Perm) {
          ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
          scal_exp=kernel->k_m2m->src_scal;
        }else{
          ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
          scal_exp=kernel->k_m2m->src_scal;
          for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
        }
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
        break;
      }
      case D2D_Type:
      {
        Vector<Real_t> scal_exp;
        Permutation<Real_t> ker_perm;
        if(perm_indx<C_Perm){
          ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
          scal_exp=kernel->k_l2l->trg_scal;
          for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
        }else{
          ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
          scal_exp=kernel->k_l2l->trg_scal;
        }
        P=equiv_surf_perm(m, p_indx, ker_perm, (this->ScaleInvar()?&scal_exp:NULL));
        break;
      }
      default:
        break;
    }
#pragma omp critical (PRECOMP_MATRIX_PTS)
    {
      if(P_.Dim()==0) P_=P;
    }
    return P_;
  }
  
  Matrix<Real_t>& Precomp(int level, Mat_Type type, size_t mat_indx) {
    if(this->ScaleInvar()) level=0;
    Matrix<Real_t>& M_ = this->mat->Mat(level, type, mat_indx);
    if(M_.Dim(0)!=0 && M_.Dim(1)!=0) return M_;
    else{
      size_t class_indx = this->interac_list.InteracClass(type, mat_indx);
      if(class_indx!=mat_indx){
        Matrix<Real_t>& M0 = this->Precomp(level, type, class_indx);
        if(M0.Dim(0)==0 || M0.Dim(1)==0) return M_;
  
        for(size_t i=0;i<Perm_Count;i++) this->PrecompPerm(type, (Perm_Type) i);
        Permutation<Real_t>& Pr = this->interac_list.Perm_R(level, type, mat_indx);
        Permutation<Real_t>& Pc = this->interac_list.Perm_C(level, type, mat_indx);
        if(Pr.Dim()>0 && Pc.Dim()>0 && M0.Dim(0)>0 && M0.Dim(1)>0) return M_;
      }
    }
    Matrix<Real_t> M;
    switch (type){
      case UC2UE0_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2m->ker_dim;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> uc_coord=u_check_surf(MultipoleOrder(),c,level);
        size_t n_uc=uc_coord.size()/3;
        std::vector<Real_t> ue_coord=u_equiv_surf(MultipoleOrder(),c,level);
        size_t n_ue=ue_coord.size()/3;
        Matrix<Real_t> M_e2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
        kernel->k_m2m->BuildMatrix(&ue_coord[0], n_ue, &uc_coord[0], n_uc, &(M_e2c[0][0]));
        Matrix<Real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        Real_t eps=1, max_S=0;
        while(eps*(Real_t)0.5+(Real_t)1.0>1.0) eps*=0.5;
        for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
          if(pvfmm::fabs<Real_t>(S[i][i])>max_S) max_S=pvfmm::fabs<Real_t>(S[i][i]);
        }
        for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
        M=V.Transpose()*S;
        break;
      }
      case UC2UE1_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2m->ker_dim;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> uc_coord=u_check_surf(MultipoleOrder(),c,level);
        size_t n_uc=uc_coord.size()/3;
        std::vector<Real_t> ue_coord=u_equiv_surf(MultipoleOrder(),c,level);
        size_t n_ue=ue_coord.size()/3;
        Matrix<Real_t> M_e2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
        kernel->k_m2m->BuildMatrix(&ue_coord[0], n_ue, &uc_coord[0], n_uc, &(M_e2c[0][0]));
        Matrix<Real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        M=U.Transpose();
        break;
      }
      case DC2DE0_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_l2l->ker_dim;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
        size_t n_ch=check_surf.size()/3;
        std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
        size_t n_eq=equiv_surf.size()/3;
        Matrix<Real_t> M_e2c(n_eq*ker_dim[0],n_ch*ker_dim[1]);
        kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_eq, &check_surf[0], n_ch, &(M_e2c[0][0]));
        Matrix<Real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        Real_t eps=1, max_S=0;
        while(eps*(Real_t)0.5+(Real_t)1.0>1.0) eps*=0.5;
        for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
          if(pvfmm::fabs<Real_t>(S[i][i])>max_S) max_S=pvfmm::fabs<Real_t>(S[i][i]);
        }
        for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
        M=V.Transpose()*S;
        break;
      }
      case DC2DE1_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_l2l->ker_dim;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
        size_t n_ch=check_surf.size()/3;
        std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
        size_t n_eq=equiv_surf.size()/3;
        Matrix<Real_t> M_e2c(n_eq*ker_dim[0],n_ch*ker_dim[1]);
        kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_eq, &check_surf[0], n_ch, &(M_e2c[0][0]));
        Matrix<Real_t> U,S,V;
        M_e2c.SVD(U,S,V);
        M=U.Transpose();
        break;
      }
      case U2U_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2m->ker_dim;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> check_surf=u_check_surf(MultipoleOrder(),c,level);
        size_t n_uc=check_surf.size()/3;
        Real_t s=pvfmm::pow<Real_t>(0.5,(level+2));
        int* coord=interac_list.RelativeCoord(type,mat_indx);
        Real_t child_coord[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
        std::vector<Real_t> equiv_surf=u_equiv_surf(MultipoleOrder(),child_coord,level+1);
        size_t n_ue=equiv_surf.size()/3;
        Matrix<Real_t> M_ce2c(n_ue*ker_dim[0],n_uc*ker_dim[1]);
        kernel->k_m2m->BuildMatrix(&equiv_surf[0], n_ue,
                               &check_surf[0], n_uc, &(M_ce2c[0][0]));
        Matrix<Real_t>& M_c2e0 = Precomp(level, UC2UE0_Type, 0);
        Matrix<Real_t>& M_c2e1 = Precomp(level, UC2UE1_Type, 0);
        M=(M_ce2c*M_c2e0)*M_c2e1;
        break;
      }
      case D2D_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_l2l->ker_dim;
        Real_t s=pvfmm::pow<Real_t>(0.5,level+1);
        int* coord=interac_list.RelativeCoord(type,mat_indx);
        Real_t c[3]={(coord[0]+1)*s,(coord[1]+1)*s,(coord[2]+1)*s};
        std::vector<Real_t> check_surf=d_check_surf(MultipoleOrder(),c,level);
        size_t n_dc=check_surf.size()/3;
        Real_t parent_coord[3]={0,0,0};
        std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),parent_coord,level-1);
        size_t n_de=equiv_surf.size()/3;
        Matrix<Real_t> M_pe2c(n_de*ker_dim[0],n_dc*ker_dim[1]);
        kernel->k_l2l->BuildMatrix(&equiv_surf[0], n_de, &check_surf[0], n_dc, &(M_pe2c[0][0]));
        Matrix<Real_t> M_c2e0=Precomp(level-1,DC2DE0_Type,0);
        Matrix<Real_t> M_c2e1=Precomp(level-1,DC2DE1_Type,0);
        if(ScaleInvar()) {
          Permutation<Real_t> ker_perm=this->kernel->k_l2l->perm_vec[C_Perm+Scaling];
          Vector<Real_t> scal_exp=this->kernel->k_l2l->trg_scal;
          Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
          M_c2e0=P*M_c2e0;
        }
        if(ScaleInvar()) {
          Permutation<Real_t> ker_perm=this->kernel->k_l2l->perm_vec[0     +Scaling];
          Vector<Real_t> scal_exp=this->kernel->k_l2l->src_scal;
          Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
          M_c2e1=M_c2e1*P;
        }
        M=M_c2e0*(M_c2e1*M_pe2c);
        break;
      }
      case D2T_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_l2t->ker_dim;
        std::vector<Real_t>& rel_trg_coord=mat->RelativeTrgCoord();
        Real_t r=pvfmm::pow<Real_t>(0.5,level);
        size_t n_trg=rel_trg_coord.size()/3;
        std::vector<Real_t> trg_coord(n_trg*3);
        for(size_t i=0;i<n_trg*COORD_DIM;i++) trg_coord[i]=rel_trg_coord[i]*r;
        Real_t c[3]={0,0,0};
        std::vector<Real_t> equiv_surf=d_equiv_surf(MultipoleOrder(),c,level);
        size_t n_eq=equiv_surf.size()/3;
        {
          M     .Resize(n_eq*ker_dim [0], n_trg*ker_dim [1]);
          kernel->k_l2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
        }
        Matrix<Real_t>& M_c2e0=Precomp(level,DC2DE0_Type,0);
        Matrix<Real_t>& M_c2e1=Precomp(level,DC2DE1_Type,0);
        M=M_c2e0*(M_c2e1*M);
        break;
      }
      case V_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2l->ker_dim;
        int n1=MultipoleOrder()*2;
        int n3 =n1*n1*n1;
        int n3_=n1*n1*(n1/2+1);
        Real_t s=pvfmm::pow<Real_t>(0.5,level);
        int* coord2=interac_list.RelativeCoord(type,mat_indx);
        Real_t coord_diff[3]={coord2[0]*s,coord2[1]*s,coord2[2]*s};
        std::vector<Real_t> r_trg(COORD_DIM,0.0);
        std::vector<Real_t> conv_poten(n3*ker_dim[0]*ker_dim[1]);
        std::vector<Real_t> conv_coord=conv_grid(MultipoleOrder(),coord_diff,level);
        kernel->k_m2l->BuildMatrix(&conv_coord[0],n3,&r_trg[0],1,&conv_poten[0]);
        Matrix<Real_t> M_conv(n3,ker_dim[0]*ker_dim[1],&conv_poten[0],false);
        M_conv=M_conv.Transpose();
        int nnn[3]={n1,n1,n1};
        Real_t *fftw_in, *fftw_out;
        fftw_in  = mem::aligned_new<Real_t>(  n3 *ker_dim[0]*ker_dim[1]*sizeof(Real_t));
        fftw_out = mem::aligned_new<Real_t>(2*n3_*ker_dim[0]*ker_dim[1]*sizeof(Real_t));
#pragma omp critical (FFTW_PLAN)
        {
          if (!vprecomp_fft_flag){
            vprecomp_fftplan = FFTW_t<Real_t>::fft_plan_many_dft_r2c(COORD_DIM, nnn, ker_dim[0]*ker_dim[1],
                (Real_t*)fftw_in, NULL, 1, n3, (typename FFTW_t<Real_t>::cplx*) fftw_out, NULL, 1, n3_);
            vprecomp_fft_flag=true;
          }
        }
        mem::memcopy(fftw_in, &conv_poten[0], n3*ker_dim[0]*ker_dim[1]*sizeof(Real_t));
        FFTW_t<Real_t>::fft_execute_dft_r2c(vprecomp_fftplan, (Real_t*)fftw_in, (typename FFTW_t<Real_t>::cplx*)(fftw_out));
        Matrix<Real_t> M_(2*n3_*ker_dim[0]*ker_dim[1],1,(Real_t*)fftw_out,false);
        M=M_;
        mem::aligned_delete<Real_t>(fftw_in);
        mem::aligned_delete<Real_t>(fftw_out);
        break;
      }
      case V1_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2l->ker_dim;
        size_t mat_cnt =interac_list.ListCount( V_Type);
        for(size_t k=0;k<mat_cnt;k++) Precomp(level, V_Type, k);
  
        const size_t chld_cnt=1UL<<COORD_DIM;
        size_t n1=MultipoleOrder()*2;
        size_t M_dim=n1*n1*(n1/2+1);
        size_t n3=n1*n1*n1;
  
        Vector<Real_t> zero_vec(M_dim*ker_dim[0]*ker_dim[1]*2);
        zero_vec.SetZero();
  
        Vector<Real_t*> M_ptr(chld_cnt*chld_cnt);
        for(size_t i=0;i<chld_cnt*chld_cnt;i++) M_ptr[i]=&zero_vec[0];
  
        int* rel_coord_=interac_list.RelativeCoord(V1_Type, mat_indx);
        for(int j1=0;j1<chld_cnt;j1++)
        for(int j2=0;j2<chld_cnt;j2++){
          int rel_coord[3]={rel_coord_[0]*2-(j1/1)%2+(j2/1)%2,
                            rel_coord_[1]*2-(j1/2)%2+(j2/2)%2,
                            rel_coord_[2]*2-(j1/4)%2+(j2/4)%2};
          for(size_t k=0;k<mat_cnt;k++){
            int* ref_coord=interac_list.RelativeCoord(V_Type, k);
            if(ref_coord[0]==rel_coord[0] &&
               ref_coord[1]==rel_coord[1] &&
               ref_coord[2]==rel_coord[2]){
              Matrix<Real_t>& M = this->mat->Mat(level, V_Type, k);
              M_ptr[j2*chld_cnt+j1]=&M[0][0];
              break;
            }
          }
        }
        M.Resize(ker_dim[0]*ker_dim[1]*M_dim, 2*chld_cnt*chld_cnt);
        for(int j=0;j<ker_dim[0]*ker_dim[1]*M_dim;j++){
          for(size_t k=0;k<chld_cnt*chld_cnt;k++){
            M[j][k*2+0]=M_ptr[k][j*2+0]/n3;
            M[j][k*2+1]=M_ptr[k][j*2+1]/n3;
          }
        }
        break;
      }
      case W_Type:
      {
        if(MultipoleOrder()==0) break;
        const int* ker_dim=kernel->k_m2t->ker_dim;
        std::vector<Real_t>& rel_trg_coord=mat->RelativeTrgCoord();
        Real_t s=pvfmm::pow<Real_t>(0.5,level);
        size_t n_trg=rel_trg_coord.size()/3;
        std::vector<Real_t> trg_coord(n_trg*3);
        for(size_t j=0;j<n_trg*COORD_DIM;j++) trg_coord[j]=rel_trg_coord[j]*s;
        int* coord2=interac_list.RelativeCoord(type,mat_indx);
        Real_t c[3]={(Real_t)((coord2[0]+1)*s*0.25),(Real_t)((coord2[1]+1)*s*0.25),(Real_t)((coord2[2]+1)*s*0.25)};
        std::vector<Real_t> equiv_surf=u_equiv_surf(MultipoleOrder(),c,level+1);
        size_t n_eq=equiv_surf.size()/3;
        {
          M     .Resize(n_eq*ker_dim [0],n_trg*ker_dim [1]);
          kernel->k_m2t->BuildMatrix(&equiv_surf[0], n_eq, &trg_coord[0], n_trg, &(M     [0][0]));
        }
        break;
      }
      case BC_Type:
      {
        if(!this->ScaleInvar() || MultipoleOrder()==0) break;
        if(kernel->k_m2l->ker_dim[0]!=kernel->k_m2m->ker_dim[0]) break;
        if(kernel->k_m2l->ker_dim[1]!=kernel->k_l2l->ker_dim[1]) break;
        int ker_dim[2]={kernel->k_m2l->ker_dim[0],kernel->k_m2l->ker_dim[1]};
        size_t mat_cnt_m2m=interac_list.ListCount(U2U_Type);
        size_t n_surf=(6*(MultipoleOrder()-1)*(MultipoleOrder()-1)+2);
        if((M.Dim(0)!=n_surf*ker_dim[0] || M.Dim(1)!=n_surf*ker_dim[1]) && level==0){
          Matrix<Real_t> M_m2m[BC_LEVELS+1];
          Matrix<Real_t> M_m2l[BC_LEVELS+1];
          Matrix<Real_t> M_l2l[BC_LEVELS+1];
          Matrix<Real_t> M_equiv_zero_avg(n_surf*ker_dim[0],n_surf*ker_dim[0]);
          Matrix<Real_t> M_check_zero_avg(n_surf*ker_dim[1],n_surf*ker_dim[1]);
          {
            Matrix<Real_t> M_s2c;
            {
              int ker_dim[2]={kernel->k_m2m->ker_dim[0],kernel->k_m2m->ker_dim[1]};
              M_s2c.ReInit(ker_dim[0],n_surf*ker_dim[1]);
              std::vector<Real_t> uc_coord;
              {
                Real_t c[3]={0,0,0};
                uc_coord=u_check_surf(MultipoleOrder(),c,0);
              }
#pragma omp parallel for schedule(dynamic)
              for(size_t i=0;i<n_surf;i++){
                std::vector<Real_t> M_=cheb_integ<Real_t>(0, &uc_coord[i*3], 1.0, *kernel->k_m2m);
                for(size_t j=0; j<ker_dim[0]; j++)
                  for(int k=0; k<ker_dim[1]; k++)
                    M_s2c[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
              }
            }
  
            Matrix<Real_t>& M_c2e0 = Precomp(level, UC2UE0_Type, 0);
            Matrix<Real_t>& M_c2e1 = Precomp(level, UC2UE1_Type, 0);
            Matrix<Real_t> M_s2e=(M_s2c*M_c2e0)*M_c2e1;
            for(size_t i=0;i<M_s2e.Dim(0);i++) {
              Real_t s=0;
              for(size_t j=0;j<M_s2e.Dim(1);j++) s+=M_s2e[i][j];
              s=1.0/s;
              for(size_t j=0;j<M_s2e.Dim(1);j++) M_s2e[i][j]*=s;
            }
  
            assert(M_equiv_zero_avg.Dim(0)==M_s2e.Dim(1));
            assert(M_equiv_zero_avg.Dim(1)==M_s2e.Dim(1));
            M_equiv_zero_avg.SetZero();
            for(size_t i=0;i<n_surf*ker_dim[0];i++)
              M_equiv_zero_avg[i][i]=1;
            for(size_t i=0;i<n_surf;i++)
              for(size_t k=0;k<ker_dim[0];k++)
                for(size_t j=0;j<n_surf*ker_dim[0];j++)
                  M_equiv_zero_avg[i*ker_dim[0]+k][j]-=M_s2e[k][j];
          }
          {
            M_check_zero_avg.SetZero();
            for(size_t i=0;i<n_surf*ker_dim[1];i++)
              M_check_zero_avg[i][i]+=1;
            for(size_t i=0;i<n_surf;i++)
              for(size_t j=0;j<n_surf;j++)
                for(size_t k=0;k<ker_dim[1];k++)
                  M_check_zero_avg[i*ker_dim[1]+k][j*ker_dim[1]+k]-=1.0/n_surf;
          }
          for(int level=0; level>=-BC_LEVELS; level--){
            {
              this->Precomp(level, D2D_Type, 0);
              Permutation<Real_t>& Pr = this->interac_list.Perm_R(level, D2D_Type, 0);
              Permutation<Real_t>& Pc = this->interac_list.Perm_C(level, D2D_Type, 0);
              M_l2l[-level] = M_check_zero_avg * Pr * this->Precomp(level, D2D_Type, this->interac_list.InteracClass(D2D_Type, 0)) * Pc * M_check_zero_avg;
              assert(M_l2l[-level].Dim(0)>0 && M_l2l[-level].Dim(1)>0);
            }
            for(size_t mat_indx=0; mat_indx<mat_cnt_m2m; mat_indx++){
              this->Precomp(level, U2U_Type, mat_indx);
              Permutation<Real_t>& Pr = this->interac_list.Perm_R(level, U2U_Type, mat_indx);
              Permutation<Real_t>& Pc = this->interac_list.Perm_C(level, U2U_Type, mat_indx);
              Matrix<Real_t> M = Pr * this->Precomp(level, U2U_Type, this->interac_list.InteracClass(U2U_Type, mat_indx)) * Pc;
              assert(M.Dim(0)>0 && M.Dim(1)>0);
  
              if(mat_indx==0) M_m2m[-level] = M_equiv_zero_avg*M*M_equiv_zero_avg;
              else M_m2m[-level] += M_equiv_zero_avg*M*M_equiv_zero_avg;
            }
            if(!ScaleInvar() || level==0){
              Real_t s=(1UL<<(-level));
              Real_t dc_coord[3]={0,0,0};
              std::vector<Real_t> trg_coord=d_check_surf(MultipoleOrder(), dc_coord, level);
              Matrix<Real_t> M_ue2dc(n_surf*ker_dim[0], n_surf*ker_dim[1]); M_ue2dc.SetZero();
  
              for(int x0=-2;x0<4;x0++)
              for(int x1=-2;x1<4;x1++)
              for(int x2=-2;x2<4;x2++)
              if(abs(x0)>1 || abs(x1)>1 || abs(x2)>1){
                Real_t ue_coord[3]={x0*s, x1*s, x2*s};
                std::vector<Real_t> src_coord=u_equiv_surf(MultipoleOrder(), ue_coord, level);
                Matrix<Real_t> M_tmp(n_surf*ker_dim[0], n_surf*ker_dim[1]);
                kernel->k_m2l->BuildMatrix(&src_coord[0], n_surf, &trg_coord[0], n_surf, &(M_tmp[0][0]));
                M_ue2dc+=M_tmp;
              }
              M_m2l[-level]=M_equiv_zero_avg*M_ue2dc * M_check_zero_avg;
            }else{
              M_m2l[-level]=M_equiv_zero_avg * M_m2l[-level-1] * M_check_zero_avg;
              if(ScaleInvar()) {
                Permutation<Real_t> ker_perm=this->kernel->k_m2l->perm_vec[0     +Scaling];
                Vector<Real_t> scal_exp=this->kernel->k_m2l->src_scal;
                for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
                Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
                M_m2l[-level]=P*M_m2l[-level];
              }
              if(ScaleInvar()) {
                Permutation<Real_t> ker_perm=this->kernel->k_m2l->perm_vec[C_Perm+Scaling];
                Vector<Real_t> scal_exp=this->kernel->k_m2l->trg_scal;
                for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
                Permutation<Real_t> P=equiv_surf_perm(MultipoleOrder(), Scaling, ker_perm, &scal_exp);
                M_m2l[-level]=M_m2l[-level]*P;
              }
            }
          }
          for(int level=-BC_LEVELS;level<=0;level++){
            if(level==-BC_LEVELS) M = M_m2l[-level];
            else                  M = M_equiv_zero_avg * (M_m2l[-level] + M_m2m[-level]*M*M_l2l[-level]) * M_check_zero_avg;
          }
          if(kernel->k_m2l->vol_poten){
            Matrix<Real_t> M_far;
            {
              std::vector<Real_t> dc_coord;
              {
                Real_t c[3]={1.0,1.0,1.0};
                dc_coord=d_check_surf(MultipoleOrder(),c,0);
              }
              Matrix<Real_t> M_near(ker_dim[0],n_surf*ker_dim[1]);
#pragma omp parallel for schedule(dynamic)
              for(size_t i=0;i<n_surf;i++) {
                std::vector<Real_t> M_=cheb_integ<Real_t>(0, &dc_coord[i*3], 3.0, *kernel->k_m2l);
                for(size_t j=0; j<ker_dim[0]; j++)
                  for(int k=0; k<ker_dim[1]; k++)
                    M_near[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
              }
              {
                Matrix<Real_t> M_analytic(ker_dim[0],n_surf*ker_dim[1]); M_analytic.SetZero();
                kernel->k_m2l->vol_poten(&dc_coord[0],n_surf,&M_analytic[0][0]);
                M_far=M_analytic-M_near;
              }
            }
            {
              for(size_t i=0;i<n_surf;i++)
                for(size_t k=0;k<ker_dim[0];k++)
                  for(size_t j=0;j<n_surf*ker_dim[1];j++)
                    M[i*ker_dim[0]+k][j]+=M_far[k][j];
            }
          }
          {
            std::vector<Real_t> corner_pts;
            corner_pts.push_back(0); corner_pts.push_back(0); corner_pts.push_back(0);
            corner_pts.push_back(1); corner_pts.push_back(0); corner_pts.push_back(0);
            corner_pts.push_back(0); corner_pts.push_back(1); corner_pts.push_back(0);
            corner_pts.push_back(0); corner_pts.push_back(0); corner_pts.push_back(1);
            corner_pts.push_back(0); corner_pts.push_back(1); corner_pts.push_back(1);
            corner_pts.push_back(1); corner_pts.push_back(0); corner_pts.push_back(1);
            corner_pts.push_back(1); corner_pts.push_back(1); corner_pts.push_back(0);
            corner_pts.push_back(1); corner_pts.push_back(1); corner_pts.push_back(1);
            size_t n_corner=corner_pts.size()/COORD_DIM;
            Real_t c[3]={0,0,0};
            std::vector<Real_t> up_equiv_surf=u_equiv_surf(MultipoleOrder(),c,0);
            std::vector<Real_t> dn_equiv_surf=d_equiv_surf(MultipoleOrder(),c,0);
            std::vector<Real_t> dn_check_surf=d_check_surf(MultipoleOrder(),c,0);
  
            Matrix<Real_t> M_err;
            {
              {
                Matrix<Real_t> M_e2pt(n_surf*kernel->k_l2l->ker_dim[0],n_corner*kernel->k_l2l->ker_dim[1]);
                kernel->k_l2l->BuildMatrix(&dn_equiv_surf[0], n_surf,
                                              &corner_pts[0], n_corner, &(M_e2pt[0][0]));
                Matrix<Real_t>& M_dc2de0 = Precomp(0, DC2DE0_Type, 0);
                Matrix<Real_t>& M_dc2de1 = Precomp(0, DC2DE1_Type, 0);
                M_err=(M*M_dc2de0)*(M_dc2de1*M_e2pt);
              }
              for(size_t k=0;k<n_corner;k++) {
                for(int j0=-1;j0<=1;j0++)
                for(int j1=-1;j1<=1;j1++)
                for(int j2=-1;j2<=1;j2++){
                  Real_t pt_c[3]={corner_pts[k*COORD_DIM+0]-j0,
                                      corner_pts[k*COORD_DIM+1]-j1,
                                      corner_pts[k*COORD_DIM+2]-j2};
                  if(pvfmm::fabs<Real_t>(pt_c[0]-0.5)>1.0 || pvfmm::fabs<Real_t>(pt_c[1]-0.5)>1.0 || pvfmm::fabs<Real_t>(pt_c[2]-0.5)>1.0){
                    Matrix<Real_t> M_e2pt(n_surf*ker_dim[0],ker_dim[1]);
                    kernel->k_m2l->BuildMatrix(&up_equiv_surf[0], n_surf,
                                                    &pt_c[0], 1, &(M_e2pt[0][0]));
                    for(size_t i=0;i<M_e2pt.Dim(0);i++)
                      for(size_t j=0;j<M_e2pt.Dim(1);j++)
                        M_err[i][k*ker_dim[1]+j]+=M_e2pt[i][j];
                  }
                }
              }
              if(kernel->k_m2l->vol_poten) {
                Matrix<Real_t> M_analytic(ker_dim[0],n_corner*ker_dim[1]); M_analytic.SetZero();
                kernel->k_m2l->vol_poten(&corner_pts[0],n_corner,&M_analytic[0][0]);
                for(size_t j=0;j<n_surf;j++)
                for(size_t k=0;k<ker_dim[0];k++)
                for(size_t i=0;i<M_err.Dim(1);i++){
                  M_err[j*ker_dim[0]+k][i]-=M_analytic[k][i];
                }
              }
            }
  
            Matrix<Real_t> M_grad(M_err.Dim(0),n_surf*ker_dim[1]);
            for(size_t i=0;i<M_err.Dim(0);i++)
            for(size_t k=0;k<ker_dim[1];k++)
            for(size_t j=0;j<n_surf;j++){
              M_grad[i][j*ker_dim[1]+k]=  M_err[i][0*ker_dim[1]+k]
                                        +(M_err[i][1*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+0]
                                        +(M_err[i][2*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+1]
                                        +(M_err[i][3*ker_dim[1]+k]-M_err[i][0*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+2]
                                        +(M_err[i][4*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][2*ker_dim[1]+k]-M_err[i][3*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+1]*dn_check_surf[j*COORD_DIM+2]
                                        +(M_err[i][5*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][1*ker_dim[1]+k]-M_err[i][3*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+2]*dn_check_surf[j*COORD_DIM+0]
                                        +(M_err[i][6*ker_dim[1]+k]+M_err[i][0*ker_dim[1]+k]-M_err[i][1*ker_dim[1]+k]-M_err[i][2*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+0]*dn_check_surf[j*COORD_DIM+1]
                                        +(M_err[i][7*ker_dim[1]+k]+M_err[i][1*ker_dim[1]+k]+M_err[i][2*ker_dim[1]+k]+M_err[i][3*ker_dim[1]+k]
					  -M_err[i][0*ker_dim[1]+k]-M_err[i][4*ker_dim[1]+k]-M_err[i][5*ker_dim[1]+k]-M_err[i][6*ker_dim[1]+k])*dn_check_surf[j*COORD_DIM+0]*dn_check_surf[j*COORD_DIM+1]*dn_check_surf[j*COORD_DIM+2];
            }
            M-=M_grad;
          }
          if(!this->ScaleInvar()) {
            Mat_Type type=D2D_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
            type=U2U_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
            type=DC2DE0_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
            type=DC2DE1_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
            type=UC2UE0_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
            type=UC2UE1_Type;
            for(int l=-BC_LEVELS;l<0;l++)
            for(size_t indx=0;indx<this->interac_list.ListCount(type);indx++){
              Matrix<Real_t>& M=this->mat->Mat(l, type, indx);
              M.Resize(0,0);
            }
          }
        }
        break;
      }
      default:
        break;
    }
#pragma omp critical (PRECOMP_MATRIX_PTS)
    if(M_.Dim(0)==0 && M_.Dim(1)==0){
      M_=M;
    }
    return M_;
  }

  void FFT_UpEquiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& fft_vec, Vector<Real_t>& fft_scal,
		   Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<COORD_DIM;
    size_t fftsize_in =2*n3_*chld_cnt*ker_dim0*dof;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        Real_t c[3]={0,0,0};
        Vector<Real_t> surf=surface(m, c, (Real_t)(m-1), 0);
        map.Resize(surf.Dim()/COORD_DIM);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m-1-surf[i*3]+0.5))+((size_t)(m-1-surf[i*3+1]+0.5))*n1+((size_t)(m-1-surf[i*3+2]+0.5))*n2;
      }
    }
    {
      if(!vlist_fft_flag){
        int nnn[3]={(int)n1,(int)n1,(int)n1};
        void *fftw_in, *fftw_out;
        fftw_in  = mem::aligned_new<Real_t>(  n3 *ker_dim0*chld_cnt);
        fftw_out = mem::aligned_new<Real_t>(2*n3_*ker_dim0*chld_cnt);
        vlist_fftplan = FFTW_t<Real_t>::fft_plan_many_dft_r2c(COORD_DIM,nnn,ker_dim0*chld_cnt,
            (Real_t*)fftw_in, NULL, 1, n3, (typename FFTW_t<Real_t>::cplx*)(fftw_out),NULL, 1, n3_);
        mem::aligned_delete<Real_t>((Real_t*)fftw_in );
        mem::aligned_delete<Real_t>((Real_t*)fftw_out);
        vlist_fft_flag=true;
      }
    }
    {
      size_t n_in = fft_vec.Dim();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_in*(pid  ))/omp_p;
        size_t node_end  =(n_in*(pid+1))/omp_p;
        Vector<Real_t> buffer(fftsize_in, &buffer_[fftsize_in*pid], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Matrix<Real_t>  upward_equiv(chld_cnt,n*ker_dim0*dof,&input_data[0] + fft_vec[node_idx],false);
          Vector<Real_t> upward_equiv_fft(fftsize_in, &output_data[fftsize_in *node_idx], false);
          upward_equiv_fft.SetZero();
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            for(int j1=0;j1<dof;j1++)
            for(int j0=0;j0<(int)chld_cnt;j0++)
            for(int i=0;i<ker_dim0;i++)
              upward_equiv_fft[idx+(j0+(i+j1*ker_dim0)*chld_cnt)*n3]=upward_equiv[j0][ker_dim0*(n*j1+k)+i]*fft_scal[ker_dim0*node_idx+i];
          }
          for(int i=0;i<dof;i++)
            FFTW_t<Real_t>::fft_execute_dft_r2c(vlist_fftplan, (Real_t*)&upward_equiv_fft[i*  n3 *ker_dim0*chld_cnt],
                                        (typename FFTW_t<Real_t>::cplx*)&buffer          [i*2*n3_*ker_dim0*chld_cnt]);
#ifndef FFTW3_MKL
          double add, mul, fma;
          FFTW_t<Real_t>::fftw_flops(vlist_fftplan, &add, &mul, &fma);
#endif
          for(int i=0;i<ker_dim0*dof;i++)
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            upward_equiv_fft[2*(chld_cnt*(n3_*i+j)+k)+0]=buffer[2*(n3_*(chld_cnt*i+k)+j)+0];
            upward_equiv_fft[2*(chld_cnt*(n3_*i+j)+k)+1]=buffer[2*(n3_*(chld_cnt*i+k)+j)+1];
          }
        }
      }
    }
  }
  
  void FFT_Check2Equiv(size_t dof, size_t m, size_t ker_dim1, Vector<size_t>& ifft_vec, Vector<Real_t>& ifft_scal,
		       Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_) {
    size_t n1=m*2;
    size_t n2=n1*n1;
    size_t n3=n1*n2;
    size_t n3_=n2*(n1/2+1);
    size_t chld_cnt=1UL<<COORD_DIM;
    size_t fftsize_out=2*n3_*dof*ker_dim1*chld_cnt;
    int omp_p=omp_get_max_threads();
    size_t n=6*(m-1)*(m-1)+2;
    static Vector<size_t> map;
    {
      size_t n_old=map.Dim();
      if(n_old!=n){
        Real_t c[3]={0,0,0};
        Vector<Real_t> surf=surface(m, c, (Real_t)(m-1), 0);
        map.Resize(surf.Dim()/COORD_DIM);
        for(size_t i=0;i<map.Dim();i++)
          map[i]=((size_t)(m*2-0.5-surf[i*3]))+((size_t)(m*2-0.5-surf[i*3+1]))*n1+((size_t)(m*2-0.5-surf[i*3+2]))*n2;
      }
    }
    {
      if(!vlist_ifft_flag){
        int nnn[3]={(int)n1,(int)n1,(int)n1};
        Real_t *fftw_in, *fftw_out;
        fftw_in  = mem::aligned_new<Real_t>(2*n3_*ker_dim1*chld_cnt);
        fftw_out = mem::aligned_new<Real_t>(  n3 *ker_dim1*chld_cnt);
        vlist_ifftplan = FFTW_t<Real_t>::fft_plan_many_dft_c2r(COORD_DIM,nnn,ker_dim1*chld_cnt,
            (typename FFTW_t<Real_t>::cplx*)fftw_in, NULL, 1, n3_, (Real_t*)(fftw_out),NULL, 1, n3);
        mem::aligned_delete<Real_t>(fftw_in);
        mem::aligned_delete<Real_t>(fftw_out);
        vlist_ifft_flag=true;
      }
    }
    {
      assert(buffer_.Dim()>=2*fftsize_out*omp_p);
      size_t n_out=ifft_vec.Dim();
#pragma omp parallel for
      for(int pid=0; pid<omp_p; pid++){
        size_t node_start=(n_out*(pid  ))/omp_p;
        size_t node_end  =(n_out*(pid+1))/omp_p;
        Vector<Real_t> buffer0(fftsize_out, &buffer_[fftsize_out*(2*pid+0)], false);
        Vector<Real_t> buffer1(fftsize_out, &buffer_[fftsize_out*(2*pid+1)], false);
        for(size_t node_idx=node_start; node_idx<node_end; node_idx++){
          Vector<Real_t> dnward_check_fft(fftsize_out, &input_data[fftsize_out*node_idx], false);
          Vector<Real_t> dnward_equiv(ker_dim1*n*dof*chld_cnt,&output_data[0] + ifft_vec[node_idx],false);
          for(int i=0;i<ker_dim1*dof;i++)
          for(size_t j=0;j<n3_;j++)
          for(size_t k=0;k<chld_cnt;k++){
            buffer0[2*(n3_*(ker_dim1*dof*k+i)+j)+0]=dnward_check_fft[2*(chld_cnt*(n3_*i+j)+k)+0];
            buffer0[2*(n3_*(ker_dim1*dof*k+i)+j)+1]=dnward_check_fft[2*(chld_cnt*(n3_*i+j)+k)+1];
          }
          for(int i=0;i<dof;i++)
            FFTW_t<Real_t>::fft_execute_dft_c2r(vlist_ifftplan, (typename FFTW_t<Real_t>::cplx*)&buffer0[i*2*n3_*ker_dim1*chld_cnt],
						(Real_t*)&buffer1[i*  n3 *ker_dim1*chld_cnt]);
#ifndef FFTW3_MKL
          double add, mul, fma;
          FFTW_t<Real_t>::fftw_flops(vlist_ifftplan, &add, &mul, &fma);
#endif
          for(size_t k=0;k<n;k++){
            size_t idx=map[k];
            for(int j1=0;j1<dof;j1++)
            for(int j0=0;j0<(int)chld_cnt;j0++)
            for(int i=0;i<ker_dim1;i++)
              dnward_equiv[ker_dim1*(n*(dof*j0+j1)+k)+i]+=buffer1[idx+(i+(j1+j0*dof)*ker_dim1)*n3]*ifft_scal[ker_dim1*node_idx+i];
          }
        }
      }
    }
  }
  
 public:

  struct PackedData{
    size_t len;
    Matrix<Real_t>* ptr;
    Vector<size_t> cnt;
    Vector<size_t> dsp;
  };
  struct InteracData{
    Vector<size_t> in_node;
    Vector<size_t> scal_idx;
    Vector<Real_t> coord_shift;
    Vector<size_t> interac_cnt;
    Vector<size_t> interac_dsp;
    Vector<size_t> interac_cst;
    Vector<Real_t> scal[4*MAX_DEPTH];
    Matrix<Real_t> M[4];
  };
  struct ptSetupData{
    int level;
    const Kernel<Real_t>* kernel;
    PackedData src_coord;
    PackedData src_value;
    PackedData srf_coord;
    PackedData srf_value;
    PackedData trg_coord;
    PackedData trg_value;
    InteracData interac_data;
  };

  class FMMData: public FMM_Data<Real_t>{
   public:
    ~FMMData(){}
    FMM_Data<Real_t>* NewData(){return mem::aligned_new<FMMData>();}
  };

  Vector<char> dev_buffer;
  Vector<char> staging_buffer;

  FMM_Pts(mem::MemoryManager* mem_mgr_=NULL): mem_mgr(mem_mgr_),
             vprecomp_fft_flag(false), vlist_fft_flag(false),
               vlist_ifft_flag(false), mat(NULL), kernel(NULL){};

  ~FMM_Pts() {
    if(mat!=NULL){
      delete mat;
      mat=NULL;
    }
    if(vprecomp_fft_flag) FFTW_t<Real_t>::fft_destroy_plan(vprecomp_fftplan);
    {
      if(vlist_fft_flag ) FFTW_t<Real_t>::fft_destroy_plan(vlist_fftplan );
      if(vlist_ifft_flag) FFTW_t<Real_t>::fft_destroy_plan(vlist_ifftplan);
      vlist_fft_flag =false;
      vlist_ifft_flag=false;
    }
  }

  void Initialize(int mult_order, const Kernel<Real_t>* kernel_) {
    Profile::Tic("InitFMM_Pts",true);{
    int rank=0;
    bool verbose=false;
#ifndef NDEBUG
#ifdef __VERBOSE__
    if(!rank) verbose=true;
#endif
#endif
    if(kernel_) kernel_->Initialize(verbose);
    multipole_order=mult_order;
    kernel=kernel_;
    assert(kernel!=NULL);
    bool save_precomp=false;
    mat=new PrecompMat<Real_t>(ScaleInvar());
    if(this->mat_fname.size()==0){
      std::stringstream st;
      st<<PVFMM_PRECOMP_DATA_PATH;
      if(!st.str().size()){
        char* pvfmm_dir = getenv ("PVFMM_DIR");
        if(pvfmm_dir) st<<pvfmm_dir;
      }
#ifndef STAT_MACROS_BROKEN
      if(st.str().size()){
        struct stat stat_buff;
        if(stat(st.str().c_str(), &stat_buff) || !S_ISDIR(stat_buff.st_mode)){
          std::cout<<"error: path not found: "<<st.str()<<'\n';
          exit(0);
        }
      }
#endif
      if(st.str().size()) st<<'/';
      st<<"Precomp_"<<kernel->ker_name.c_str()<<"_m"<<mult_order;
      if(sizeof(Real_t)==8) st<<"";
      else if(sizeof(Real_t)==4) st<<"_f";
      else st<<"_t"<<sizeof(Real_t);
      st<<".data";
      this->mat_fname=st.str();
      save_precomp=true;
    }
    this->mat->LoadFile(mat_fname.c_str());
    interac_list.Initialize(COORD_DIM, this->mat);
    Profile::Tic("PrecompUC2UE",false,4);
    this->PrecompAll(UC2UE0_Type);
    this->PrecompAll(UC2UE1_Type);
    Profile::Toc();
    Profile::Tic("PrecompDC2DE",false,4);
    this->PrecompAll(DC2DE0_Type);
    this->PrecompAll(DC2DE1_Type);
    Profile::Toc();
    Profile::Tic("PrecompBC",false,4);
    this->PrecompAll(BC_Type,0);
    Profile::Toc();
    Profile::Tic("PrecompU2U",false,4);
    this->PrecompAll(U2U_Type);
    Profile::Toc();
    Profile::Tic("PrecompD2D",false,4);
    this->PrecompAll(D2D_Type);
    Profile::Toc();
    if(save_precomp){
      Profile::Tic("Save2File",false,4);
      if(!rank){
        FILE* f=fopen(this->mat_fname.c_str(),"r");
        if(f==NULL) { //File does not exists.
          this->mat->Save2File(this->mat_fname.c_str());
        }else fclose(f);
      }
      Profile::Toc();
    }
    Profile::Tic("PrecompV",false,4);
    this->PrecompAll(V_Type);
    Profile::Toc();
    Profile::Tic("PrecompV1",false,4);
    this->PrecompAll(V1_Type);
    Profile::Toc();
    }Profile::Toc();
  }
  
  int MultipoleOrder(){return multipole_order;}

  bool ScaleInvar(){return kernel->scale_invar;}

  void CollectNodeData(FMMTree_t* tree, std::vector<FMMNode_t*>& node, std::vector<Matrix<Real_t> >& buff_list, std::vector<Vector<FMMNode_t*> >& n_list,
			       std::vector<std::vector<Vector<Real_t>* > > vec_list = std::vector<std::vector<Vector<Real_t>* > >(0)) {
    if(buff_list.size()<7) buff_list.resize(7);
    if(   n_list.size()<7)    n_list.resize(7);
    if( vec_list.size()<7)  vec_list.resize(7);
    int omp_p=omp_get_max_threads();
    if(node.size()==0) return;
    {
      int indx=0;
      size_t vec_sz;
      {
        Matrix<Real_t>& M_uc2ue = this->interac_list.ClassMat(0, UC2UE1_Type, 0);
        vec_sz=M_uc2ue.Dim(1);
      }
      std::vector< FMMNode_t* > node_lst;
      {
        node_lst.clear();
        std::vector<std::vector< FMMNode_t* > > node_lst_(MAX_DEPTH+1);
        FMMNode_t* r_node=NULL;
        for(size_t i=0;i<node.size();i++){
          if(!node[i]->IsLeaf()){
            node_lst_[node[i]->depth].push_back(node[i]);
          }else{
            node[i]->pt_cnt[0]+=node[i]-> src_coord.Dim()/COORD_DIM;
            node[i]->pt_cnt[0]+=node[i]->surf_coord.Dim()/COORD_DIM;
            if(node[i]->IsGhost()) node[i]->pt_cnt[0]++; // TODO: temporary fix, pt_cnt not known for ghost nodes
          }
          if(node[i]->depth==0) r_node=node[i];
        }
        size_t chld_cnt=1UL<<COORD_DIM;
        for(int i=MAX_DEPTH;i>=0;i--){
          for(size_t j=0;j<node_lst_[i].size();j++){
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=node_lst_[i][j]->Child(k);
              node_lst_[i][j]->pt_cnt[0]+=node->pt_cnt[0];
            }
          }
        }
        for(int i=0;i<=MAX_DEPTH;i++){
          for(size_t j=0;j<node_lst_[i].size();j++){
            if(node_lst_[i][j]->pt_cnt[0])
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=node_lst_[i][j]->Child(k);
              node_lst.push_back(node);
            }
          }
        }
        if(r_node!=NULL) node_lst.push_back(r_node);
        n_list[indx]=node_lst;
      }
      std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMMNode_t* node=node_lst[i];
        Vector<Real_t>& data_vec=node->FMMData()->upward_equiv;
        data_vec.ReInit(vec_sz,NULL,false);
        vec_lst.push_back(&data_vec);
      }
    }
    {
      int indx=1;
      size_t vec_sz;
      {
        Matrix<Real_t>& M_dc2de0 = this->interac_list.ClassMat(0, DC2DE0_Type, 0);
        vec_sz=M_dc2de0.Dim(0);
      }
      std::vector< FMMNode_t* > node_lst;
      {
        node_lst.clear();
        std::vector<std::vector< FMMNode_t* > > node_lst_(MAX_DEPTH+1);
        FMMNode_t* r_node=NULL;
        for(size_t i=0;i<node.size();i++){
          if(!node[i]->IsLeaf()){
            node_lst_[node[i]->depth].push_back(node[i]);
          }else{
            node[i]->pt_cnt[1]+=node[i]->trg_coord.Dim()/COORD_DIM;
          }
          if(node[i]->depth==0) r_node=node[i];
        }
        size_t chld_cnt=1UL<<COORD_DIM;
        for(int i=MAX_DEPTH;i>=0;i--){
          for(size_t j=0;j<node_lst_[i].size();j++){
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=node_lst_[i][j]->Child(k);
              node_lst_[i][j]->pt_cnt[1]+=node->pt_cnt[1];
            }
          }
        }
        for(int i=0;i<=MAX_DEPTH;i++){
          for(size_t j=0;j<node_lst_[i].size();j++){
            if(node_lst_[i][j]->pt_cnt[1])
            for(size_t k=0;k<chld_cnt;k++){
              FMMNode_t* node=node_lst_[i][j]->Child(k);
              node_lst.push_back(node);
            }
          }
        }
        if(r_node!=NULL) node_lst.push_back(r_node);
        n_list[indx]=node_lst;
      }
      std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMMNode_t* node=node_lst[i];
        Vector<Real_t>& data_vec=node->FMMData()->dnward_equiv;
        data_vec.ReInit(vec_sz,NULL,false);
        vec_lst.push_back(&data_vec);
      }
    }
    {
      int indx=2;
      std::vector< FMMNode_t* > node_lst;
      {
        std::vector<std::vector< FMMNode_t* > > node_lst_(MAX_DEPTH+1);
        for(size_t i=0;i<node.size();i++)
          if(!node[i]->IsLeaf())
            node_lst_[node[i]->depth].push_back(node[i]);
        for(int i=0;i<=MAX_DEPTH;i++)
          for(size_t j=0;j<node_lst_[i].size();j++)
            node_lst.push_back(node_lst_[i][j]);
      }
      n_list[indx]=node_lst;
    }
    {
      int indx=3;
      std::vector< FMMNode_t* > node_lst;
      {
        std::vector<std::vector< FMMNode_t* > > node_lst_(MAX_DEPTH+1);
        for(size_t i=0;i<node.size();i++)
          if(!node[i]->IsLeaf() && !node[i]->IsGhost())
            node_lst_[node[i]->depth].push_back(node[i]);
        for(int i=0;i<=MAX_DEPTH;i++)
          for(size_t j=0;j<node_lst_[i].size();j++)
            node_lst.push_back(node_lst_[i][j]);
      }
      n_list[indx]=node_lst;
    }
    {
      int indx=4;
      int src_dof=kernel->ker_dim[0];
      int surf_dof=COORD_DIM+src_dof;
      std::vector< FMMNode_t* > node_lst;
      for(size_t i=0;i<node.size();i++) {
        if(node[i]->IsLeaf()){
          node_lst.push_back(node[i]);
        }else{
          node[i]->src_value.ReInit(0);
          node[i]->surf_value.ReInit(0);
        }
      }
      n_list[indx]=node_lst;
      std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMMNode_t* node=node_lst[i];
        {
          Vector<Real_t>& data_vec=node->src_value;
          size_t vec_sz=(node->src_coord.Dim()/COORD_DIM)*src_dof;
          if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz,NULL,false);
          vec_lst.push_back(&data_vec);
        }
        {
          Vector<Real_t>& data_vec=node->surf_value;
          size_t vec_sz=(node->surf_coord.Dim()/COORD_DIM)*surf_dof;
          if(data_vec.Dim()!=vec_sz) data_vec.ReInit(vec_sz,NULL,false);
          vec_lst.push_back(&data_vec);
        }
      }
    }
    {
      int indx=5;
      int trg_dof=kernel->ker_dim[1];
      std::vector< FMMNode_t* > node_lst;
      for(size_t i=0;i<node.size();i++) {
        if(node[i]->IsLeaf() && !node[i]->IsGhost()){
          node_lst.push_back(node[i]);
        }else{
          node[i]->trg_value.ReInit(0);
        }
      }
      n_list[indx]=node_lst;
      std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMMNode_t* node=node_lst[i];
        {
          Vector<Real_t>& data_vec=node->trg_value;
          size_t vec_sz=(node->trg_coord.Dim()/COORD_DIM)*trg_dof;
          data_vec.ReInit(vec_sz,NULL,false);
          vec_lst.push_back(&data_vec);
        }
      }
    }
    {
      int indx=6;
      std::vector< FMMNode_t* > node_lst;
      for(size_t i=0;i<node.size();i++){
        if(node[i]->IsLeaf()){
          node_lst.push_back(node[i]);
        }else{
          node[i]->src_coord.ReInit(0);
          node[i]->surf_coord.ReInit(0);
          node[i]->trg_coord.ReInit(0);
        }
      }
      n_list[indx]=node_lst;
      std::vector<Vector<Real_t>*>& vec_lst=vec_list[indx];
      for(size_t i=0;i<node_lst.size();i++){
        FMMNode_t* node=node_lst[i];
        {
          Vector<Real_t>& data_vec=node->src_coord;
          vec_lst.push_back(&data_vec);
        }
        {
          Vector<Real_t>& data_vec=node->surf_coord;
          vec_lst.push_back(&data_vec);
        }
        {
          Vector<Real_t>& data_vec=node->trg_coord;
          vec_lst.push_back(&data_vec);
        }
      }
      {
        if(tree->upwd_check_surf.size()==0){
          size_t m=MultipoleOrder();
          tree->upwd_check_surf.resize(MAX_DEPTH);
          tree->upwd_equiv_surf.resize(MAX_DEPTH);
          tree->dnwd_check_surf.resize(MAX_DEPTH);
          tree->dnwd_equiv_surf.resize(MAX_DEPTH);
          for(size_t depth=0;depth<MAX_DEPTH;depth++){
            Real_t c[3]={0.0,0.0,0.0};
            tree->upwd_check_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
            tree->upwd_equiv_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
            tree->dnwd_check_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
            tree->dnwd_equiv_surf[depth].ReInit((6*(m-1)*(m-1)+2)*COORD_DIM);
            tree->upwd_check_surf[depth]=u_check_surf(m,c,depth);
            tree->upwd_equiv_surf[depth]=u_equiv_surf(m,c,depth);
            tree->dnwd_check_surf[depth]=d_check_surf(m,c,depth);
            tree->dnwd_equiv_surf[depth]=d_equiv_surf(m,c,depth);
          }
        }
        for(size_t depth=0;depth<MAX_DEPTH;depth++){
          vec_lst.push_back(&tree->upwd_check_surf[depth]);
          vec_lst.push_back(&tree->upwd_equiv_surf[depth]);
          vec_lst.push_back(&tree->dnwd_check_surf[depth]);
          vec_lst.push_back(&tree->dnwd_equiv_surf[depth]);
        }
      }
    }
    if(buff_list.size()<=vec_list.size()) buff_list.resize(vec_list.size()+1);
    for(size_t indx=0;indx<vec_list.size();indx++){
      Matrix<Real_t>&                  buff=buff_list[indx];
      std::vector<Vector<Real_t>*>& vec_lst= vec_list[indx];
      bool keep_data=(indx==4 || indx==6);
      size_t n_vec=vec_lst.size();
      {
        if(!n_vec) continue;
        if(buff.Dim(0)*buff.Dim(1)>0){
          bool init_buff=false;
          Real_t* buff_start=&buff[0][0];
          Real_t* buff_end=&buff[0][0]+buff.Dim(0)*buff.Dim(1);
#pragma omp parallel for reduction(||:init_buff)
          for(size_t i=0;i<n_vec;i++){
            if(vec_lst[i]->Dim() && (&(*vec_lst[i])[0]<buff_start || &(*vec_lst[i])[0]>=buff_end)){
              init_buff=true;
            }
          }
          if(!init_buff) continue;
        }
      }
  
      std::vector<size_t> vec_size(n_vec);
      std::vector<size_t> vec_disp(n_vec);
      if(n_vec) {
#pragma omp parallel for
        for(size_t i=0;i<n_vec;i++) {
          vec_size[i]=vec_lst[i]->Dim();
        }
        vec_disp[0]=0;
        omp_par::scan(&vec_size[0],&vec_disp[0],n_vec);
      }
      size_t buff_size=vec_size[n_vec-1]+vec_disp[n_vec-1];
      if(!buff_size) continue;
      if(keep_data){
        if(dev_buffer.Dim()<buff_size*sizeof(Real_t)){
          dev_buffer.ReInit(buff_size*sizeof(Real_t)*1.05);
        }
#pragma omp parallel for
        for(size_t i=0;i<n_vec;i++){
          if(&(*vec_lst[i])[0]){
            mem::memcopy(((Real_t*)&dev_buffer[0])+vec_disp[i],&(*vec_lst[i])[0],vec_size[i]*sizeof(Real_t));
          }
        }
      }
      if(buff.Dim(0)*buff.Dim(1)<buff_size){
        buff.ReInit(1,buff_size*1.05);
      }
      if(keep_data){
#pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          size_t a=(buff_size*(tid+0))/omp_p;
          size_t b=(buff_size*(tid+1))/omp_p;
          mem::memcopy(&buff[0][0]+a,((Real_t*)&dev_buffer[0])+a,(b-a)*sizeof(Real_t));
        }
      }
#pragma omp parallel for
      for(size_t i=0;i<n_vec;i++){
        vec_lst[i]->ReInit(vec_size[i],&buff[0][0]+vec_disp[i],false);
      }
    }
  }
  
  void SetupPrecomp(SetupData<Real_t,FMMNode_t>& setup_data){
    if(setup_data.precomp_data==NULL || setup_data.level>MAX_DEPTH) return;
    Profile::Tic("SetupPrecomp",true,25);
    {
      size_t precomp_offset=0;
      int level=setup_data.level;
      Matrix<char>& precomp_data=*setup_data.precomp_data;
      std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;
      for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){
        Mat_Type& interac_type=interac_type_lst[type_indx];
        this->PrecompAll(interac_type, level);
        precomp_offset=this->mat->CompactData(level, interac_type, precomp_data, precomp_offset);
      }
    }
    Profile::Toc();
  }
  
  void SetupInterac(SetupData<Real_t,FMMNode_t>& setup_data){
    int level=setup_data.level;
    std::vector<Mat_Type>& interac_type_lst=setup_data.interac_type;
    std::vector<FMMNode_t*>& nodes_in =setup_data.nodes_in ;
    std::vector<FMMNode_t*>& nodes_out=setup_data.nodes_out;
    Matrix<Real_t>&  input_data=*setup_data. input_data;
    Matrix<Real_t>& output_data=*setup_data.output_data;
    std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;
    std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector;
    size_t n_in =nodes_in .size();
    size_t n_out=nodes_out.size();
    if(setup_data.precomp_data->Dim(0)*setup_data.precomp_data->Dim(1)==0) SetupPrecomp(setup_data);
    Profile::Tic("Interac-Data",true,25);
    Matrix<char>& interac_data=setup_data.interac_data;
    {
      std::vector<size_t> interac_mat;
      std::vector<size_t> interac_cnt;
      std::vector<size_t> interac_blk;
      std::vector<size_t>  input_perm;
      std::vector<size_t> output_perm;
      size_t dof=0, M_dim0=0, M_dim1=0;
      size_t precomp_offset=0;
      size_t buff_size=DEVICE_BUFFER_SIZE*1024l*1024l;
      if(n_out && n_in) for(size_t type_indx=0; type_indx<interac_type_lst.size(); type_indx++){
        Mat_Type& interac_type=interac_type_lst[type_indx];
        size_t mat_cnt=this->interac_list.ListCount(interac_type);
        Matrix<size_t> precomp_data_offset;
        {
          struct HeaderData{
            size_t total_size;
            size_t      level;
            size_t   mat_cnt ;
            size_t  max_depth;
          };
          Matrix<char>& precomp_data=*setup_data.precomp_data;
          char* indx_ptr=precomp_data[0]+precomp_offset;
          HeaderData& header=*(HeaderData*)indx_ptr;indx_ptr+=sizeof(HeaderData);
          precomp_data_offset.ReInit(header.mat_cnt,(1+(2+2)*header.max_depth), (size_t*)indx_ptr, false);
          precomp_offset+=header.total_size;
        }
        Matrix<FMMNode_t*> src_interac_list(n_in ,mat_cnt); src_interac_list.SetZero();
        Matrix<FMMNode_t*> trg_interac_list(n_out,mat_cnt); trg_interac_list.SetZero();
        {
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            if(!nodes_out[i]->IsGhost() && (level==-1 || nodes_out[i]->depth==level)){
              Vector<FMMNode_t*>& lst=nodes_out[i]->interac_list[interac_type];
              mem::memcopy(&trg_interac_list[i][0], &lst[0], lst.Dim()*sizeof(FMMNode_t*));
              assert(lst.Dim()==mat_cnt);
            }
          }
        }
        {
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            for(size_t j=0;j<mat_cnt;j++)
            if(trg_interac_list[i][j]!=NULL){
              trg_interac_list[i][j]->node_id=n_in;
            }
          }
#pragma omp parallel for
          for(size_t i=0;i<n_in ;i++) nodes_in[i]->node_id=i;
#pragma omp parallel for
          for(size_t i=0;i<n_out;i++){
            for(size_t j=0;j<mat_cnt;j++){
              if(trg_interac_list[i][j]!=NULL){
                if(trg_interac_list[i][j]->node_id==n_in){
                  trg_interac_list[i][j]=NULL;
                }else{
                  src_interac_list[trg_interac_list[i][j]->node_id][j]=nodes_out[i];
                }
              }
            }
          }
        }
        Matrix<size_t> interac_dsp(n_out,mat_cnt);
        std::vector<size_t> interac_blk_dsp(1,0);
        {
          dof=1;
          Matrix<Real_t>& M0 = this->interac_list.ClassMat(level, interac_type_lst[0], 0);
          M_dim0=M0.Dim(0); M_dim1=M0.Dim(1);
        }
        {
          size_t vec_size=(M_dim0+M_dim1)*sizeof(Real_t)*dof;
          for(size_t j=0;j<mat_cnt;j++){
            size_t vec_cnt=0;
            for(size_t i=0;i<n_out;i++){
              if(trg_interac_list[i][j]!=NULL) vec_cnt++;
            }
            if(buff_size<vec_cnt*vec_size)
              buff_size=vec_cnt*vec_size;
          }
          size_t interac_dsp_=0;
          for(size_t j=0;j<mat_cnt;j++){
            for(size_t i=0;i<n_out;i++){
              interac_dsp[i][j]=interac_dsp_;
              if(trg_interac_list[i][j]!=NULL) interac_dsp_++;
            }
            if(interac_dsp_*vec_size>buff_size) {
              interac_blk.push_back(j-interac_blk_dsp.back());
              interac_blk_dsp.push_back(j);
  
              size_t offset=interac_dsp[0][j];
              for(size_t i=0;i<n_out;i++) interac_dsp[i][j]-=offset;
              interac_dsp_-=offset;
              assert(interac_dsp_*vec_size<=buff_size);
            }
            interac_mat.push_back(precomp_data_offset[this->interac_list.InteracClass(interac_type,j)][0]);
            interac_cnt.push_back(interac_dsp_-interac_dsp[0][j]);
          }
          interac_blk.push_back(mat_cnt-interac_blk_dsp.back());
          interac_blk_dsp.push_back(mat_cnt);
        }
        {
          size_t vec_size=M_dim0*dof;
          for(size_t i=0;i<n_out;i++) nodes_out[i]->node_id=i;
          for(size_t k=1;k<interac_blk_dsp.size();k++){
            for(size_t i=0;i<n_in ;i++){
              for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
                FMMNode_t* trg_node=src_interac_list[i][j];
                if(trg_node!=NULL && trg_node->node_id<n_out){
                  size_t depth=(this->ScaleInvar()?trg_node->depth:0);
                  input_perm .push_back(precomp_data_offset[j][1+4*depth+0]);
                  input_perm .push_back(precomp_data_offset[j][1+4*depth+1]);
                  input_perm .push_back(interac_dsp[trg_node->node_id][j]*vec_size*sizeof(Real_t));
                  input_perm .push_back((size_t)(& input_vector[i][0][0]- input_data[0]));
                  assert(input_vector[i]->Dim()==vec_size);
                }
              }
            }
          }
        }
        {
          size_t vec_size=M_dim1*dof;
          for(size_t k=1;k<interac_blk_dsp.size();k++){
            for(size_t i=0;i<n_out;i++){
              for(size_t j=interac_blk_dsp[k-1];j<interac_blk_dsp[k];j++){
                if(trg_interac_list[i][j]!=NULL){
                  size_t depth=(this->ScaleInvar()?nodes_out[i]->depth:0);
                  output_perm.push_back(precomp_data_offset[j][1+4*depth+2]);
                  output_perm.push_back(precomp_data_offset[j][1+4*depth+3]);
                  output_perm.push_back(interac_dsp[               i ][j]*vec_size*sizeof(Real_t));
                  output_perm.push_back((size_t)(&output_vector[i][0][0]-output_data[0]));
                  assert(output_vector[i]->Dim()==vec_size);
                }
              }
            }
          }
        }
      }
      if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);
      {
        size_t data_size=sizeof(size_t)*4;
        data_size+=sizeof(size_t)+interac_blk.size()*sizeof(size_t);
        data_size+=sizeof(size_t)+interac_cnt.size()*sizeof(size_t);
        data_size+=sizeof(size_t)+interac_mat.size()*sizeof(size_t);
        data_size+=sizeof(size_t)+ input_perm.size()*sizeof(size_t);
        data_size+=sizeof(size_t)+output_perm.size()*sizeof(size_t);
        if(interac_data.Dim(0)*interac_data.Dim(1)<sizeof(size_t)){
          data_size+=sizeof(size_t);
          interac_data.ReInit(1,data_size);
          ((size_t*)&interac_data[0][0])[0]=sizeof(size_t);
        }else{
          size_t pts_data_size=*((size_t*)&interac_data[0][0]);
          assert(interac_data.Dim(0)*interac_data.Dim(1)>=pts_data_size);
          data_size+=pts_data_size;
          if(data_size>interac_data.Dim(0)*interac_data.Dim(1)){
            Matrix< char> pts_interac_data=interac_data;
            interac_data.ReInit(1,data_size);
            mem::memcopy(&interac_data[0][0],&pts_interac_data[0][0],pts_data_size);
          }
        }
        char* data_ptr=&interac_data[0][0];
        data_ptr+=((size_t*)data_ptr)[0];
        ((size_t*)data_ptr)[0]=data_size; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=   M_dim0; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=   M_dim1; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=      dof; data_ptr+=sizeof(size_t);
        ((size_t*)data_ptr)[0]=interac_blk.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_blk[0], interac_blk.size()*sizeof(size_t));
        data_ptr+=interac_blk.size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]=interac_cnt.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_cnt[0], interac_cnt.size()*sizeof(size_t));
        data_ptr+=interac_cnt.size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]=interac_mat.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_mat[0], interac_mat.size()*sizeof(size_t));
        data_ptr+=interac_mat.size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]= input_perm.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, & input_perm[0],  input_perm.size()*sizeof(size_t));
        data_ptr+= input_perm.size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]=output_perm.size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &output_perm[0], output_perm.size()*sizeof(size_t));
        data_ptr+=output_perm.size()*sizeof(size_t);
      }
    }
    Profile::Toc();
  }
      
};

}//end namespace

#endif //_PVFMM_FMM_PTS_HPP_

