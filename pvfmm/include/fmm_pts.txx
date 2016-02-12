namespace pvfmm{

template <class Real_t>
void FMM_Data<Real_t>::Clear(){
  upward_equiv.Resize(0);
}

template <class FMMNode_t>
FMM_Pts<FMMNode_t>::~FMM_Pts() {
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

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Source2Up(SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Up2UpSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  {
    setup_data.level=level;
    setup_data.kernel=kernel->k_m2m;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=U2U_Type;
    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[0];
    Vector<FMMNode_t*>& nodes_in =n_list[0];
    Vector<FMMNode_t*>& nodes_out=n_list[0];
    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level+1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level  ) && nodes_out[i]->pt_cnt[0]) setup_data.nodes_out.push_back(nodes_out[i]);
  }
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_in [i])->FMMData())->upward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_out[i])->FMMData())->upward_equiv);
  SetupInterac(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Up2Up     (SetupData<Real_t>& setup_data){
  if(!this->MultipoleOrder()) return;
  EvalList(setup_data);
}



template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::PeriodicBC(FMMNode_t* node){
  if(!this->ScaleInvar() || this->MultipoleOrder()==0) return;
  Matrix<Real_t>& M = Precomp(0, BC_Type, 0);
  assert(node->FMMData()->upward_equiv.Dim()>0);
  int dof=1;
  Vector<Real_t>& upward_equiv=node->FMMData()->upward_equiv;
  Vector<Real_t>& dnward_equiv=node->FMMData()->dnward_equiv;
  assert(upward_equiv.Dim()==M.Dim(0)*dof);
  assert(dnward_equiv.Dim()==M.Dim(1)*dof);
  Matrix<Real_t> d_equiv(dof,M.Dim(1),&dnward_equiv[0],false);
  Matrix<Real_t> u_equiv(dof,M.Dim(0),&upward_equiv[0],false);
  Matrix<Real_t>::GEMM(d_equiv,u_equiv,M);
}

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

template <class Real_t>
void VListHadamard(size_t dof, size_t M_dim, size_t ker_dim0, size_t ker_dim1, Vector<size_t>& interac_dsp,
    Vector<size_t>& interac_vec, Vector<Real_t*>& precomp_mat, Vector<Real_t>& fft_in, Vector<Real_t>& fft_out){
  size_t chld_cnt=1UL<<COORD_DIM;
  size_t fftsize_in =M_dim*ker_dim0*chld_cnt*2;
  size_t fftsize_out=M_dim*ker_dim1*chld_cnt*2;
  Real_t* zero_vec0=mem::aligned_new<Real_t>(fftsize_in );
  Real_t* zero_vec1=mem::aligned_new<Real_t>(fftsize_out);
  size_t n_out=fft_out.Dim()/fftsize_out;
#pragma omp parallel for
  for(size_t k=0;k<n_out;k++){
    Vector<Real_t> dnward_check_fft(fftsize_out, &fft_out[k*fftsize_out], false);
    dnward_check_fft.SetZero();
  }
  size_t mat_cnt=precomp_mat.Dim();
  size_t blk1_cnt=interac_dsp.Dim()/mat_cnt;
  const size_t V_BLK_SIZE=V_BLK_CACHE*64/sizeof(Real_t);
  Real_t** IN_ =mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
  Real_t** OUT_=mem::aligned_new<Real_t*>(2*V_BLK_SIZE*blk1_cnt*mat_cnt);
  #pragma omp parallel for
  for(size_t interac_blk1=0; interac_blk1<blk1_cnt*mat_cnt; interac_blk1++){
    size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
    size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
    size_t interac_cnt  = interac_dsp1-interac_dsp0;
    for(size_t j=0;j<interac_cnt;j++){
      IN_ [2*V_BLK_SIZE*interac_blk1 +j]=&fft_in [interac_vec[(interac_dsp0+j)*2+0]];
      OUT_[2*V_BLK_SIZE*interac_blk1 +j]=&fft_out[interac_vec[(interac_dsp0+j)*2+1]];
    }
    IN_ [2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec0;
    OUT_[2*V_BLK_SIZE*interac_blk1 +interac_cnt]=zero_vec1;
  }
  int omp_p=omp_get_max_threads();
#pragma omp parallel for
  for(int pid=0; pid<omp_p; pid++){
    size_t a=( pid   *M_dim)/omp_p;
    size_t b=((pid+1)*M_dim)/omp_p;
    for(int in_dim=0;in_dim<ker_dim0;in_dim++)
    for(int ot_dim=0;ot_dim<ker_dim1;ot_dim++)
    for(size_t     blk1=0;     blk1<blk1_cnt;    blk1++)
    for(size_t        k=a;        k<       b;       k++)
    for(size_t mat_indx=0; mat_indx< mat_cnt;mat_indx++){
      size_t interac_blk1 = blk1*mat_cnt+mat_indx;
      size_t interac_dsp0 = (interac_blk1==0?0:interac_dsp[interac_blk1-1]);
      size_t interac_dsp1 =                    interac_dsp[interac_blk1  ] ;
      size_t interac_cnt  = interac_dsp1-interac_dsp0;
      Real_t** IN = IN_ + 2*V_BLK_SIZE*interac_blk1;
      Real_t** OUT= OUT_+ 2*V_BLK_SIZE*interac_blk1;
      Real_t* M = precomp_mat[mat_indx] + k*chld_cnt*chld_cnt*2 + (ot_dim+in_dim*ker_dim1)*M_dim*128;
      {
        for(size_t j=0;j<interac_cnt;j+=2){
          Real_t* M_   = M;
          Real_t* IN0  = IN [j+0] + (in_dim*M_dim+k)*chld_cnt*2;
          Real_t* IN1  = IN [j+1] + (in_dim*M_dim+k)*chld_cnt*2;
          Real_t* OUT0 = OUT[j+0] + (ot_dim*M_dim+k)*chld_cnt*2;
          Real_t* OUT1 = OUT[j+1] + (ot_dim*M_dim+k)*chld_cnt*2;
#ifdef __SSE__
          if (j+2 < interac_cnt) {
            _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+2] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(IN[j+3] + (in_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+2] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2)), _MM_HINT_T0);
            _mm_prefetch(((char *)(OUT[j+3] + (ot_dim*M_dim+k)*chld_cnt*2) + 64), _MM_HINT_T0);
          }
#endif
          matmult_8x8x2(M_, IN0, IN1, OUT0, OUT1);
        }
      }
    }
  }
  {
    Profile::Add_FLOP(8*8*8*(interac_vec.Dim()/2)*M_dim*ker_dim0*ker_dim1*dof);
  }
  mem::aligned_delete<Real_t*>(IN_ );
  mem::aligned_delete<Real_t*>(OUT_);
  mem::aligned_delete<Real_t>(zero_vec0);
  mem::aligned_delete<Real_t>(zero_vec1);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::V_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  if(level==0) return;
  {
    setup_data.level=level;
    setup_data.kernel=kernel->k_m2l;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=V1_Type;
    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[1];
    Vector<FMMNode_t*>& nodes_in =n_list[2];
    Vector<FMMNode_t*>& nodes_out=n_list[3];
    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level-1 || level==-1) && nodes_in [i]->pt_cnt[0]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level-1 || level==-1) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
  }
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode_t*)((FMMNode_t*)nodes_in [i])->Child(0))->FMMData())->upward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode_t*)((FMMNode_t*)nodes_out[i])->Child(0))->FMMData())->dnward_equiv);
  Real_t eps=1e-10;
  size_t n_in =nodes_in .size();
  size_t n_out=nodes_out.size();
  Profile::Tic("Interac-Data",true,25);
  Matrix<char>& interac_data=setup_data.interac_data;
  if(n_out>0 && n_in >0){
    size_t precomp_offset=0;
    Mat_Type& interac_type=setup_data.interac_type[0];
    size_t mat_cnt=this->interac_list.ListCount(interac_type);
    Matrix<size_t> precomp_data_offset;
    std::vector<size_t> interac_mat;
    std::vector<Real_t*> interac_mat_ptr;
    {
      for(size_t mat_id=0;mat_id<mat_cnt;mat_id++){
        Matrix<Real_t>& M = this->mat->Mat(level, interac_type, mat_id);
        interac_mat_ptr.push_back(&M[0][0]);
      }
    }
    size_t dof;
    size_t m=MultipoleOrder();
    size_t ker_dim0=setup_data.kernel->ker_dim[0];
    size_t ker_dim1=setup_data.kernel->ker_dim[1];
    size_t fftsize;
    {
      size_t n1=m*2;
      size_t n2=n1*n1;
      size_t n3_=n2*(n1/2+1);
      size_t chld_cnt=1UL<<COORD_DIM;
      fftsize=2*n3_*chld_cnt;
      dof=1;
    }
    int omp_p=omp_get_max_threads();
    size_t buff_size=DEVICE_BUFFER_SIZE*1024l*1024l;
    size_t n_blk0=2*fftsize*dof*(ker_dim0*n_in +ker_dim1*n_out)*sizeof(Real_t)/buff_size;
    if(n_blk0==0) n_blk0=1;
    std::vector<std::vector<size_t> >  fft_vec(n_blk0);
    std::vector<std::vector<size_t> > ifft_vec(n_blk0);
    std::vector<std::vector<Real_t> >  fft_scl(n_blk0);
    std::vector<std::vector<Real_t> > ifft_scl(n_blk0);
    std::vector<std::vector<size_t> > interac_vec(n_blk0);
    std::vector<std::vector<size_t> > interac_dsp(n_blk0);
    {
      Matrix<Real_t>&  input_data=*setup_data. input_data;
      Matrix<Real_t>& output_data=*setup_data.output_data;
      std::vector<std::vector<FMMNode_t*> > nodes_blk_in (n_blk0);
      std::vector<std::vector<FMMNode_t*> > nodes_blk_out(n_blk0);
      Vector<Real_t> src_scal=this->kernel->k_m2l->src_scal;
      Vector<Real_t> trg_scal=this->kernel->k_m2l->trg_scal;

      for(size_t i=0;i<n_in;i++) ((FMMNode_t*)nodes_in[i])->node_id=i;
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        size_t blk0_start=(n_out* blk0   )/n_blk0;
        size_t blk0_end  =(n_out*(blk0+1))/n_blk0;
        std::vector<FMMNode_t*>& nodes_in_ =nodes_blk_in [blk0];
        std::vector<FMMNode_t*>& nodes_out_=nodes_blk_out[blk0];
        {
          std::set<void*> nodes_in;
          for(size_t i=blk0_start;i<blk0_end;i++){
            nodes_out_.push_back((FMMNode_t*)nodes_out[i]);
            Vector<FMMNode_t*>& lst=((FMMNode_t*)nodes_out[i])->interac_list[interac_type];
            for(size_t k=0;k<mat_cnt;k++) if(lst[k]!=NULL && lst[k]->pt_cnt[0]) nodes_in.insert(lst[k]);
          }
          for(std::set<void*>::iterator node=nodes_in.begin(); node != nodes_in.end(); node++){
            nodes_in_.push_back((FMMNode_t*)*node);
          }
          size_t  input_dim=nodes_in_ .size()*ker_dim0*dof*fftsize;
          size_t output_dim=nodes_out_.size()*ker_dim1*dof*fftsize;
          size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;
          if(buff_size<(input_dim + output_dim + buffer_dim)*sizeof(Real_t))
            buff_size=(input_dim + output_dim + buffer_dim)*sizeof(Real_t);
        }
        {
          for(size_t i=0;i<nodes_in_ .size();i++) fft_vec[blk0].push_back((size_t)(& input_vector[nodes_in_[i]->node_id][0][0]- input_data[0]));
          for(size_t i=0;i<nodes_out_.size();i++)ifft_vec[blk0].push_back((size_t)(&output_vector[blk0_start   +     i ][0][0]-output_data[0]));
          size_t scal_dim0=src_scal.Dim();
          size_t scal_dim1=trg_scal.Dim();
          fft_scl [blk0].resize(nodes_in_ .size()*scal_dim0);
          ifft_scl[blk0].resize(nodes_out_.size()*scal_dim1);
          for(size_t i=0;i<nodes_in_ .size();i++){
            size_t depth=nodes_in_[i]->depth+1;
            for(size_t j=0;j<scal_dim0;j++){
              fft_scl[blk0][i*scal_dim0+j]=pvfmm::pow<Real_t>(2.0, src_scal[j]*depth);
            }
          }
          for(size_t i=0;i<nodes_out_.size();i++){
            size_t depth=nodes_out_[i]->depth+1;
            for(size_t j=0;j<scal_dim1;j++){
              ifft_scl[blk0][i*scal_dim1+j]=pvfmm::pow<Real_t>(2.0, trg_scal[j]*depth);
            }
          }
        }
      }
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        std::vector<FMMNode_t*>& nodes_in_ =nodes_blk_in [blk0];
        std::vector<FMMNode_t*>& nodes_out_=nodes_blk_out[blk0];
        for(size_t i=0;i<nodes_in_.size();i++) nodes_in_[i]->node_id=i;
        {
          size_t n_blk1=nodes_out_.size()*(2)*sizeof(Real_t)/(64*V_BLK_CACHE);
          if(n_blk1==0) n_blk1=1;
          size_t interac_dsp_=0;
          for(size_t blk1=0;blk1<n_blk1;blk1++){
            size_t blk1_start=(nodes_out_.size()* blk1   )/n_blk1;
            size_t blk1_end  =(nodes_out_.size()*(blk1+1))/n_blk1;
            for(size_t k=0;k<mat_cnt;k++){
              for(size_t i=blk1_start;i<blk1_end;i++){
                Vector<FMMNode_t*>& lst=((FMMNode_t*)nodes_out_[i])->interac_list[interac_type];
                if(lst[k]!=NULL && lst[k]->pt_cnt[0]){
                  interac_vec[blk0].push_back(lst[k]->node_id*fftsize*ker_dim0*dof);
                  interac_vec[blk0].push_back(    i          *fftsize*ker_dim1*dof);
                  interac_dsp_++;
                }
              }
              interac_dsp[blk0].push_back(interac_dsp_);
            }
          }
        }
      }
    }
    {
      size_t data_size=sizeof(size_t)*6;
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        data_size+=sizeof(size_t)+    fft_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+   ifft_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+    fft_scl[blk0].size()*sizeof(Real_t);
        data_size+=sizeof(size_t)+   ifft_scl[blk0].size()*sizeof(Real_t);
        data_size+=sizeof(size_t)+interac_vec[blk0].size()*sizeof(size_t);
        data_size+=sizeof(size_t)+interac_dsp[blk0].size()*sizeof(size_t);
      }
      data_size+=sizeof(size_t)+interac_mat.size()*sizeof(size_t);
      data_size+=sizeof(size_t)+interac_mat_ptr.size()*sizeof(Real_t*);
      if(data_size>interac_data.Dim(0)*interac_data.Dim(1))
        interac_data.ReInit(1,data_size);
      char* data_ptr=&interac_data[0][0];
      ((size_t*)data_ptr)[0]=buff_size; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=        m; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=      dof; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]= ker_dim0; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]= ker_dim1; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]=   n_blk0; data_ptr+=sizeof(size_t);
      ((size_t*)data_ptr)[0]= interac_mat.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_mat[0], interac_mat.size()*sizeof(size_t));
      data_ptr+=interac_mat.size()*sizeof(size_t);
      ((size_t*)data_ptr)[0]= interac_mat_ptr.size(); data_ptr+=sizeof(size_t);
      mem::memcopy(data_ptr, &interac_mat_ptr[0], interac_mat_ptr.size()*sizeof(Real_t*));
      data_ptr+=interac_mat_ptr.size()*sizeof(Real_t*);
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        ((size_t*)data_ptr)[0]= fft_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, & fft_vec[blk0][0],  fft_vec[blk0].size()*sizeof(size_t));
        data_ptr+= fft_vec[blk0].size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]=ifft_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &ifft_vec[blk0][0], ifft_vec[blk0].size()*sizeof(size_t));
        data_ptr+=ifft_vec[blk0].size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]= fft_scl[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, & fft_scl[blk0][0],  fft_scl[blk0].size()*sizeof(Real_t));
        data_ptr+= fft_scl[blk0].size()*sizeof(Real_t);
        ((size_t*)data_ptr)[0]=ifft_scl[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &ifft_scl[blk0][0], ifft_scl[blk0].size()*sizeof(Real_t));
        data_ptr+=ifft_scl[blk0].size()*sizeof(Real_t);
        ((size_t*)data_ptr)[0]=interac_vec[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_vec[blk0][0], interac_vec[blk0].size()*sizeof(size_t));
        data_ptr+=interac_vec[blk0].size()*sizeof(size_t);
        ((size_t*)data_ptr)[0]=interac_dsp[blk0].size(); data_ptr+=sizeof(size_t);
        mem::memcopy(data_ptr, &interac_dsp[blk0][0], interac_dsp[blk0].size()*sizeof(size_t));
        data_ptr+=interac_dsp[blk0].size()*sizeof(size_t);
      }
    }
  }
  Profile::Toc();
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::V_List     (SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  int np=1;
  if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
    return;
  }
  Profile::Tic("Host2Device",false,25);
  int level=setup_data.level;
  size_t buff_size=*((size_t*)&setup_data.interac_data[0][0]);
  typename Vector<char>::Device          buff;
  typename Matrix<char>::Device  interac_data;
  typename Matrix<Real_t>::Device  input_data;
  typename Matrix<Real_t>::Device output_data;
  if(this->dev_buffer.Dim()<buff_size) this->dev_buffer.ReInit(buff_size);
  buff        =       this-> dev_buffer;
  interac_data= setup_data.interac_data;
  input_data  =*setup_data.  input_data;
  output_data =*setup_data. output_data;
  Profile::Toc();
  {
    size_t m, dof, ker_dim0, ker_dim1, n_blk0;
    std::vector<Vector<size_t> >  fft_vec;
    std::vector<Vector<size_t> > ifft_vec;
    std::vector<Vector<Real_t> >  fft_scl;
    std::vector<Vector<Real_t> > ifft_scl;
    std::vector<Vector<size_t> > interac_vec;
    std::vector<Vector<size_t> > interac_dsp;
    Vector<Real_t*> precomp_mat;
    {
      char* data_ptr=&interac_data[0][0];
      buff_size=((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      m        =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      dof      =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      ker_dim0 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      ker_dim1 =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      n_blk0   =((size_t*)data_ptr)[0]; data_ptr+=sizeof(size_t);
      fft_vec .resize(n_blk0);
      ifft_vec.resize(n_blk0);
      fft_scl .resize(n_blk0);
      ifft_scl.resize(n_blk0);
      interac_vec.resize(n_blk0);
      interac_dsp.resize(n_blk0);
      Vector<size_t> interac_mat;
      interac_mat.ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_mat.Dim()*sizeof(size_t);
      Vector<Real_t*> interac_mat_ptr;
      interac_mat_ptr.ReInit(((size_t*)data_ptr)[0],(Real_t**)(data_ptr+sizeof(size_t)),false);
      data_ptr+=sizeof(size_t)+interac_mat_ptr.Dim()*sizeof(Real_t*);
      precomp_mat.Resize(interac_mat_ptr.Dim());
      for(size_t i=0;i<interac_mat_ptr.Dim();i++){
        precomp_mat[i]=interac_mat_ptr[i];
      }
      for(size_t blk0=0;blk0<n_blk0;blk0++){
        fft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+fft_vec[blk0].Dim()*sizeof(size_t);
        ifft_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+ifft_vec[blk0].Dim()*sizeof(size_t);
        fft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+fft_scl[blk0].Dim()*sizeof(Real_t);
        ifft_scl[blk0].ReInit(((size_t*)data_ptr)[0],(Real_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+ifft_scl[blk0].Dim()*sizeof(Real_t);
        interac_vec[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_vec[blk0].Dim()*sizeof(size_t);
        interac_dsp[blk0].ReInit(((size_t*)data_ptr)[0],(size_t*)(data_ptr+sizeof(size_t)),false);
        data_ptr+=sizeof(size_t)+interac_dsp[blk0].Dim()*sizeof(size_t);
      }
    }
    int omp_p=omp_get_max_threads();
    size_t M_dim, fftsize;
    {
      size_t n1=m*2;
      size_t n2=n1*n1;
      size_t n3_=n2*(n1/2+1);
      size_t chld_cnt=1UL<<COORD_DIM;
      fftsize=2*n3_*chld_cnt;
      M_dim=n3_;
    }
    for(size_t blk0=0;blk0<n_blk0;blk0++){
      size_t n_in = fft_vec[blk0].Dim();
      size_t n_out=ifft_vec[blk0].Dim();
      size_t  input_dim=n_in *ker_dim0*dof*fftsize;
      size_t output_dim=n_out*ker_dim1*dof*fftsize;
      size_t buffer_dim=2*(ker_dim0+ker_dim1)*dof*fftsize*omp_p;
      Vector<Real_t> fft_in ( input_dim, (Real_t*)&buff[         0                           ],false);
      Vector<Real_t> fft_out(output_dim, (Real_t*)&buff[ input_dim            *sizeof(Real_t)],false);
      Vector<Real_t>  buffer(buffer_dim, (Real_t*)&buff[(input_dim+output_dim)*sizeof(Real_t)],false);
      {
        if(np==1) Profile::Tic("FFT",false,100);
        Vector<Real_t>  input_data_( input_data.dim[0]* input_data.dim[1],  input_data[0], false);
        FFT_UpEquiv(dof, m, ker_dim0,  fft_vec[blk0],  fft_scl[blk0],  input_data_, fft_in, buffer);
        if(np==1) Profile::Toc();
      }
      {
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
        std::cout << "Starting counters new\n";
        if (PAPI_start(EventSet) != PAPI_OK) std::cout << "handle_error3" << std::endl;
#endif
#endif
        if(np==1) Profile::Tic("HadamardProduct",false,100);
        VListHadamard<Real_t>(dof, M_dim, ker_dim0, ker_dim1, interac_dsp[blk0], interac_vec[blk0], precomp_mat, fft_in, fft_out);
        if(np==1) Profile::Toc();
#ifdef PVFMM_HAVE_PAPI
#ifdef __VERBOSE__
        if (PAPI_stop(EventSet, values) != PAPI_OK) std::cout << "handle_error4" << std::endl;
        std::cout << "Stopping counters\n";
#endif
#endif
      }
      {
        if(np==1) Profile::Tic("IFFT",false,100);
        Vector<Real_t> output_data_(output_data.dim[0]*output_data.dim[1], output_data[0], false);
        FFT_Check2Equiv(dof, m, ker_dim1, ifft_vec[blk0], ifft_scl[blk0], fft_out, output_data_, buffer);
        if(np==1) Profile::Toc();
      }
    }
  }
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2DownSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  {
    setup_data.level=level;
    setup_data.kernel=kernel->k_l2l;
    setup_data.interac_type.resize(1);
    setup_data.interac_type[0]=D2D_Type;
    setup_data. input_data=&buff[1];
    setup_data.output_data=&buff[1];
    Vector<FMMNode_t*>& nodes_in =n_list[1];
    Vector<FMMNode_t*>& nodes_out=n_list[1];
    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level-1) && nodes_in [i]->pt_cnt[1]) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level  ) && nodes_out[i]->pt_cnt[1]) setup_data.nodes_out.push_back(nodes_out[i]);
  }
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;
  std::vector<Vector<Real_t>*>&  input_vector=setup_data. input_vector;  input_vector.clear();
  std::vector<Vector<Real_t>*>& output_vector=setup_data.output_vector; output_vector.clear();
  for(size_t i=0;i<nodes_in .size();i++)  input_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_in [i])->FMMData())->dnward_equiv);
  for(size_t i=0;i<nodes_out.size();i++) output_vector.push_back(&((FMMData*)((FMMNode_t*)nodes_out[i])->FMMData())->dnward_equiv);
  SetupInterac(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2Down     (SetupData<Real_t>& setup_data){
  if(!this->MultipoleOrder()) return;
  EvalList(setup_data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::EvalListPts(SetupData<Real_t>& setup_data){
  if(setup_data.kernel->ker_dim[0]*setup_data.kernel->ker_dim[1]==0) return;
  if(setup_data.interac_data.Dim(0)==0 || setup_data.interac_data.Dim(1)==0){
    return;
  }
  bool have_gpu=false;
  Profile::Tic("Host2Device",false,25);
  typename Vector<char>::Device      dev_buff;
  typename Matrix<char>::Device  interac_data;
  typename Matrix<Real_t>::Device  coord_data;
  typename Matrix<Real_t>::Device  input_data;
  typename Matrix<Real_t>::Device output_data;
  size_t ptr_single_layer_kernel=(size_t)NULL;
  size_t ptr_double_layer_kernel=(size_t)NULL;
  dev_buff    =       this-> dev_buffer;
  interac_data= setup_data.interac_data;
  if(setup_data.  coord_data!=NULL) coord_data  =*setup_data.  coord_data;
  if(setup_data.  input_data!=NULL) input_data  =*setup_data.  input_data;
  if(setup_data. output_data!=NULL) output_data =*setup_data. output_data;
  ptr_single_layer_kernel=(size_t)setup_data.kernel->ker_poten;
  ptr_double_layer_kernel=(size_t)setup_data.kernel->dbl_layer_poten;
  Profile::Toc();
  Profile::Tic("DeviceComp",false,20);
  int lock_idx=-1;
  int wait_lock_idx=-1;
  {
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
    ptSetupData data;
    {
      struct PackedSetupData{
        size_t size;
        int level;
        const Kernel<Real_t>* kernel;
        Matrix<Real_t>* src_coord;
        Matrix<Real_t>* src_value;
        Matrix<Real_t>* srf_coord;
        Matrix<Real_t>* srf_value;
        Matrix<Real_t>* trg_coord;
        Matrix<Real_t>* trg_value;
        size_t src_coord_cnt_size; size_t src_coord_cnt_offset;
        size_t src_coord_dsp_size; size_t src_coord_dsp_offset;
        size_t src_value_cnt_size; size_t src_value_cnt_offset;
        size_t src_value_dsp_size; size_t src_value_dsp_offset;
        size_t srf_coord_cnt_size; size_t srf_coord_cnt_offset;
        size_t srf_coord_dsp_size; size_t srf_coord_dsp_offset;
        size_t srf_value_cnt_size; size_t srf_value_cnt_offset;
        size_t srf_value_dsp_size; size_t srf_value_dsp_offset;
        size_t trg_coord_cnt_size; size_t trg_coord_cnt_offset;
        size_t trg_coord_dsp_size; size_t trg_coord_dsp_offset;
        size_t trg_value_cnt_size; size_t trg_value_cnt_offset;
        size_t trg_value_dsp_size; size_t trg_value_dsp_offset;
        size_t          in_node_size; size_t           in_node_offset;
        size_t         scal_idx_size; size_t          scal_idx_offset;
        size_t      coord_shift_size; size_t       coord_shift_offset;
        size_t      interac_cnt_size; size_t       interac_cnt_offset;
        size_t      interac_dsp_size; size_t       interac_dsp_offset;
        size_t      interac_cst_size; size_t       interac_cst_offset;
        size_t scal_dim[4*MAX_DEPTH]; size_t scal_offset[4*MAX_DEPTH];
        size_t            Mdim[4][2]; size_t              M_offset[4];
      };
      typename Matrix<char>::Device& setupdata=interac_data;
      PackedSetupData& pkd_data=*((PackedSetupData*)setupdata[0]);
      data. level=pkd_data. level;
      data.kernel=pkd_data.kernel;
      data.src_coord.ptr=pkd_data.src_coord;
      data.src_value.ptr=pkd_data.src_value;
      data.srf_coord.ptr=pkd_data.srf_coord;
      data.srf_value.ptr=pkd_data.srf_value;
      data.trg_coord.ptr=pkd_data.trg_coord;
      data.trg_value.ptr=pkd_data.trg_value;
      data.src_coord.cnt.ReInit(pkd_data.src_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.src_coord_cnt_offset], false);
      data.src_coord.dsp.ReInit(pkd_data.src_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.src_coord_dsp_offset], false);
      data.src_value.cnt.ReInit(pkd_data.src_value_cnt_size, (size_t*)&setupdata[0][pkd_data.src_value_cnt_offset], false);
      data.src_value.dsp.ReInit(pkd_data.src_value_dsp_size, (size_t*)&setupdata[0][pkd_data.src_value_dsp_offset], false);
      data.srf_coord.cnt.ReInit(pkd_data.srf_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.srf_coord_cnt_offset], false);
      data.srf_coord.dsp.ReInit(pkd_data.srf_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.srf_coord_dsp_offset], false);
      data.srf_value.cnt.ReInit(pkd_data.srf_value_cnt_size, (size_t*)&setupdata[0][pkd_data.srf_value_cnt_offset], false);
      data.srf_value.dsp.ReInit(pkd_data.srf_value_dsp_size, (size_t*)&setupdata[0][pkd_data.srf_value_dsp_offset], false);
      data.trg_coord.cnt.ReInit(pkd_data.trg_coord_cnt_size, (size_t*)&setupdata[0][pkd_data.trg_coord_cnt_offset], false);
      data.trg_coord.dsp.ReInit(pkd_data.trg_coord_dsp_size, (size_t*)&setupdata[0][pkd_data.trg_coord_dsp_offset], false);
      data.trg_value.cnt.ReInit(pkd_data.trg_value_cnt_size, (size_t*)&setupdata[0][pkd_data.trg_value_cnt_offset], false);
      data.trg_value.dsp.ReInit(pkd_data.trg_value_dsp_size, (size_t*)&setupdata[0][pkd_data.trg_value_dsp_offset], false);
      InteracData& intdata=data.interac_data;
      intdata.    in_node.ReInit(pkd_data.    in_node_size, (size_t*)&setupdata[0][pkd_data.    in_node_offset],false);
      intdata.   scal_idx.ReInit(pkd_data.   scal_idx_size, (size_t*)&setupdata[0][pkd_data.   scal_idx_offset],false);
      intdata.coord_shift.ReInit(pkd_data.coord_shift_size, (Real_t*)&setupdata[0][pkd_data.coord_shift_offset],false);
      intdata.interac_cnt.ReInit(pkd_data.interac_cnt_size, (size_t*)&setupdata[0][pkd_data.interac_cnt_offset],false);
      intdata.interac_dsp.ReInit(pkd_data.interac_dsp_size, (size_t*)&setupdata[0][pkd_data.interac_dsp_offset],false);
      intdata.interac_cst.ReInit(pkd_data.interac_cst_size, (size_t*)&setupdata[0][pkd_data.interac_cst_offset],false);
      for(size_t i=0;i<4*MAX_DEPTH;i++){
        intdata.scal[i].ReInit(pkd_data.scal_dim[i], (Real_t*)&setupdata[0][pkd_data.scal_offset[i]],false);
      }
      for(size_t i=0;i<4;i++){
        intdata.M[i].ReInit(pkd_data.Mdim[i][0], pkd_data.Mdim[i][1], (Real_t*)&setupdata[0][pkd_data.M_offset[i]],false);
      }
    }
    {
      InteracData& intdata=data.interac_data;
      typename Kernel<Real_t>::Ker_t single_layer_kernel=(typename Kernel<Real_t>::Ker_t)ptr_single_layer_kernel;
      typename Kernel<Real_t>::Ker_t double_layer_kernel=(typename Kernel<Real_t>::Ker_t)ptr_double_layer_kernel;
      int omp_p=omp_get_max_threads();
#pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){

        Matrix<Real_t> src_coord, src_value;
        Matrix<Real_t> srf_coord, srf_value;
        Matrix<Real_t> trg_coord, trg_value;
        Vector<Real_t> buff;
        { // init buff
          size_t thread_buff_size=dev_buff.dim/sizeof(Real_t)/omp_p;
          buff.ReInit(thread_buff_size, (Real_t*)&dev_buff[tid*thread_buff_size*sizeof(Real_t)], false);
        }

        size_t vcnt=0;
        std::vector<Matrix<Real_t> > vbuff(6);
        { // init vbuff[0:5]
          size_t vdim_=0, vdim[6];
          for(size_t indx=0;indx<6;indx++){
            vdim[indx]=0;
            switch(indx){
              case 0:
                vdim[indx]=intdata.M[0].Dim(0); break;
              case 1:
                assert(intdata.M[0].Dim(1)==intdata.M[1].Dim(0));
                vdim[indx]=intdata.M[0].Dim(1); break;
              case 2:
                vdim[indx]=intdata.M[1].Dim(1); break;
              case 3:
                vdim[indx]=intdata.M[2].Dim(0); break;
              case 4:
                assert(intdata.M[2].Dim(1)==intdata.M[3].Dim(0));
                vdim[indx]=intdata.M[2].Dim(1); break;
              case 5:
                vdim[indx]=intdata.M[3].Dim(1); break;
              default:
                vdim[indx]=0; break;
            }
            vdim_+=vdim[indx];
          }
          if(vdim_){
            vcnt=buff.Dim()/vdim_/2;
            assert(vcnt>0); // Thread buffer is too small
          }

          for(size_t indx=0;indx<6;indx++){ // init vbuff[0:5]
            vbuff[indx].ReInit(vcnt,vdim[indx],&buff[0],false);
            buff.ReInit(buff.Dim()-vdim[indx]*vcnt, &buff[vdim[indx]*vcnt], false);
          }
        }

        size_t trg_a=0, trg_b=0;
        if(intdata.interac_cst.Dim()){ // Determine trg_a, trg_b
          //trg_a=((tid+0)*intdata.interac_cnt.Dim())/omp_p;
          //trg_b=((tid+1)*intdata.interac_cnt.Dim())/omp_p;
          Vector<size_t>& interac_cst=intdata.interac_cst;
          size_t cost=interac_cst[interac_cst.Dim()-1];
          trg_a=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+0))/omp_p)-&interac_cst[0]+1;
          trg_b=std::lower_bound(&interac_cst[0],&interac_cst[interac_cst.Dim()-1],(cost*(tid+1))/omp_p)-&interac_cst[0]+1;
          if(tid==omp_p-1) trg_b=interac_cst.Dim();
          if(tid==0) trg_a=0;
        }
        for(size_t trg0=trg_a;trg0<trg_b;){
          size_t trg1_max=1;
          if(vcnt){ // Find trg1_max
            size_t interac_cnt=intdata.interac_cnt[trg0];
            while(trg0+trg1_max<trg_b){
              interac_cnt+=intdata.interac_cnt[trg0+trg1_max];
              if(interac_cnt>vcnt){
                interac_cnt-=intdata.interac_cnt[trg0+trg1_max];
                break;
              }
              trg1_max++;
            }
            assert(interac_cnt<=vcnt);
            for(size_t k=0;k<6;k++){
              if(vbuff[k].Dim(0)*vbuff[k].Dim(1)){
                vbuff[k].ReInit(interac_cnt,vbuff[k].Dim(1),vbuff[k][0],false);
              }
            }
          }else{
            trg1_max=trg_b-trg0;
          }

          if(intdata.M[0].Dim(0) && intdata.M[0].Dim(1) && intdata.M[1].Dim(0) && intdata.M[1].Dim(1)){ // src mat-vec
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){ // Copy src_value to vbuff[0]
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t src=intdata.in_node[int_id];
                src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                { // Copy src_value to vbuff[0]
                  size_t vdim=vbuff[0].Dim(1);
                  assert(src_value.Dim(1)==vdim);
                  for(size_t j=0;j<vdim;j++) vbuff[0][interac_idx][j]=src_value[0][j];
                }
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[0];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+0];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }

            Matrix<Real_t>::GEMM(vbuff[1],vbuff[0],intdata.M[0]);
            Matrix<Real_t>::GEMM(vbuff[2],vbuff[1],intdata.M[1]);

            interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[2];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+1];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }
          }

          if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){ // init vbuff[3]
            size_t vdim=vbuff[3].Dim(0)*vbuff[3].Dim(1);
            for(size_t i=0;i<vdim;i++) vbuff[3][0][i]=0;
          }

          { // Evaluate kernel functions
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              trg_coord.ReInit(1, data.trg_coord.cnt[trg], &data.trg_coord.ptr[0][0][data.trg_coord.dsp[trg]], false);
              trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t src=intdata.in_node[int_id];
                src_coord.ReInit(1, data.src_coord.cnt[src], &data.src_coord.ptr[0][0][data.src_coord.dsp[src]], false);
                src_value.ReInit(1, data.src_value.cnt[src], &data.src_value.ptr[0][0][data.src_value.dsp[src]], false);
                srf_coord.ReInit(1, data.srf_coord.cnt[src], &data.srf_coord.ptr[0][0][data.srf_coord.dsp[src]], false);
                srf_value.ReInit(1, data.srf_value.cnt[src], &data.srf_value.ptr[0][0][data.srf_value.dsp[src]], false);

                Real_t* vbuff2_ptr=(vbuff[2].Dim(0)*vbuff[2].Dim(1)?vbuff[2][interac_idx]:src_value[0]);
                Real_t* vbuff3_ptr=(vbuff[3].Dim(0)*vbuff[3].Dim(1)?vbuff[3][interac_idx]:trg_value[0]);

                if(src_coord.Dim(1)){
                  { // coord_shift
                    Real_t* shift=&intdata.coord_shift[int_id*COORD_DIM];
                    if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                      size_t vdim=src_coord.Dim(1);
                      Vector<Real_t> new_coord(vdim, &buff[0], false);
                      assert(buff.Dim()>=vdim); // Thread buffer is too small
                      //buff.ReInit(buff.Dim()-vdim, &buff[vdim], false);
                      for(size_t j=0;j<vdim;j+=COORD_DIM){
                        for(size_t k=0;k<COORD_DIM;k++){
                          new_coord[j+k]=src_coord[0][j+k]+shift[k];
                        }
                      }
                      src_coord.ReInit(1, vdim, &new_coord[0], false);
                    }
                  }
                  assert(ptr_single_layer_kernel); // assert(Single-layer kernel is implemented)
                  single_layer_kernel(src_coord[0], src_coord.Dim(1)/COORD_DIM, vbuff2_ptr, 1,
                                      trg_coord[0], trg_coord.Dim(1)/COORD_DIM, vbuff3_ptr, NULL);
                }
                if(srf_coord.Dim(1)){
                  { // coord_shift
                    Real_t* shift=&intdata.coord_shift[int_id*COORD_DIM];
                    if(shift[0]!=0 || shift[1]!=0 || shift[2]!=0){
                      size_t vdim=srf_coord.Dim(1);
                      Vector<Real_t> new_coord(vdim, &buff[0], false);
                      assert(buff.Dim()>=vdim); // Thread buffer is too small
                      //buff.ReInit(buff.Dim()-vdim, &buff[vdim], false);
                      for(size_t j=0;j<vdim;j+=COORD_DIM){
                        for(size_t k=0;k<COORD_DIM;k++){
                          new_coord[j+k]=srf_coord[0][j+k]+shift[k];
                        }
                      }
                      srf_coord.ReInit(1, vdim, &new_coord[0], false);
                    }
                  }
                  assert(ptr_double_layer_kernel); // assert(Double-layer kernel is implemented)
                  double_layer_kernel(srf_coord[0], srf_coord.Dim(1)/COORD_DIM, srf_value[0], 1,
                                      trg_coord[0], trg_coord.Dim(1)/COORD_DIM, vbuff3_ptr, NULL);
                }
                interac_idx++;
              }
            }
          }

          if(intdata.M[2].Dim(0) && intdata.M[2].Dim(1) && intdata.M[3].Dim(0) && intdata.M[3].Dim(1)){ // trg mat-vec
            size_t interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[3];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+2];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                interac_idx++;
              }
            }

            Matrix<Real_t>::GEMM(vbuff[4],vbuff[3],intdata.M[2]);
            Matrix<Real_t>::GEMM(vbuff[5],vbuff[4],intdata.M[3]);

            interac_idx=0;
            for(size_t trg1=0;trg1<trg1_max;trg1++){
              size_t trg=trg0+trg1;
              trg_value.ReInit(1, data.trg_value.cnt[trg], &data.trg_value.ptr[0][0][data.trg_value.dsp[trg]], false);
              for(size_t i=0;i<intdata.interac_cnt[trg];i++){
                size_t int_id=intdata.interac_dsp[trg]+i;
                size_t scal_idx=intdata.scal_idx[int_id];
                { // scaling
                  Matrix<Real_t>& vec=vbuff[5];
                  Vector<Real_t>& scal=intdata.scal[scal_idx*4+3];
                  size_t scal_dim=scal.Dim();
                  if(scal_dim){
                    size_t vdim=vec.Dim(1);
                    for(size_t j=0;j<vdim;j+=scal_dim){
                      for(size_t k=0;k<scal_dim;k++){
                        vec[interac_idx][j+k]*=scal[k];
                      }
                    }
                  }
                }
                { // Add vbuff[5] to trg_value
                  size_t vdim=vbuff[5].Dim(1);
                  assert(trg_value.Dim(1)==vdim);
                  for(size_t i=0;i<vdim;i++) trg_value[0][i]+=vbuff[5][interac_idx][i];
                }
                interac_idx++;
              }
            }
          }

          trg0+=trg1_max;
        }
      }
    }
  }
  Profile::Toc();
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::X_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_s2l;
    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[1];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[1];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) &&  nodes_in [i]->IsLeaf ()) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) &&  nodes_out[i]->pt_cnt[1]                                          && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

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

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=tree->dnwd_check_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->dnward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        if(tnode->IsLeaf() && tnode->pt_cnt[1]<=Nsrf){ // skip: handled in U-list
          interac_cnt.push_back(0);
          continue;
        }
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);

        size_t interac_cnt_=0;
        { // X_Type
          Mat_Type type=X_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(0+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(0+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(0+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::X_List     (SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  //Add X_List contribution.
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::W_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_m2t;
    setup_data. input_data=&buff[0];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[0];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && nodes_in [i]->pt_cnt[0]                                                            ) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

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

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=tree->upwd_equiv_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->upward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      coord.dsp[i]=0;
      coord.cnt[i]=0;
      value.dsp[i]=0;
      value.cnt[i]=0;
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);

        size_t interac_cnt_=0;
        { // W_Type
          Mat_Type type=W_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0){ // Is non-leaf ghost node
            }else if(snode->IsLeaf() && snode->pt_cnt[0]<=Nsrf) continue; // skip: handled in U-list
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(0+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(0+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(0+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::W_List     (SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  //Add W_List contribution.
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::U_ListSetup(SetupData<Real_t>& setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_s2t;
    setup_data. input_data=&buff[4];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[4];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((level==0 || level==-1) && (nodes_in [i]->src_coord.Dim() || nodes_in [i]->surf_coord.Dim()) && nodes_in [i]->IsLeaf()                            ) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((level==0 || level==-1) && (nodes_out[i]->trg_coord.Dim()                                  ) && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

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

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->src_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->src_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->surf_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->surf_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);

    size_t m=this->MultipoleOrder();
    size_t Nsrf=(6*(m-1)*(m-1)+2);
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid]        ;

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);

        size_t interac_cnt_=0;
        { // U0_Type
          Mat_Type type=U0_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // U1_Type
          Mat_Type type=U1_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*1.0*s-(scoord[0]+0.5*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*1.0*s-(scoord[1]+0.5*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*1.0*s-(scoord[2]+0.5*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // U2_Type
          Mat_Type type=U2_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // X_Type
          Mat_Type type=X_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          if(tnode->pt_cnt[1]<=Nsrf)
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(scoord[0]+1.0*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(scoord[1]+1.0*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(scoord[2]+1.0*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        { // W_Type
          Mat_Type type=W_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            if(snode->IsGhost() && snode->src_coord.Dim()+snode->surf_coord.Dim()==0) continue; // Is non-leaf ghost node
            if(snode->pt_cnt[0]> Nsrf) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.25*s-(scoord[0]+0.25*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.25*s-(scoord[1]+0.25*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.25*s-(scoord[2]+0.25*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::U_List     (SetupData<Real_t>&  setup_data){
  //Add U_List contribution.
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2TargetSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, int level){
  if(!this->MultipoleOrder()) return;
  { // Set setup_data
    setup_data. level=level;
    setup_data.kernel=kernel->k_l2t;
    setup_data. input_data=&buff[1];
    setup_data.output_data=&buff[5];
    setup_data. coord_data=&buff[6];
    Vector<FMMNode_t*>& nodes_in =n_list[1];
    Vector<FMMNode_t*>& nodes_out=n_list[5];

    setup_data.nodes_in .clear();
    setup_data.nodes_out.clear();
    for(size_t i=0;i<nodes_in .Dim();i++) if((nodes_in [i]->depth==level || level==-1) && nodes_in [i]->trg_coord.Dim() && nodes_in [i]->IsLeaf() && !nodes_in [i]->IsGhost()) setup_data.nodes_in .push_back(nodes_in [i]);
    for(size_t i=0;i<nodes_out.Dim();i++) if((nodes_out[i]->depth==level || level==-1) && nodes_out[i]->trg_coord.Dim() && nodes_out[i]->IsLeaf() && !nodes_out[i]->IsGhost()) setup_data.nodes_out.push_back(nodes_out[i]);
  }

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

    PackedData src_coord; // Src coord
    PackedData src_value; // Src density
    PackedData srf_coord; // Srf coord
    PackedData srf_value; // Srf density
    PackedData trg_coord; // Trg coord
    PackedData trg_value; // Trg potential

    InteracData interac_data;
  };

  ptSetupData data;
  data. level=setup_data. level;
  data.kernel=setup_data.kernel;
  std::vector<void*>& nodes_in =setup_data.nodes_in ;
  std::vector<void*>& nodes_out=setup_data.nodes_out;

  { // Set src data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.src_coord;
    PackedData& value=data.src_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      ((FMMNode_t*)nodes[i])->node_id=i;
      Vector<Real_t>& coord_vec=tree->dnwd_equiv_surf[((FMMNode_t*)nodes[i])->depth];
      Vector<Real_t>& value_vec=((FMMData*)((FMMNode_t*)nodes[i])->FMMData())->dnward_equiv;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set srf data
    std::vector<void*>& nodes=nodes_in;
    PackedData& coord=data.srf_coord;
    PackedData& value=data.srf_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data. input_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      coord.dsp[i]=0;
      coord.cnt[i]=0;
      value.dsp[i]=0;
      value.cnt[i]=0;
    }
  }
  { // Set trg data
    std::vector<void*>& nodes=nodes_out;
    PackedData& coord=data.trg_coord;
    PackedData& value=data.trg_value;
    coord.ptr=setup_data. coord_data;
    value.ptr=setup_data.output_data;

    coord.len=coord.ptr->Dim(0)*coord.ptr->Dim(1);
    value.len=value.ptr->Dim(0)*value.ptr->Dim(1);

    coord.cnt.ReInit(nodes.size());
    coord.dsp.ReInit(nodes.size());
    value.cnt.ReInit(nodes.size());
    value.dsp.ReInit(nodes.size());

    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      Vector<Real_t>& coord_vec=((FMMNode_t*)nodes[i])->trg_coord;
      Vector<Real_t>& value_vec=((FMMNode_t*)nodes[i])->trg_value;
      if(coord_vec.Dim()){
        coord.dsp[i]=&coord_vec[0]-coord.ptr[0][0];
        assert(coord.dsp[i]<coord.len);
        coord.cnt[i]=coord_vec.Dim();
      }else{
        coord.dsp[i]=0;
        coord.cnt[i]=0;
      }
      if(value_vec.Dim()){
        value.dsp[i]=&value_vec[0]-value.ptr[0][0];
        assert(value.dsp[i]<value.len);
        value.cnt[i]=value_vec.Dim();
      }else{
        value.dsp[i]=0;
        value.cnt[i]=0;
      }
    }
  }
  { // Set interac_data
    int omp_p=omp_get_max_threads();
    std::vector<std::vector<size_t> > in_node_(omp_p);
    std::vector<std::vector<size_t> > scal_idx_(omp_p);
    std::vector<std::vector<Real_t> > coord_shift_(omp_p);
    std::vector<std::vector<size_t> > interac_cnt_(omp_p);
    if(this->ScaleInvar()){ // Set scal
      const Kernel<Real_t>* ker=kernel->k_l2l;
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+0]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+0];
        Vector<Real_t>& scal_exp=ker->trg_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
      for(size_t l=0;l<MAX_DEPTH;l++){ // scal[l*4+1]
        Vector<Real_t>& scal=data.interac_data.scal[l*4+1];
        Vector<Real_t>& scal_exp=ker->src_scal;
        scal.ReInit(scal_exp.Dim());
        for(size_t i=0;i<scal.Dim();i++){
          scal[i]=pvfmm::pow<Real_t>(2.0,-scal_exp[i]*l);
        }
      }
    }

    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<size_t>& in_node    =in_node_[tid]    ;
      std::vector<size_t>& scal_idx   =scal_idx_[tid]   ;
      std::vector<Real_t>& coord_shift=coord_shift_[tid];
      std::vector<size_t>& interac_cnt=interac_cnt_[tid];

      size_t a=(nodes_out.size()*(tid+0))/omp_p;
      size_t b=(nodes_out.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMMNode_t* tnode=(FMMNode_t*)nodes_out[i];
        Real_t s=pvfmm::pow<Real_t>(0.5,tnode->depth);

        size_t interac_cnt_=0;
        { // D2T_Type
          Mat_Type type=D2T_Type;
          Vector<FMMNode_t*>& intlst=tnode->interac_list[type];
          for(size_t j=0;j<intlst.Dim();j++) if(intlst[j]){
            FMMNode_t* snode=intlst[j];
            size_t snode_id=snode->node_id;
            if(snode_id>=nodes_in.size() || nodes_in[snode_id]!=snode) continue;
            in_node.push_back(snode_id);
            scal_idx.push_back(snode->depth);
            { // set coord_shift
              const int* rel_coord=interac_list.RelativeCoord(type,j);
              const Real_t* scoord=snode->Coord();
              const Real_t* tcoord=tnode->Coord();
              Real_t shift[COORD_DIM];
              shift[0]=rel_coord[0]*0.5*s-(0+0.5*s)+(tcoord[0]+0.5*s);
              shift[1]=rel_coord[1]*0.5*s-(0+0.5*s)+(tcoord[1]+0.5*s);
              shift[2]=rel_coord[2]*0.5*s-(0+0.5*s)+(tcoord[2]+0.5*s);
              coord_shift.push_back(shift[0]);
              coord_shift.push_back(shift[1]);
              coord_shift.push_back(shift[2]);
            }
            interac_cnt_++;
          }
        }
        interac_cnt.push_back(interac_cnt_);
      }
    }
    { // Combine interac data
      InteracData& interac_data=data.interac_data;
      { // in_node
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=in_node_;
        pvfmm::Vector<ElemType>& vec=interac_data.in_node;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // scal_idx
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=scal_idx_;
        pvfmm::Vector<ElemType>& vec=interac_data.scal_idx;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // coord_shift
        typedef Real_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=coord_shift_;
        pvfmm::Vector<ElemType>& vec=interac_data.coord_shift;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_cnt
        typedef size_t ElemType;
        std::vector<std::vector<ElemType> >& vec_=interac_cnt_;
        pvfmm::Vector<ElemType>& vec=interac_data.interac_cnt;

        std::vector<size_t> vec_dsp(omp_p+1,0);
        for(size_t tid=0;tid<omp_p;tid++){
          vec_dsp[tid+1]=vec_dsp[tid]+vec_[tid].size();
        }
        vec.ReInit(vec_dsp[omp_p]);
        #pragma omp parallel for
        for(size_t tid=0;tid<omp_p;tid++){
          memcpy(&vec[0]+vec_dsp[tid],&vec_[tid][0],vec_[tid].size()*sizeof(ElemType));
        }
      }
      { // interac_dsp
        pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
        pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
        dsp.ReInit(cnt.Dim()); if(dsp.Dim()) dsp[0]=0;
        omp_par::scan(&cnt[0],&dsp[0],dsp.Dim());
      }
    }
    { // Set M[0], M[1]
      InteracData& interac_data=data.interac_data;
      pvfmm::Vector<size_t>& cnt=interac_data.interac_cnt;
      pvfmm::Vector<size_t>& dsp=interac_data.interac_dsp;
      if(cnt.Dim() && cnt[cnt.Dim()-1]+dsp[dsp.Dim()-1]){
        data.interac_data.M[0]=this->mat->Mat(level, DC2DE0_Type, 0);
        data.interac_data.M[1]=this->mat->Mat(level, DC2DE1_Type, 0);
      }else{
        data.interac_data.M[0].ReInit(0,0);
        data.interac_data.M[1].ReInit(0,0);
      }
    }
  }

  PtSetup(setup_data, &data);
}

template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::Down2Target(SetupData<Real_t>&  setup_data){
  if(!this->MultipoleOrder()) return;
  //Add Down2Target contribution.
  this->EvalListPts(setup_data);
}


template <class FMMNode_t>
void FMM_Pts<FMMNode_t>::PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry){
  if(kernel->k_m2l->vol_poten && bndry==Periodic){ // Add analytical near-field to target potential
    const Kernel<Real_t>& k_m2t=*kernel->k_m2t;
    int ker_dim[2]={k_m2t.ker_dim[0],k_m2t.ker_dim[1]};

    Vector<Real_t>& up_equiv=((FMMData*)tree->RootNode()->FMMData())->upward_equiv;
    Matrix<Real_t> avg_density(1,ker_dim[0]); avg_density.SetZero();
    for(size_t i0=0;i0<up_equiv.Dim();i0+=ker_dim[0]){
      for(size_t i1=0;i1<ker_dim[0];i1++){
        avg_density[0][i1]+=up_equiv[i0+i1];
      }
    }

    int omp_p=omp_get_max_threads();
    std::vector<Matrix<Real_t> > M_tmp(omp_p);
    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++)
    if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
      Vector<Real_t>& trg_coord=nodes[i]->trg_coord;
      Vector<Real_t>& trg_value=nodes[i]->trg_value;
      size_t n_trg=trg_coord.Dim()/COORD_DIM;

      Matrix<Real_t>& M_vol=M_tmp[omp_get_thread_num()];
      M_vol.ReInit(ker_dim[0],n_trg*ker_dim[1]); M_vol.SetZero();
      k_m2t.vol_poten(&trg_coord[0],n_trg,&M_vol[0][0]);

      Matrix<Real_t> M_trg(1,n_trg*ker_dim[1],&trg_value[0],false);
      M_trg-=avg_density*M_vol;
    }
  }
}

}//end namespace
