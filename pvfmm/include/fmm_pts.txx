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


}//end namespace
