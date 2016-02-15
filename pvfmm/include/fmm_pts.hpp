#ifndef _PVFMM_FMM_PTS_HPP_
#define _PVFMM_FMM_PTS_HPP_

#include <fmm_node.hpp>

namespace pvfmm{

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
    
 public:

  class FMMData: public FMM_Data<Real_t>{
   public:
    ~FMMData(){}
    FMM_Data<Real_t>* NewData(){return mem::aligned_new<FMMData>();}
  };

  Vector<char> dev_buffer;

  FMM_Pts(): vprecomp_fft_flag(false), vlist_fft_flag(false),
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
  
};

}//end namespace

#endif //_PVFMM_FMM_PTS_HPP_

