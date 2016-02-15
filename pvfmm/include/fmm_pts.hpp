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

