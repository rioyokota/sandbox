#include <fmm_node.hpp>

#ifndef _PVFMM_FMM_PTS_HPP_
#define _PVFMM_FMM_PTS_HPP_

namespace pvfmm{

template <class Real_t>
struct SetupData{
  int level;
  const Kernel<Real_t>* kernel;
  std::vector<Mat_Type> interac_type;
  std::vector<void*> nodes_in ;
  std::vector<void*> nodes_out;
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
class FMM_Pts{

public:

  typedef FMMNode FMMNode_t;
  typedef FMM_Tree<FMM_Pts<FMMNode_t> > FMMTree_t;

 private:

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
    if(p_indx==ReflecX || p_indx==ReflecY || p_indx==ReflecZ){ // Set P.perm
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
  
  virtual void PrecompAll(Mat_Type type, int level=-1) {
    if(level==-1) {
      for(int l=0;l<MAX_DEPTH;l++) {
        PrecompAll(type, l);
      }
      return;
    }
    for(size_t i=0;i<Perm_Count;i++) {
      this->PrecompPerm(type, (Perm_Type) i);
    }
    size_t mat_cnt=interac_list.ListCount((Mat_Type)type);
    mat->Mat(level, (Mat_Type)type, mat_cnt-1);
    std::vector<size_t> indx_lst;
    for(size_t i=0; i<mat_cnt; i++) {
      if(interac_list.InteracClass((Mat_Type)type,i)==i) {
        indx_lst.push_back(i);
      }
    }
    for(size_t i=0; i<indx_lst.size(); i++){
      Precomp(level, (Mat_Type)type, indx_lst[i]);
    }
    for(size_t mat_indx=0;mat_indx<mat_cnt;mat_indx++){
      Matrix<Real_t>& M0=interac_list.ClassMat(level,(Mat_Type)type,mat_indx);
      Permutation<Real_t>& pr=interac_list.Perm_R(level, (Mat_Type)type, mat_indx);
      Permutation<Real_t>& pc=interac_list.Perm_C(level, (Mat_Type)type, mat_indx);
      if(pr.Dim()!=M0.Dim(0) || pc.Dim()!=M0.Dim(1)) Precomp(level, (Mat_Type)type, mat_indx);
    }
  }
  
  virtual Permutation<Real_t>& PrecompPerm(Mat_Type type, Perm_Type perm_indx) {
    Permutation<Real_t>& P_ = mat->Perm((Mat_Type)type, perm_indx);
    if(P_.Dim()!=0) return P_;
    size_t m=this->MultipoleOrder();
    size_t p_indx=perm_indx % C_Perm;
    Permutation<Real_t> P;
    switch (type){
      case U2U_Type:
      {
        Vector<Real_t> scal_exp;
        Permutation<Real_t> ker_perm;
        if(perm_indx<C_Perm){ // Source permutation
          ker_perm=kernel->k_m2m->perm_vec[0     +p_indx];
          scal_exp=kernel->k_m2m->src_scal;
        }else{ // Target permutation
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
        if(perm_indx<C_Perm){ // Source permutation
          ker_perm=kernel->k_l2l->perm_vec[C_Perm+p_indx];
          scal_exp=kernel->k_l2l->trg_scal;
          for(size_t i=0;i<scal_exp.Dim();i++) scal_exp[i]=-scal_exp[i];
        }else{ // Target permutation
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
  
  virtual Matrix<Real_t>& Precomp(int level, Mat_Type type, size_t mat_indx) {
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

  void FFT_UpEquiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& fft_vec, Vector<Real_t>& fft_scl,
      Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_);
  void FFT_Check2Equiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& ifft_vec, Vector<Real_t>& ifft_scl,
      Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_);

 public:

  class FMMData: public FMM_Data<Real_t>{
   public:
    virtual ~FMMData(){}
    virtual FMM_Data<Real_t>* NewData(){return mem::aligned_new<FMMData>();}
  };

  Vector<char> dev_buffer;
  Vector<char> staging_buffer;

  FMM_Pts(mem::MemoryManager* mem_mgr_=NULL): mem_mgr(mem_mgr_),
             vprecomp_fft_flag(false), vlist_fft_flag(false),
               vlist_ifft_flag(false), mat(NULL), kernel(NULL){};

  virtual ~FMM_Pts();

  void Initialize(int mult_order, const Kernel<Real_t>* kernel);

  int MultipoleOrder(){return multipole_order;}

  bool ScaleInvar(){return kernel->scale_invar;}

  virtual void CollectNodeData(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list,
std::vector<std::vector<Vector<Real_t>* > > vec_list = std::vector<std::vector<Vector<Real_t>* > >(0));

  void SetupPrecomp(SetupData<Real_t>& setup_data);
  void SetupInterac(SetupData<Real_t>& setup_data);
  void EvalList    (SetupData<Real_t>& setup_data);
  void PtSetup(SetupData<Real_t>&  setup_data, void* data_);
  void EvalListPts(SetupData<Real_t>& setup_data);

  virtual void Source2UpSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void Source2Up     (SetupData<Real_t>&  setup_data);
  virtual void Up2UpSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void Up2Up     (SetupData<Real_t>&  setup_data);
  virtual void PeriodicBC(FMMNode_t* node);
  virtual void V_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void V_List     (SetupData<Real_t>&  setup_data);
  virtual void X_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void X_List     (SetupData<Real_t>&  setup_data);
  virtual void Down2DownSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void Down2Down     (SetupData<Real_t>&  setup_data);
  virtual void Down2TargetSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void Down2Target     (SetupData<Real_t>&  setup_data);
  virtual void W_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void W_List     (SetupData<Real_t>&  setup_data);
  virtual void U_ListSetup(SetupData<Real_t>&  setup_data, FMMTree_t* tree, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level);
  virtual void U_List     (SetupData<Real_t>&  setup_data);
  virtual void PostProcessing(FMMTree_t* tree, std::vector<FMMNode_t*>& nodes, BoundaryType bndry=FreeSpace);
};

}//end namespace

#include <fmm_pts.txx>

#endif //_PVFMM_FMM_PTS_HPP_

