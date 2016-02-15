#include <parUtils.h>
#include <ompUtils.h>
#include <pvfmm_common.hpp>
#include <precomp_mat.hpp>
#include <matrix.hpp>

#ifndef _PVFMM_INTERAC_LIST_HPP_
#define _PVFMM_INTERAC_LIST_HPP_

namespace pvfmm{

template <class Node_t>
class InteracList{
public:

  unsigned int dim;
  std::vector<Matrix<int> > rel_coord;
  std::vector<std::vector<int> > hash_lut;
  std::vector<std::vector<size_t> > interac_class;
  std::vector<std::vector<std::vector<Perm_Type> > > perm_list;
  PrecompMat<Real_t>* mat;
  bool use_symmetries;

  InteracList(){}

  InteracList(unsigned int dim_){
    Initialize(dim_);
  }

  void Initialize(unsigned int dim_, PrecompMat<Real_t>* mat_=NULL);

  size_t ListCount(Mat_Type t);

  int* RelativeCoord(Mat_Type t, size_t i);

  void BuildList(Node_t* n, Mat_Type t);

  size_t InteracClass(Mat_Type t, size_t i);

  Matrix<Real_t>& ClassMat(int l, Mat_Type type, size_t indx);

  Permutation<Real_t>& Perm_R(int l, Mat_Type type, size_t indx){
    size_t indx0=InteracClass(type, indx);
    Matrix     <Real_t>& M0      =mat->Mat   (l, type, indx0);
    Permutation<Real_t>& row_perm=mat->Perm_R(l, type, indx );
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return row_perm;
    if(row_perm.Dim()==0){
      std::vector<Perm_Type> p_list=PermutList(type, indx);
      for(int i=0;i<l;i++) p_list.push_back(Scaling);
      Permutation<Real_t> row_perm_=Permutation<Real_t>(M0.Dim(0));
      for(int i=0;i<C_Perm;i++){
	Permutation<Real_t>& pr=mat->Perm(type, R_Perm + i);
	if(!pr.Dim()) row_perm_=Permutation<Real_t>(0);
      }
      if(row_perm_.Dim()>0)
	for(int i=p_list.size()-1; i>=0; i--){
	  assert(type!=V_Type);
	  Permutation<Real_t>& pr=mat->Perm(type, R_Perm + p_list[i]);
	  row_perm_=pr.Transpose()*row_perm_;
	}
      row_perm=row_perm_;
    }
    return row_perm;
  }

  Permutation<Real_t>& Perm_C(int l, Mat_Type type, size_t indx){
    size_t indx0=InteracClass(type, indx);
    Matrix     <Real_t>& M0      =mat->Mat   (l, type, indx0);
    Permutation<Real_t>& col_perm=mat->Perm_C(l, type, indx );
    if(M0.Dim(0)==0 || M0.Dim(1)==0) return col_perm;
    if(col_perm.Dim()==0){
      std::vector<Perm_Type> p_list=PermutList(type, indx);
      for(int i=0;i<l;i++) p_list.push_back(Scaling);
      Permutation<Real_t> col_perm_=Permutation<Real_t>(M0.Dim(1));
      for(int i=0;i<C_Perm;i++){
	Permutation<Real_t>& pc=mat->Perm(type, C_Perm + i);
	if(!pc.Dim()) col_perm_=Permutation<Real_t>(0);
      }
      if(col_perm_.Dim()>0)
	for(int i=p_list.size()-1; i>=0; i--){
	  assert(type!=V_Type);
	  Permutation<Real_t>& pc=mat->Perm(type, C_Perm + p_list[i]);
	  col_perm_=col_perm_*pc;
	}
      col_perm=col_perm_;
    }
    return col_perm;
  }


  std::vector<Perm_Type>& PermutList(Mat_Type t, size_t i);

  void InitList(int max_r, int min_r, int step, Mat_Type t){
    size_t count=pvfmm::pow<unsigned int>((max_r*2)/step+1,dim)
      -(min_r>0?pvfmm::pow<unsigned int>((min_r*2)/step-1,dim):0);
    Matrix<int>& M=rel_coord[t];
    M.Resize(count,dim);
    hash_lut[t].assign(PVFMM_MAX_COORD_HASH, -1);
    std::vector<int> class_size_hash(PVFMM_MAX_COORD_HASH, 0);
    std::vector<int> class_disp_hash(PVFMM_MAX_COORD_HASH, 0);
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
	    class_size_hash[class_hash(c)]++;
	  }
    omp_par::scan(&class_size_hash[0], &class_disp_hash[0], PVFMM_MAX_COORD_HASH);
    size_t count_=0;
    for(int k=-max_r;k<=max_r;k+=step)
      for(int j=-max_r;j<=max_r;j+=step)
	for(int i=-max_r;i<=max_r;i+=step)
	  if(abs(i)>=min_r || abs(j)>=min_r || abs(k) >= min_r){
	    int c[3]={i,j,k};
	    int& idx=class_disp_hash[class_hash(c)];
	    for(size_t l=0;l<dim;l++) M[idx][l]=c[l];
	    hash_lut[t][coord_hash(c)]=idx;
	    count_++;
	    idx++;
	  }
    assert(count_==count);
    interac_class[t].resize(count);
    perm_list[t].resize(count);
    if(!use_symmetries){
      for(size_t j=0;j<count;j++){
	int c_hash = coord_hash(&M[j][0]);
	interac_class[t][j]=hash_lut[t][c_hash];
      }
    } else {
      for(size_t j=0;j<count;j++){
	if(M[j][0]<0) perm_list[t][j].push_back(ReflecX);
	if(M[j][1]<0) perm_list[t][j].push_back(ReflecY);
	if(M[j][2]<0) perm_list[t][j].push_back(ReflecZ);
	int coord[3];
	coord[0]=abs(M[j][0]);
	coord[1]=abs(M[j][1]);
	coord[2]=abs(M[j][2]);
	if(coord[1]>coord[0] && coord[1]>coord[2]){
	  perm_list[t][j].push_back(SwapXY);
	  int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
	}
	if(coord[0]>coord[2]){
	  perm_list[t][j].push_back(SwapXZ);
	  int tmp=coord[0]; coord[0]=coord[2]; coord[2]=tmp;
	}
	if(coord[0]>coord[1]){
	  perm_list[t][j].push_back(SwapXY);
	  int tmp=coord[0]; coord[0]=coord[1]; coord[1]=tmp;
	}
	assert(coord[0]<=coord[1] && coord[1]<=coord[2]);
	int c_hash = coord_hash(&coord[0]);
	interac_class[t][j]=hash_lut[t][c_hash];
      }
    }
  }

  int coord_hash(int* c){
    const int n=5;
    return ( (c[2]+n) * (2*n) + (c[1]+n) ) *(2*n) + (c[0]+n);
  }

  int class_hash(int* c_){
    if(!use_symmetries) return coord_hash(c_);
    int c[3]={abs(c_[0]), abs(c_[1]), abs(c_[2])};
    if(c[1]>c[0] && c[1]>c[2])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    if(c[0]>c[2])
      {int tmp=c[0]; c[0]=c[2]; c[2]=tmp;}
    if(c[0]>c[1])
      {int tmp=c[0]; c[0]=c[1]; c[1]=tmp;}
    assert(c[0]<=c[1] && c[1]<=c[2]);
    return coord_hash(&c[0]);
  }

};

}//end namespace

#include <interac_list.txx>

#endif //_PVFMM_INTERAC_LIST_HPP_

