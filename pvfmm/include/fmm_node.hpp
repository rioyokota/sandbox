#include <profile.hpp>
#include <cheb_utils.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#ifdef PVFMM_HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

#include <fft_wrapper.hpp>
#include <interac_list.hpp>
#include <kernel.hpp>
#include <matrix.hpp>
#include <mem_mgr.hpp>
#include <mortonid.hpp>
#include <mpi_tree.hpp>
#include <precomp_mat.hpp>
#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_

namespace pvfmm{

template <class Real_t>
class FMM_Data{

 public:

  virtual ~FMM_Data(){}

  virtual FMM_Data* NewData(){return mem::aligned_new<FMM_Data>();}

  virtual void Clear();

  Vector<Real_t> upward_equiv;
  Vector<Real_t> dnward_equiv;
};

class FMM_Node {

 private:

  FMM_Data<Real_t>* fmm_data;

 public:

  int dim;
  int depth;
  int max_depth;
  int path2node;
  FMM_Node* parent;
  FMM_Node** child;
  int status;

  bool ghost;
  size_t max_pts;
  size_t node_id;
  long long weight;

  Real_t coord[COORD_DIM];
  FMM_Node * colleague[COLLEAGUE_COUNT];

  Vector<Real_t> pt_coord;
  Vector<Real_t> pt_value;
  Vector<size_t> pt_scatter;

  Vector<Real_t> src_coord;
  Vector<Real_t> src_value;
  Vector<size_t> src_scatter;
  Vector<Real_t> surf_coord;
  Vector<Real_t> surf_value;
  Vector<size_t> surf_scatter;
  Vector<Real_t> trg_coord;
  Vector<Real_t> trg_value;
  Vector<size_t> trg_scatter;
  size_t pt_cnt[2]; // Number of source, target pts.
  Vector<FMM_Node*> interac_list[Type_Count];

  class NodeData {
    public:
     virtual ~NodeData(){};
     virtual void Clear(){}
     int max_depth;
     int dim;
     size_t max_pts;
     Vector<Real_t> coord;
     Vector<Real_t> value;
     Vector<Real_t> src_coord;
     Vector<Real_t> src_value;
     Vector<Real_t> surf_coord;
     Vector<Real_t> surf_value;
     Vector<Real_t> trg_coord;
     Vector<Real_t> trg_value;
  };

  FMM_Node() : dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1),
	       ghost(false), weight(1) {
    fmm_data=NULL;
  }

  virtual ~FMM_Node(){
    if(fmm_data!=NULL) mem::aligned_delete(fmm_data);
    fmm_data=NULL;
    if(!child) return;
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	mem::aligned_delete(child[i]);
    }
    mem::aligned_delete(child);
    child=NULL;
  }

  virtual void Initialize(FMM_Node* parent_, int path2node_, FMM_Node::NodeData* data_){
    parent=parent_;
    depth=(parent==NULL?0:parent->depth+1);
    if(data_!=NULL){
      dim=data_->dim;
      max_depth=data_->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
    }else if(parent!=NULL){
      dim=((FMM_Node*)parent)->Dim();
      max_depth=((FMM_Node*)parent)->max_depth;
    }
    assert(path2node_>=0 && path2node_<(int)(1U<<dim));
    path2node=path2node_;

    Real_t coord_offset=((Real_t)1.0)/((Real_t)(((uint64_t)1)<<depth));
    if(!parent_){
      for(int j=0;j<dim;j++) coord[j]=0;
    }else if(parent_){
      int flag=1;
      for(int j=0;j<dim;j++){
	coord[j]=((FMM_Node*)parent_)->coord[j]+
	  ((Path2Node() & flag)?coord_offset:0.0f);
	flag=flag<<1;
      }
    }

    int n=pvfmm::pow<unsigned int>(3,Dim());
    for(int i=0;i<n;i++) colleague[i]=NULL;

    NodeData* mpi_data=dynamic_cast<NodeData*>(data_);
    if(data_){
      max_pts =mpi_data->max_pts;
      pt_coord=mpi_data->coord;
      pt_value=mpi_data->value;
    }else if(parent){
      max_pts =parent->max_pts;
      SetGhost(((FMM_Node*)parent)->IsGhost());
    }

    typename FMM_Node::NodeData* data=dynamic_cast<typename FMM_Node::NodeData*>(data_);
    if(data_!=NULL){
      src_coord=data->src_coord;
      src_value=data->src_value;
      surf_coord=data->surf_coord;
      surf_value=data->surf_value;
      trg_coord=data->trg_coord;
      trg_value=data->trg_value;
    }
  }

  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
    scatter.push_back(&pt_scatter);
    coord  .push_back(&src_coord  );
    value  .push_back(&src_value  );
    scatter.push_back(&src_scatter);
    coord  .push_back(&surf_coord  );
    value  .push_back(&surf_value  );
    scatter.push_back(&surf_scatter);
    coord  .push_back(&trg_coord  );
    value  .push_back(&trg_value  );
    scatter.push_back(&trg_scatter);
  }

  virtual void ClearData(){
    ClearFMMData();
    pt_coord.ReInit(0);
    pt_value.ReInit(0);
  }

  virtual void ClearFMMData(){
    if(fmm_data!=NULL)
      fmm_data->Clear();
  }

  virtual void Truncate() {
    if(!child) return;
    SetStatus(1);
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	mem::aligned_delete(child[i]);
    }
    mem::aligned_delete(child);
    child=NULL;
  }

  FMM_Data<Real_t>*& FMMData() {
    return fmm_data;
  }

  FMM_Node* NewNode(FMM_Node* n_=NULL) {
    FMM_Node* n=(n_==NULL?mem::aligned_new<FMM_Node>():static_cast<FMM_Node*>(n_));
    if(fmm_data!=NULL) n->fmm_data=fmm_data->NewData();
    n->dim=dim;
    n->max_depth=max_depth;
    n->max_pts=max_pts;
    return n;
  }

  virtual void Subdivide(){
    if(!this->IsLeaf()) return;
    if(child) return;
    SetStatus(1);
    int n=(1UL<<dim);
    child=mem::aligned_new<FMM_Node*>(n);
    for(int i=0;i<n;i++){
      child[i]=NewNode();
      child[i]->parent=this;
      ((FMM_Node*)child[i])->Initialize(this,i,NULL);
    }
    int nchld=(1UL<<this->Dim());
    if(!IsGhost()){ // Partition point coordinates and values.
      std::vector<Vector<Real_t>*> pt_c;
      std::vector<Vector<Real_t>*> pt_v;
      std::vector<Vector<size_t>*> pt_s;
      this->NodeDataVec(pt_c, pt_v, pt_s);

      std::vector<std::vector<Vector<Real_t>*> > chld_pt_c(nchld);
      std::vector<std::vector<Vector<Real_t>*> > chld_pt_v(nchld);
      std::vector<std::vector<Vector<size_t>*> > chld_pt_s(nchld);
      for(size_t i=0;i<nchld;i++){
	((FMM_Node*)this->Child(i))->NodeDataVec(chld_pt_c[i], chld_pt_v[i], chld_pt_s[i]);
      }

      Real_t* c=this->Coord();
      Real_t s=pvfmm::pow<Real_t>(0.5,depth+1);
      for(size_t j=0;j<pt_c.size();j++){
	if(!pt_c[j] || !pt_c[j]->Dim()) continue;
	Vector<Real_t>& coord=*pt_c[j];
	size_t npts=coord.Dim()/this->dim;

	Vector<size_t> cdata(nchld+1);
	for(size_t i=0;i<nchld+1;i++){
	  long long pt1=-1, pt2=npts;
	  while(pt2-pt1>1){ // binary search
	    long long pt3=(pt1+pt2)/2;
	    assert(pt3<npts);
	    if(pt3<0) pt3=0;
	    int ch_id=(coord[pt3*3+0]>=c[0]+s)*1+
	      (coord[pt3*3+1]>=c[1]+s)*2+
	      (coord[pt3*3+2]>=c[2]+s)*4;
	    if(ch_id< i) pt1=pt3;
	    if(ch_id>=i) pt2=pt3;
	  }
	  cdata[i]=pt2;
	}

	if(pt_c[j]){
	  Vector<Real_t>& vec=*pt_c[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<Real_t>& chld_vec=*chld_pt_c[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
	if(pt_v[j]){
	  Vector<Real_t>& vec=*pt_v[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<Real_t>& chld_vec=*chld_pt_v[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
	if(pt_s[j]){
	  Vector<size_t>& vec=*pt_s[j];
	  size_t dof=vec.Dim()/npts;
	  if(dof>0) for(size_t i=0;i<nchld;i++){
	      Vector<size_t>& chld_vec=*chld_pt_s[i][j];
	      chld_vec.ReInit((cdata[i+1]-cdata[i])*dof, &vec[0]+cdata[i]*dof);
	    }
	  vec.ReInit(0);
	}
      }
    }
  }

  void ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost=true) {
    if(!pt_coord.Dim()) return;
    size_t n_pts=pt_coord.Dim()/dim;
    size_t data_dof=pt_value.Dim()/n_pts;
    std::vector<Real_t> v(data_dof,0);
    for(size_t i=0;i<n_pts;i++)
      for(int j=0;j<data_dof;j++)
	v[j]+=pt_value[i*data_dof+j];
    for(int j=0;j<data_dof;j++)
      v[j]=v[j]/n_pts;
    for(size_t i=0;i<x.size()*y.size()*z.size()*data_dof;i++){
      val[i]=v[i%data_dof];
    }
  }

  int Dim() {
    return dim;
  }

  bool IsLeaf() {
    return child == NULL;
  }

  bool IsGhost() {
    return ghost;
  }

  void SetGhost(bool x) {
    ghost=x;
  }

  int& GetStatus() {
    return status;
  }

  void SetStatus(int flag) {
    status=(status|flag);
    if(parent && !(((FMM_Node*)parent)->GetStatus() & flag))
      ((FMM_Node*)parent)->SetStatus(flag);
  }


  FMM_Node* Child(int id){
    assert(id<(1<<dim));
    if(child==NULL) return NULL;
    return child[id];
  }

  FMM_Node* Parent(){
    return parent;
  }

  inline MortonId GetMortonId() {
    assert(coord);
    Real_t s=0.25/(1UL<<MAX_DEPTH);
    return MortonId(coord[0]+s,coord[1]+s,coord[2]+s, depth);
  }

  inline void SetCoord(MortonId& mid) {
    assert(coord);
    mid.GetCoord(coord);
    depth=mid.GetDepth();
  }

  virtual int Path2Node(){
    return path2node;
  }

  void SetParent(FMM_Node* p, int path2node_) {
    assert(path2node_>=0 && path2node_<(1<<dim));
    assert(p==NULL?true:p->Child(path2node_)==this);
    parent=p;
    path2node=path2node_;
    depth=(parent==NULL?0:parent->depth+1);
    if(parent!=NULL) max_depth=parent->max_depth;
  }

  void SetChild(FMM_Node* c, int id) {
    assert(id<(1<<dim));
    child[id]=c;
    if(c!=NULL) ((FMM_Node*)child[id])->SetParent(this,id);
  }

  FMM_Node * Colleague(int index) {
    return colleague[index];
  }

  void SetColleague(FMM_Node * node_, int index) {
    colleague[index]=node_;
  }

  Real_t* Coord() {
    assert(coord!=NULL);
    return coord;
  }

};

}//end namespace

#endif //_PVFMM_FMM_NODE_HPP_
