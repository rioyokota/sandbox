#include <cmath>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <stdint.h>

#include <pvfmm_common.hpp>
#include <matrix.hpp>
#include <mem_mgr.hpp>
#include <mortonid.hpp>
#include <vector.hpp>

#ifndef _PVFMM_MPI_NODE_HPP_
#define _PVFMM_MPI_NODE_HPP_

namespace pvfmm{

class MPI_Node {

 public:

  int dim;
  int depth;
  int max_depth;
  int path2node;
  MPI_Node* parent;
  MPI_Node** child;
  int status;

  bool ghost;
  size_t max_pts;
  size_t node_id;
  long long weight;

  Real_t coord[COORD_DIM];
  MPI_Node * colleague[COLLEAGUE_COUNT];

  Vector<Real_t> pt_coord;
  Vector<Real_t> pt_value;
  Vector<size_t> pt_scatter;

  class NodeData{
   public:
     virtual ~NodeData(){};
     virtual void Clear(){}
     int max_depth;
     int dim;
     size_t max_pts;
     Vector<Real_t> coord;
     Vector<Real_t> value;
  };

  MPI_Node(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1),
              ghost(false), weight(1) {}

  virtual ~MPI_Node() {
    if(!child) return;
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	mem::aligned_delete(child[i]);
    }
    mem::aligned_delete(child);
    child=NULL;
  }

  void ClearData() {
    pt_coord.ReInit(0);
    pt_value.ReInit(0);
  }

  void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
		   std::vector<Vector<Real_t>*>& value,
		   std::vector<Vector<size_t>*>& scatter){
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
    scatter.push_back(&pt_scatter);
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

  void Truncate() {
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
    if(parent && !(parent->GetStatus() & flag))
      parent->SetStatus(flag);
  }


  MPI_Node* Child(int id){
    assert(id<(1<<dim));
    if(child==NULL) return NULL;
    return child[id];
  }

  MPI_Node* Parent(){
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

  void SetParent(MPI_Node* p, int path2node_) {
    assert(path2node_>=0 && path2node_<(1<<dim));
    assert(p==NULL?true:p->Child(path2node_)==this);
    parent=p;
    path2node=path2node_;
    depth=(parent==NULL?0:parent->depth+1);
    if(parent!=NULL) max_depth=parent->max_depth;
  }

  void SetChild(MPI_Node* c, int id) {
    assert(id<(1<<dim));
    child[id]=c;
    if(c!=NULL) child[id]->SetParent(this,id);
  }

  MPI_Node * Colleague(int index) {
    return colleague[index];
  }

  void SetColleague(MPI_Node * node_, int index) {
    colleague[index]=node_;
  }

  Real_t* Coord() {
    assert(coord!=NULL);
    return coord;
  }

};

}//end namespace

#endif //_PVFMM_MPI_NODE_HPP_
