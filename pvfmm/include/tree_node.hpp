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

#ifndef _PVFMM_TREE_NODE_HPP_
#define _PVFMM_TREE_NODE_HPP_

namespace pvfmm{

class TreeNode{

 public:

  int dim;               //Dimension of the tree
  int depth;             //Depth of the node (root -> 0)
  int max_depth;         //Maximum depth
  int path2node;         //Identity among siblings
  TreeNode* parent;      //Pointer to parent node
  TreeNode** child;      //Pointer child nodes
  int status;

  bool ghost;
  size_t max_pts;
  size_t node_id;
  long long weight;

  Real_t coord[COORD_DIM];
  TreeNode * colleague[COLLEAGUE_COUNT];

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

  TreeNode(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1) {
    ghost=false;
    weight=1;
  }

  ~TreeNode() {
    if(!child) return;
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      if(child[i]!=NULL)
	mem::aligned_delete(child[i]);
    }
    mem::aligned_delete(child);
    child=NULL;
  }

  int Dim(){return dim;}

  int Depth(){return depth;}

  bool IsLeaf(){return child == NULL;}

  TreeNode* Child(int id){
    assert(id<(1<<dim));
    if(child==NULL) return NULL;
    return child[id];
  }

  TreeNode* Parent(){
    return parent;
  }

  inline MortonId GetMortonId() {
    assert(coord);
    Real_t s=0.25/(1UL<<MAX_DEPTH);
    return MortonId(coord[0]+s,coord[1]+s,coord[2]+s, Depth());
  }

  inline void SetCoord(MortonId& mid){
    assert(coord);
    mid.GetCoord(coord);
    depth=mid.GetDepth();
  }

  virtual int Path2Node(){
    return path2node;
  }

  virtual TreeNode* NewNode(TreeNode* n_=NULL){
    TreeNode* n=(n_==NULL?mem::aligned_new<TreeNode>():n_);
    n->dim=dim;
    n->max_depth=max_depth;
    n->max_pts=max_pts;
    return n_;
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

  void SetParent(TreeNode* p, int path2node_) {
    assert(path2node_>=0 && path2node_<(1<<dim));
    assert(p==NULL?true:p->Child(path2node_)==this);

    parent=p;
    path2node=path2node_;
    depth=(parent==NULL?0:parent->Depth()+1);
    if(parent!=NULL) max_depth=parent->max_depth;
  }

  void SetChild(TreeNode* c, int id) {
    assert(id<(1<<dim));
    child[id]=c;
    if(c!=NULL) child[id]->SetParent(this,id);
  }

  int& GetStatus(){
    return status;
  }

  void SetStatus(int flag){
    status=(status|flag);
    if(parent && !(parent->GetStatus() & flag))
      parent->SetStatus(flag);
  }

  TreeNode * Colleague(int index){return colleague[index];}

  void SetColleague(TreeNode * node_, int index){colleague[index]=node_;}

  virtual long long& NodeCost(){return weight;}

  Real_t* Coord(){assert(coord!=NULL); return coord;}

  bool IsGhost(){return ghost;}

  void SetGhost(bool x){ghost=x;}


};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
