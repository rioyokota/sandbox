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

 protected:

  int dim;               //Dimension of the tree
  int depth;             //Depth of the node (root -> 0)
  int max_depth;         //Maximum depth
  int path2node;         //Identity among siblings
  TreeNode* parent;      //Pointer to parent node
  TreeNode** child;      //Pointer child nodes
  int status;

  bool ghost;
  size_t max_pts;
  long long weight;

  Real_t coord[COORD_DIM];
  TreeNode * colleague[COLLEAGUE_COUNT];

 public:

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

  TreeNode(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1) {ghost=false; weight=1;}

  virtual ~TreeNode();

  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData* data_) ;

  virtual void ClearData(){}

  int Dim(){return dim;}

  int Depth(){return depth;}

  bool IsLeaf(){return child == NULL;}

  TreeNode* Child(int id);

  TreeNode* Parent();

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

  int Path2Node();

  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  virtual void Subdivide() ;

  virtual void Truncate() ;

  void SetParent(TreeNode* p, int path2node_) ;

  void SetChild(TreeNode* c, int id) ;

  TreeNode * Colleague(int index){return colleague[index];}

  void SetColleague(TreeNode * node_, int index){colleague[index]=node_;}

  virtual long long& NodeCost(){return weight;}

  Real_t* Coord(){assert(coord!=NULL); return coord;}

  bool IsGhost(){return ghost;}

  void SetGhost(bool x){ghost=x;}

  int& GetStatus();

  void SetStatus(int flag);

  size_t node_id; //For translating node pointer to index.

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
