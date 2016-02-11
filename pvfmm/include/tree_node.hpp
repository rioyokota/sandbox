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

  int dim;
  int depth;
  int max_depth;
  int path2node;
  TreeNode* parent;
  TreeNode** child;
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

  TreeNode(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1),
	      ghost(false), weight(1) {}

  virtual TreeNode* NewNode(TreeNode* n_=NULL){
    TreeNode* n=(n_==NULL?mem::aligned_new<TreeNode>():n_);
    return n;
  }

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
