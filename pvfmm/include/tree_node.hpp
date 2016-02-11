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

  virtual TreeNode* NewNode(TreeNode* n_=NULL){
    TreeNode* n=(n_==NULL?mem::aligned_new<TreeNode>():n_);
    return n;
  }

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
