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

  virtual TreeNode* NewNode(TreeNode* n_=NULL){
    TreeNode* n=(n_==NULL?mem::aligned_new<TreeNode>():n_);
    return NULL;
  }

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
