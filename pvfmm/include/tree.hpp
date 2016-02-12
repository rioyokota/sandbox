#include <cassert>
#include <vector>

#include <pvfmm_common.hpp>
#include <mem_mgr.hpp>

#ifndef _PVFMM_TREE_HPP_
#define _PVFMM_TREE_HPP_

namespace pvfmm{

template <class TreeNode>
class Tree{

 protected:

  int dim;
  TreeNode* root_node;
  int max_depth;
  std::vector<TreeNode*> node_lst;
  mem::MemoryManager memgr;

 public:

  Tree(): dim(0), root_node(NULL), max_depth(MAX_DEPTH), memgr(0) {}

  virtual ~Tree() {}

};

}//end namespace

#endif //_PVFMM_TREE_HPP_
