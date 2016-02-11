#include <cassert>
#include <vector>

#include <pvfmm_common.hpp>
#include <mem_mgr.hpp>

#ifndef _PVFMM_TREE_HPP_
#define _PVFMM_TREE_HPP_

namespace pvfmm{

template <class TreeNode>
class Tree{

 public:

  Tree(): dim(0), root_node(NULL), max_depth(MAX_DEPTH), memgr(0) { };

  virtual ~Tree();

  virtual void Initialize(typename TreeNode::NodeData* init_data) ;

  TreeNode* RootNode() {return root_node;}

  TreeNode* PreorderFirst();

  TreeNode* PreorderNxt(TreeNode* curr_node);

  TreeNode* PostorderFirst();

  TreeNode* PostorderNxt(TreeNode* curr_node);

  std::vector<TreeNode*>& GetNodeList();

  int Dim() {return dim;}

 protected:

  int dim;              // dimension of the tree
  TreeNode* root_node;    // pointer to root node
  int max_depth;        // maximum tree depth
  std::vector<TreeNode*> node_lst;
  mem::MemoryManager memgr;
};

}//end namespace

#include <tree.txx>

#endif //_PVFMM_TREE_HPP_
