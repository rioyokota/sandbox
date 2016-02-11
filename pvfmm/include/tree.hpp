#include <cassert>
#include <vector>

#include <pvfmm_common.hpp>
#include <tree_node.hpp>
#include <mem_mgr.hpp>

#ifndef _PVFMM_TREE_HPP_
#define _PVFMM_TREE_HPP_

namespace pvfmm{

template <class TreeNode>
class Tree{

 public:

   typedef TreeNode Node_t;

  Tree(): dim(0), root_node(NULL), max_depth(MAX_DEPTH), memgr(0) { };

  virtual ~Tree();

  virtual void Initialize(typename Node_t::NodeData* init_data) ;

  Node_t* RootNode() {return root_node;}

  Node_t* PreorderFirst();

  Node_t* PreorderNxt(Node_t* curr_node);

  Node_t* PostorderFirst();

  Node_t* PostorderNxt(Node_t* curr_node);

  std::vector<TreeNode*>& GetNodeList();

  int Dim() {return dim;}

 protected:

  int dim;              // dimension of the tree
  Node_t* root_node;    // pointer to root node
  int max_depth;        // maximum tree depth
  std::vector<TreeNode*> node_lst;
  mem::MemoryManager memgr;
};

}//end namespace

#include <tree.txx>

#endif //_PVFMM_TREE_HPP_
