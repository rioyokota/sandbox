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

  virtual ~Tree() {
    if(RootNode()!=NULL){
      mem::aligned_delete(root_node);
    }
  }

  virtual void Initialize(typename TreeNode::NodeData* init_data_) {
    dim=init_data_->dim;
    max_depth=init_data_->max_depth;
    if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
    if(root_node) mem::aligned_delete(root_node);
    root_node=mem::aligned_new<TreeNode>();
    root_node->Initialize(NULL,0,init_data_);
  }

  TreeNode* RootNode() {return root_node;}

  TreeNode* PreorderFirst() {
    return root_node;
  }

  TreeNode* PreorderNxt(TreeNode* curr_node) {
    assert(curr_node!=NULL);
    int n=(1UL<<dim);
    if(!curr_node->IsLeaf())
      for(int i=0;i<n;i++)
	if(curr_node->Child(i)!=NULL)
	  return (TreeNode*)curr_node->Child(i);
    TreeNode* node=curr_node;
    while(true){
      int i=node->Path2Node()+1;
      node=(TreeNode*)node->Parent();
      if(node==NULL) return NULL;

      for(;i<n;i++)
	if(node->Child(i)!=NULL)
	  return (TreeNode*)node->Child(i);
    }
  }


  TreeNode* PostorderFirst() {
    TreeNode* node=root_node;
    int n=(1UL<<dim);
    while(true){
      if(node->IsLeaf()) return node;
      for(int i=0;i<n;i++)
	if(node->Child(i)!=NULL){
	  node=(TreeNode*)node->Child(i);
	  break;
	}
    }
  }

  TreeNode* PostorderNxt(TreeNode* curr_node) {
    assert(curr_node!=NULL);
    TreeNode* node=curr_node;
    int j=node->Path2Node()+1;
    node=(TreeNode*)node->Parent();
    if(node==NULL) return NULL;
    int n=(1UL<<dim);
    for(;j<n;j++){
      if(node->Child(j)!=NULL){
	node=(TreeNode*)node->Child(j);
	while(true){
	  if(node->IsLeaf()) return node;
	  for(int i=0;i<n;i++) {
	    if(node->Child(i)!=NULL){
	      node=(TreeNode*)node->Child(i);
	      break;
	    }
	  }
	}
      }
    }
    return node;
  }

  std::vector<TreeNode*>& GetNodeList() {
    if(root_node->GetStatus() & 1){
      node_lst.clear();
      TreeNode* n=this->PreorderFirst();
      while(n!=NULL){
	int& status=n->GetStatus();
	status=(status & (~(int)1));
	node_lst.push_back(n);
	n=this->PreorderNxt(n);
      }
    }
    return node_lst;
  }

  int Dim() {return dim;}

 protected:

  int dim;              // dimension of the tree
  TreeNode* root_node;    // pointer to root node
  int max_depth;        // maximum tree depth
  std::vector<TreeNode*> node_lst;
  mem::MemoryManager memgr;
};

}//end namespace

#endif //_PVFMM_TREE_HPP_
