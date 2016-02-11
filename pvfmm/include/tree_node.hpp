#include <pvfmm_common.hpp>
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

  virtual void Initialize(TreeNode* parent_, int path2node_, NodeData* data_) ;

  virtual void ClearData(){}

  int Dim(){return dim;}

  int Depth(){return depth;}

  bool IsLeaf(){return child == NULL;}

  TreeNode* Child(int id);

  TreeNode* Parent();

  int Path2Node();

  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  virtual void Subdivide() ;

  virtual void Truncate() ;

  void SetParent(TreeNode* p, int path2node_) ;

  void SetChild(TreeNode* c, int id) ;

  int& GetStatus();

  void SetStatus(int flag);

  size_t node_id; //For translating node pointer to index.

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
