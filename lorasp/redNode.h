#ifndef redNode_h
#define redNode_h

#include <vector>
#include "node.h"
#include <set>

class blackNode;

/**************************************************************************/
/*                             CLASS Red NODE                             */
/**************************************************************************/

//! Class redNode
/*!
  Inherited from the general class node. It is a node correponding to the multipole variables, 
  and local equations.
*/

class redNode:public node
{
  
  //! Pointer to the parent (which is a blackNode/NULL)
  blackNode* parent_;
  
  //! Pointer to the child (which is a blackNode)
  blackNode* child_;

  //! The level of node ( only redNodes have level)
  unsigned int level_;

  //! Indicate if it is a left(0) or right(1) child;
  bool which_;

  //! The adjacency set (from original interaction of the symbolic matrix)
  std::set<redNode*> AdjList_;

 public:
  
  //! Default constructor
  redNode(){};

  //! Constructor:
  /*!
    input arguments:
    pointer to the blackNode parent, pointer to the tree, which child, range of belonging rows/cols
  */
  redNode( blackNode*, tree*, bool, int, int );

  //! Destructor
  ~redNode();

  //! Returns child_
  blackNode* child() const { return child_; }

  //! True if this is leaf node
  bool IsLeaf() const { return ( child_ == 0 ); }

  //! Return parent_
  blackNode* parent() const { return parent_; }

  //! Returns level_
  unsigned int level() const { return level_; }

  //! Returns which_
  bool which() const { return which_; }

  // Continue creating the tree
  void createBlackNode();

  // Returns &AdjList_
  std::set<redNode*>* AdjList() { return &AdjList_; }

  //! Returns pointer to it self
  redNode* redParent() { return this; }

  //! overloaded version for many levels up
  redNode* redParent(int);

};

#endif
