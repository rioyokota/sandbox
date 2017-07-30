#ifndef blackNode_h
#define blackNode_h

#include "node.h"
#include "Eigen/Dense"
#include <array>

class tree;
class redNode;
class blackNode;
class superNode;

/*! Define a dynamic size dense matrix stored by edge */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

/*! Define a type pointer to dynamic size dense matrix stored by edge */
typedef densMat* densMatStr;

//! List of four dense matrices
/*!
  This type is used to  merge two sibling redNodes.
  Order of matrices: Ls-Ld, Rs-Ld, Ls-Rd, Rs-Rd
  Here we assumed (L)eft and (R)ight children of (S)ource and (D)estination are merged.
 */
typedef std::array<densMatStr, 4> densMatStr4;

/*! Tuple of times for (time1, time2) */
typedef std::array<double, 2> timeTuple2;

/**************************************************************************/
/*                            CLASS Black NODE                            */
/**************************************************************************/

//! Class blackNode
/*!
  Inherited from the general class node. It is a node correponding to the local variables, 
  and multipole equations.
*/

class blackNode:public node
{

  //! Pointer to the parent (which is a redNode)
  redNode* parent_;

  //! Pointer to its left redNode child
  redNode* leftChild_;

  //! Pointer to its right redNode child
  redNode* rightChild_;

  //! Pointer to its superNode child
  superNode* superChild_;

  //! The outer indices of the sparse sub-matrix corresponds to this node
  int* outer_index_ptr_;
  
  //! The inner indices of the sparse sub-matrix corresponds to this node
  int* inner_index_ptr_;

   public:
  
  //! Default constructor
  blackNode(){};

  //! Constructor:
  /*!
    input arguments:
    pointer to the redNode parent, pointer to the tree, range of belonging rows/cols
  */
  blackNode( redNode*, tree*, int, int );

  //! Destructor
  ~blackNode();

  //! Returns parent_
  redNode* parent() const { return parent_; }

  //! Returns outer_index_ptr_
  int* OuterIndex() const { return outer_index_ptr_; }

  //! Returns inner_index_ptr_
  int* InnerIndex() const { return inner_index_ptr_; }

  //! Returns pointer to the left child
  redNode* leftChild() const { return leftChild_; }
  
  //! Returns pointer to the right child
  redNode* rightChild() const { return rightChild_; }

  //! Returns pointer to the super child
  superNode* superChild() const { return superChild_; }

  //! Returns pointer to its parent
  redNode* redParent() { return parent();}

  //! overloaded version for many levels up
  redNode* redParent(int);

  //! Merge left and right redNodes to a superNode
  /*!
    Output is the time elapsed.
   */     
  double mergeChildren();

  //! Merge RHS of children
  void mergeRHS();

  //! Apply schur-complement
  /*!
    This function is called immediately after its superNode child is eliminated.
   */
  timeTuple2 schurComp();
  
};


#endif
