#ifndef superNode_h
#define superNode_h

#include "node.h"
#include "blackNode.h"
#include "Eigen/Dense"
#include <vector>
#include "rsvd.h"
#include <array>

class tree;
class redNode;
class blackNode;
class superNode;

/*! Define a dynamic size dense matrix stored by edge */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

/*! Define a type pointer to dynamic size dense matrix stored by edge */
typedef densMat* densMatStr;

/*! A dense vector class used for permutation */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

/*! Tuple of times for (time1, time2) */
typedef std::array<double, 2> timeTuple2;

/**************************************************************************/
/*                           CLASS Super NODE                             */
/**************************************************************************/

//! Class superNode
/*!
  Inherited from the general class node. It is a node correponding to particles
*/

class superNode:public node
{

  //! zero
  double MACHINE_PRECISION;

  //! Pointer to the parent (which is a blackNode)
  blackNode* parent_;

  //! The level of node ( only redNodes have level)
  unsigned int level_;

  //! cut-off criterion for compression
  /*!
    It is used when deciding what singular values are important.
    It gets the vector of singular values, and returns k, the number of important singular values.
    Second input argument is the method.
    Third argument is the epsilon for cutoff
    Fourth argument is a given constant rank (for method 8)
    Last argument is the growth factor for rank cap (geometric series). Zero means no cap.
  */
  int cutOff( Eigen::JacobiSVD<densMat>&, int, double, int, double );

  //! Temporary list of interpolation matrices(U)
  /*!
    Every incoming well-separated interaction after being compressed results in a U.
    Later we do the recompression step and end up with only one U.
   */
  std::vector<densMatStr> matrixUList;

  //! Temporary list of anterpolation matrices(V)
  /*!
    Every outgoing well-separated interaction after being compressed results in a V.
    Later we do the recompression step and end up with only one V.
   */
  std::vector<densMatStr> matrixVList;
  
  //! Threshold for compression
  double epsilon_;
  
  //! Method for compression (0:SVD, 1:rSVD)
  int lowRankMethod_;

  //! Method for compression cut-off
  int cutOffMethod_;

  //! A priori rank of compression (used for rSVD or constant rank cutOff method)
  int aPrioriRank_;

  //! Rank cap factor for compression (factor in a geom. series), zero means no cap
  double rankCapFactor_;

  //! rSVD deploy factor for compression
  double deployFactor_;
  
  //! Pointer to the frob norm list of the tree
  double* frobNorms_;

  //! Pointer to the frob norm of the full matrix
  double* globFrobNorm_;

  //! Pointer to the size of the full matrix
  double* globSize_;

  //! Pointer to the size list of the tree
  double* totalSizes_;

  //! total depth of the tree
  int treeDepth_;

 public:
  
  //! Default constructor
  superNode(){}
  
  //! Constructor:
  /*!
    input arguments:
    pointer to the blackNode parent, pointer to the tree, range of belonging rows/cols
  */
  superNode( blackNode*, tree*, int, int );

  //! Destructor
  ~superNode();

  //! Returns parent_
  blackNode* parent() const { return parent_; }

  //! Returns pointer to its parent
  redNode* redParent() { return parent_->parent();}
  
  //! overloaded version for many levels up
  redNode* redParent(int);

  //! Returns level_
  unsigned int level() const { return level_; }

  //! Compress all well separated interactions
  /*!
    Exit is a timeTuple: (lowRank time, everything else time)
   */
  timeTuple2 compress();

  //! Actually eliminate the node, and its parent
  /*!
    Eliminating a node invloves going through all edges, and create new edges based on schur complement
   */
  timeTuple2 schurComp();

  //! Split the solution to left/right redNodes
  void splitVAR();
  
  //! Pad some random columns to the matrix (on the right)
  void addRandomCols( densMat&, int );

  //! Pad some random rows to the matrix (at the bottom)
  void addRandomRows( densMat&, int );

  //! Pad some random columns and apply QR (on the right )
  /*!
    We assume the input matrix consists of orthonormal columns.
   */
  void extendOrthoCols( densMat&, int );

  //! Pad some random rows and apply QR (at the bottom )
  /*!
    We assume the input matrix consists of orthonormal rows.
   */
  void extendOrthoRows( densMat&, int );

  //! check if the approximation is fine
  bool criterionCheck( int, double, RedSVD::RedSVD<densMat>&, densMat&, int );


};

#endif
