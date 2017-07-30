#ifndef node_h
#define node_h

class tree;
class redNode;
class blackNode;
class superNode;
class edge;

#include <vector>
#include "Eigen/Dense"

/*! A dense vector class used for RHS */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

/*! Define a dynamic size dense matrix used to store the inverse of pivot */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

/**************************************************************************/
/*                           CLASS NODE (Abstract)                        */
/**************************************************************************/

//! Class node
/*!
  This (virtual) class is an interface for the three inherited classes
  blackNode, redNode, and superNode
*/

class node
{

  //! Pointer to the tree that this node belongs to
  tree* tree_ptr_;

  //! Pointer to the parent node
  node* parent_;

  //! Index of the first column in the matrix (inclusive) correspondig to this node
  int index_first_;

  //! Index of the last column in the matrix (inclusive) correspondig to this node
  int index_last_;

  //! Indicate number of variables in this node
  /*!
    Note that for a leaf redNode [n] is the number of columns of the matrix corresponding to this node.
    However, for redNodes upper in the tree, [n] is determined by low rank approximation.
   */
  int n_;

  //! Indicate number of equations in this node
  /*!
    The number of euqations [m], is also equal to the number of corresponding rows for leaf redNodes.
    But for other redNodes, it depends on the rank in our low-rank decomposition.
   */
  int m_;

  //! The right hand side vector
  VectorXd *RHS_;

  //! The vector of unknowns
  VectorXd *VAR_;

 public:
  
  //! Default constructor
  node(){};

  //! Constructor:
  /*!
    input arguments:
    pointer to parent, pointer to the tree, range of belonging rows/cols
  */
  //! Constructor by passing the ptr to the tree, and range of belonging columns
  node( node*, tree*, int, int );

  //! Destructor
  virtual ~node();

  //! Returns parent_
  node* parent() const { return parent_; }
   
  //! Returns index_first_
  int IndexFirst() const { return index_first_; }

  //! Returns index_last_
  int IndexLast() const { return index_last_; }  

  //! Returns tree_ptr_
  tree* Tree() const { return tree_ptr_; }

  //! True if this node contains no column of the matrix
  bool isEmpty() { return index_first_>index_last_; }

  //! True if this node is eliminated
  bool isEliminated() const { return eliminated_; }

  //! Set the elimination flag to [true]
  void eliminate();
  
  //! Set the elimination flag to [false]
  void deEliminate();

  //! Returns m_
  int m() const { return m_; }

  //! Returns n_
  int n() const { return n_; }

  //! Set m_
  void m( int val ) { m_ = val; }

  //! Set n_
  void n( int val ) { n_ = val; }

  //! Returns pointer to the redParent (pure virtual)
  /*!
    redNode -> returns itself
    blackNode -> returns parent()
    superNode -> returns parent() of paranet()
   */
  virtual redNode* redParent()=0;
  
  //! Overload: int is the lvel of grand parent
  virtual redNode* redParent(int)=0;

  //! List of incoming edges
  std::vector<edge*> inEdges;

  //! List of outgoing edges
  std::vector<edge*> outEdges;

  //! A boolean flag to keep track of elimination
  bool eliminated_;

  //! Erase removed edges from the list of incoming/outgoing edges
  void eraseCompressedEdges();

  //! Access to the pointer of the RHS vector
  VectorXd* RHS() { return RHS_; }

  //! Set the pointer of the RHS vector
  void RHS( VectorXd* in ) { RHS_ = in; }

  //! Access to the pointer of the variables vector
  VectorXd* VAR() { return VAR_; }

  //! Set the pointer of the VAR vector
  void VAR( VectorXd* in ) { VAR_ = in; }

  //! The inevrse of the selfEdge matrix (i.e., pivot)
  densMat *invPivot;

  //! solve L z = b
  /*!
    This function updates RHS, which is solve for z in L z = b
    It uses the order of elimination.
   */
  void solveL();

  //! solve U VAR = RHS
  /*!
    Solve for unknowns of this cluster.
    It uses the order of elimination.
   */
  void solveU();
  
  //! A boolean flag to keep track of updated RHS
  bool rhsUpdated_;

  //! The order of elimination
  int order;
}; 

#endif
