#ifndef edge_h
#define edge_h

#include "Eigen/Dense"

/*! Define a dynamic size dense matrix stored by edge */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

class node;

//! Class edge
/*!
  An edge can connect any two nodes (red, black, superNode).
  It has the data of the corresponding interaction block.
  Convention: Edge is created by its source node.
*/

class edge
{
  
  //! Pointer to the source node
  node* source_;

  //! Pointer to the destination node
  node* destination_;

  //! True if the edge is compressed
  bool compressed_;

 public:
  
  //! Default constructor
  edge(){};

  //! Construcotr
  /*!
    Constructor inputs are:
    pointer to the source node,
    pointer to the destination node,
   */
  edge( node*, node*);

  // Destructor
  ~edge();

  //! Returns pointer to the source node
  node* source() const { return source_; }

  //! Returns pointer to the destination node
  node* destination() const { return destination_; }

  //! Check if edge is between two spearated nodes
  /*!
    This function check if the edges source and separation are well separated.
    The adjacency list of nodes are used to determine that.
    Note that source and dest. can be different node types at different levels.
   */
  bool isWellSeparated();

  //! Pointer to the interaction matrix
  /*!
    a pointer to a dense m by n matrix
    note: n is the number of columns (variables at source node),
    and m is the number of rows ( equations at destination node).
  */
  densMat* matrix;

  //! Compress the edge
  /*! This function should be called after an edge is compressed, and moved to the parent level.
    As a result the following steps happen:
    - Set the compressed_ flag to [true]
    - Free the matrix memory
  */
  void compress();
    

  //! Check if the edge is eliminated
  /*!
    An edge is eliminated during the elimination process iff either its source or destination node is eliminated
   */
  bool isEliminated();

  //! Returns the compressed flag
  bool isCompressed() const{ return compressed_; }

};

#endif
