#ifndef tree_h
#define tree_h
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <map>
#include <string>
#include<array>

class params;
class node;
class redNode;
class superNode;

//! TRIPLET Type
/*! Define a triplet type to store entries of a matrix */
typedef Eigen::Triplet<double> TRIPLET;

/*! Define a triplet type to store symbolic matrix */
typedef Eigen::Triplet<bool> TRIPLETBOOL;

//! Sparse matrix type
/*! Define a sparse matrix (column major) */
typedef Eigen::SparseMatrix<double> spMat;

/*! Define a sparse matrix (column major) with boolean entries */
typedef Eigen::SparseMatrix<bool> spMatBool;
 
/*! Define a redNode* list */
typedef std::vector<redNode*> redNodeStrList;

/*! Define a superNode* list */
typedef std::vector<superNode*> superNodeStrList;

/*! A dense vector class used for permutation */
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;

/*! The permutation matrix type */
typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,  int> permMat;

/*! Define a dynamic size dense matrix stored by edge */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

/*! A dense vector class used for RHS */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

/*! Tuple of times for (time1, time2) */
typedef std::array<double, 2> timeTuple2;

/*! Tuple of times for (compression (svd), compression (rest), sChurComp (inverse), sChurComp (rest)) */
typedef std::array<double, 4> timeTuple4;

//! Class tree
//! This class represents a tree of nodes, and provide necessary information/routines for tree
//! The full matrix is also stored as a member of this class, only one version that is globally accessible.
class tree
{

  /***************************************************/
  /*     TREE RELATED VARIABLES/FUNCTION (Private)   */
  /***************************************************/
  
  //! pointer to the parameter class
  params *param_;

  //! pointer to the root node
  node* root_;

  //! pointer to the matrix (and a temporary for permutation)
  spMat* matrix_;
  spMat* permuted_matrix_;

  //! pointer to solution (considered dense)
  VectorXd* VAR_;

  //! pointer to RHS (considered dense)
  VectorXd* RHS_;

  //! pointer to the symbolic matrix (and a temporary for permutation)
  spMatBool* symb_matrix_;
  spMatBool* permuted_symb_matrix_;

  //! A permutation matrix used once at each level
  permMat* permutationMatrix;

  //! Size of the matrix
  int n_;

  //! Number of non-zero entries in the matrix
  int nnz;

  //! The list of triplets (matrix entries)
  std::vector<TRIPLET> TripletList_;

  //! The list of triplets (symbolic matrix)
  std::vector<TRIPLETBOOL> TripletBoolList_;
  
  //! The list of pointers of red nodes at each level
  std::vector<redNodeStrList> redNodeList_;

  //! The list of pointers of superNodes at each level
  std::vector<superNodeStrList> superNodeList_;

  //! The list containing mean rank at each level
  std::vector<double> meanRanks_;

  //! The list containing minimum rank at each level
  std::vector<int> minRanks_;

  //! The list containing maximum rank at each level
  std::vector<int> maxRanks_;

  //! Use permutationVector and permute the full matrix
  void permuteMatrix();

  //! A map from the permuted matrix columns to the leaves of the tree
  std::map <int,redNode*> col2Leaf_;

 public:

  /***************************************************/
  /*     TREE RELATED VARIABLES/FUNCTION (Public)    */
  /***************************************************/
  
  //! The vecotr contains required permutation of the last level
  VectorXi* permutationVector;
  
  //! Constructor (for root node)
  /*! 
    In the constructor we read the matrix data from input files (in market-matrix format),
    and store it as EigenSparse matrix.
    Also, the nodes of the tree will be generated in the constructor
  */

  //! Default constructor
  tree(){};

  //! Constructor with parameters as input
  tree( params* );

  //! Destructor
  ~tree(){};

  //! Provide a submatrix with given range of columns
  /*!
    The order of input arguments:
    first column, last column, InnerIndex list, OuterIndex liset
    Note: The indices in the in/out-er lists always start from 0!
    Returns 0 if successful, returns 1 otherwise
   */

  //! Returns param_
  params* Parameters() const { return param_; }
  
  //! Returns matrix_
  spMat* Matrix() const { return matrix_; }

  //! Returns symb_matrix_
  spMatBool* SymbMatrix() const { return symb_matrix_; }

  // Returns size of the matrix n_
  int n() const { return n_; }

  //! Returns pointer to the list of redNodes
  std::vector<redNodeStrList>* redNodeList() { return &redNodeList_;}

  //! Returns pointer to the list of superNodes
  std::vector<superNodeStrList>* superNodeList() { return &superNodeList_;}

  //! Returns pointer to mean rank list
  std::vector<double>* meanRanks() { return &meanRanks_;}

  //! Returns pointer to min rank list
  std::vector<int>* minRanks() { return &minRanks_;}

  //! Returns pointer to max rank list
  std::vector<int>* maxRanks() { return &maxRanks_;}

  //! Tree assembling time
  double assembleTime;

  //! Time for preconditioner factorization time ( solve )
  double precondFactTime;

  //! GMRES final relative accuracy
  double accuracy;

  //! GMRES final relative residual
  double residual;

  //! GMRES residuals
  VectorXd* residuals;

  //! GMRES total iterations
  int gmresTotalIters;
  
  //! GMRES total time
  double gmresTotalTime;

  //! Padding happend?
  bool padding;

  //! Large pivots happend?
  bool largePivot;

  //! Returns the maximum level number
  unsigned int maxLevel() { return redNodeList_.size()-1; }

  //! Add a new redNode to the list
  /*!
    The blackNode will use this function to create its red children.
    The first argument is the level of the new red node and the second is the pointer to it
  */
  void addRedNode( unsigned int, redNode*);

  //! Add a new superNode to the list
  /*!
    The blackNode will use this function to create its superNode child.
    The first argument is the level of the new superNode and the second is the pointer to it
  */
  void addSuperNode( unsigned int, superNode*);

  //! Form the adjacency list for all redNodes
  /*! 
    Note that this is the list of original interactions induced by children.
    Later on during the elimination, nodes may have new interactions, and will be stored through edges.
    Only symbolic matrix will be used here.
  */
  void createAdjList();

  //! Using a Map we create key-value pairs for all leaf nodes
  /*!
    For each non-empty leaf the key is the index of the first columns it posses,
    and the value is the pointer to the leaf.
   */
  void createCol2LeafMap();

  //! Load the data to edges between tree leaves
  /*!
    After the tree, and adjacency lists are created,
    load the sub block of matrix to the edges between leaves.
  */
  void createLeafEdges();

  //! Creating superNodes
  /*!
    This function combine sibling redNodes at level [l] of the tree, and create a new superNode.
    By construction, the parent of the resulted superNode is a blackNode.
    Note that in order to acess pairs of sibling redNodes at level [l] of the tree,
    we go through the redNodes at level [l-1] in the tree, and look at their grand red children.
    Output is the time elapsed: tuple( compression, schurComp ).
   */
  double createSuperNodes( unsigned int );

  //! Eliminate variables at level [l] in the tree
  /*!
    Output is the time elapsed.
   */
  timeTuple4 eliminate( unsigned int );

  //! Set the RHS for all nodes
  void setRHS( VectorXd& );

  //! Dedicate memory, and set the RHS for leaf nodes
  void setLeafRHS( VectorXd& );

  //! Dedicate memory, and set the VAR for leaf nodes
  void setLeafRHSVAR( );

  //! SolveL for black, super, and red Nodes in a level
  /*!
    This functions work from bottom to top
   */
  void solveL( unsigned int ) ;  

  //! SolveU for black, super, and red Nodes in a level
  /*!
    This functions work from top to the bottom
   */
  void solveU( unsigned int ) ;

  //! Top to bottom traverse to solve for a given RHS
  /*!
    If no RHS is provided it uses the RHS from input params, and automatically permute it.
    For the overloaded  version, it takes a permuted RHS as an input.
   */
  VectorXd& solve( );
  VectorXd& solve( VectorXd& );

  //! Bottom to Top traverse to decompose A = L U 
  void factorize();

  //! Compute solution
  /*!
    i.e., collect the solution of all leaves
  */
  VectorXd& computeSolution();

  //! Access to the pointer of the variables vector
  VectorXd* VAR() { return VAR_; }

  //! set ranks
  /*!
    At each level compute mean, min, and max size of the redNodes
   */
  void setRanks();

  //! store everything in a log file
  void log( std::string );

  //! Store Forb. of each level
  /*!
    The frob. norm of level l is defined as:
    sqrt( sum_{e is edge between two superNodes} e.matrix.frob^2)
   */
  std::vector<double> frobNorms;

  //! Totoal size of each level
  /*!
    For each level, l, it computes:
    sum of sizes (= m*n for a m-by-n matrix) of all blocks in that level
   */
  std::vector<double> totalSizes;
  
  //! global frob norm = sum(frobNorms)
  double globFrobNorm;

  //! global size = sum(totalSizes)
  double globSize;

  //! Compute the Frob. norm at level l
  void computeFrobNorm( unsigned int );

  //! COunt the number of eliminated nodes (so far)
  int count;

  //! Returns a pointer to the solution
  VectorXd* retVal();

  //! Divide each col of the matrix by its maximum value
  void normalize_cols( spMat* );

};
#endif
