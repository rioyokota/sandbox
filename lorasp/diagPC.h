#ifndef diagPC_h
#define diagPC_h

#include "Eigen/Dense"
#include "Eigen/Sparse"

/*! A dense vector class used for RHS */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

/*! Define a sparse matrix (column major) */
typedef Eigen::SparseMatrix<double> spMat;

/***************************************************************/
/*                     Diagonal Preconditioner                 */
/***************************************************************/
//! This is the diagonal preconditioner.
class diagPC
{
  
  //! Size of the matrix
  int n_;

  //! inverse of diagonal entries
  VectorXd* invDiag_;
  VectorXd* x_;

 public:
  
  //! Default constructor
  diagPC(){};

  //! constructor
  /*!
    Takes pointer to the sparse matrix
   */
  diagPC( spMat* A )
    {
      n_ = A->rows();
      invDiag_ = new VectorXd(n_);
      x_ = new VectorXd(n_);
      (*invDiag_) = A->diagonal();
      for ( int i = 0; i < n_; i++ )
	{
	  (*invDiag_)(i) = 1. / (*invDiag_)(i);
	}
    }
  
  //! Destructor
  ~diagPC()
  {
    delete invDiag_;
    delete x_;
  }

  //! Solve function
  VectorXd& solve( VectorXd& b ) 
  { 
    for ( int i = 0; i < n_; i++ )
      {
	(*x_)(i) = (*invDiag_)(i)*b(i);
      }
    return *x_;
  }

};

#endif
