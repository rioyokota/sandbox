#ifndef gmres_h
#define gmres_h

#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "time.h"

//! Sparse matrix type
/*! Define a sparse matrix (column major) */
typedef Eigen::SparseMatrix<double> spMat;

/*! Define a dynamic size dense matrix stored by edge */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> densMat;

/*! A dense vector class used for RHS */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;


/**************************************************************************/
/*                                CLASS GMRES                             */
/**************************************************************************/
//! Class gmres
/*!
This class get a sparse matrix, and a rhs, apply the GMRES method until convergence.
 */

template <class precond>
class gmres
{
  //! Maximum number of iterations
  int m_;
  
  //! Pointer to the preconditioner
  /*!
    Preconditioner supposed to have a function:
    x* solve( *rhs ): This function solve A x = rhs (approximately)
    and returns pointer to solution.
   */
  precond* precond_;

  //! size of the matrix
  int n_;

  //! total iterations applied
  int totalIters_;

  //! total time
  double totalTime_;

  //! required variables for the method:
  int k_;

  //! print per-iteration info
  bool verbose_;
  
  double beta0_;
  double beta_;
  double epsilon_;
  double norm_b_;

  VectorXd* z_;
  VectorXd* h_;
  VectorXd* p_;
  VectorXd* rho_;
  VectorXd* y_;
  VectorXd* r_;
  VectorXd* x0_;
  VectorXd* b_;
  VectorXd* x_;
  
  densMat* Q_;
  densMat* R_;
  densMat* G_;

  spMat* A_;
  
  //! QR factorization
  /*!
    This funciton will be called every iteration.
    The QR factorization on an upper Hessenberg matrix is done using Givens rotations.
   */
  void givensQR();

  //! Solve R * y = p
  void solveUpperTri();
  
 public:
  
  //! Default constructor
  gmres() {};

  //! Destructor
  ~gmres();

  //! Constructor with input parameters
  /*!
    Pointer to the sparse matrix.
    Pointer to the preconditioner.
    Pointer to the RHS vector.
    Pointer to the initial gauess vector.
    Maximum number of iterations.
    Accuracy threshold.
   */
  gmres( spMat*, precond*, VectorXd*, VectorXd*, int, double, bool );
  
  //! solve: start iteration until:
  /*!
    1) Krylov sub-space stops growing
    or
    2) Reach maximum number of iteration
    or
    3) Reach desired accuracy
   */
  void solve();

  //! Returns vector of residuals
  VectorXd* residuals();

  //! Returns solution
  VectorXd* retVal();

  //! Returns total number of itertions
  int totalIters() const { return totalIters_; }

  //! Returns total time to solve
  double totalTime() const { return totalTime_; }
};

template <class precond>
gmres< precond >::gmres( spMat* A, precond* P, VectorXd* rhs, VectorXd* x0, int m, double eps, bool verb )
{

  A_ = A;
  precond_ = P;
  n_ = A->cols(); // = A->rows()
  b_ = rhs;
  x0_ = x0;
  m_ = m;
  epsilon_ = eps;
  verbose_ = verb;

  //! Allocate memory to matrices and vectors
  h_ = new  VectorXd( m );
  p_ = new  VectorXd( m );
  rho_ = new  VectorXd( m );
  y_ = new  VectorXd( m );
  x_ = new  VectorXd( n_ );
  z_ = new  VectorXd( n_ );
  r_ = new  VectorXd( n_ );
  
  Q_ = new densMat( n_, m );
  R_ = new densMat( m, m );
  G_ = new densMat( m, 2 );
}

template <class precond>
gmres<precond>::~gmres()
{
  delete z_;
  delete h_;
  delete p_;
  delete rho_;
  delete y_;
  delete x_;
  delete r_;

  delete Q_;
  delete R_;
  delete G_;
}

template <class precond>
void gmres<precond>::solve()
{
  k_ = -1;
  (*r_) = (*b_) - (*A_) * (*x0_);
  (*z_) = precond_ -> solve( *r_ );
  beta0_ = z_ -> norm();
  beta_ = beta0_;
  
  clock_t start, finish;
  start = clock();
  
  while ( (beta_ > 0) && ( k_ < m_-2 ) )
    {
      k_ ++;
      
      Q_ -> col( k_ ) = ( 1. / beta_ ) * (*z_);
      
      (*r_) = (*A_) * Q_->col( k_ );
      (*z_) = precond_ -> solve( *r_ );
      
      for ( int i = 0; i <= k_; i++ )
	{
	  (*h_)(i) = z_ -> dot( Q_ -> col( i ) );
	  (*z_) -= (*h_)(i) * Q_ -> col( i );
	}

      beta_ = z_ -> norm();
      (*h_)( k_+1 ) = beta_;

      givensQR();

      if ( verbose_ )
	{
	  std::cout<<"   Residual at iteration "<<k_<<" = "<<(*rho_)( k_ )<<std::endl;
	}
      
      if ( (*rho_)( k_ ) < epsilon_ )
	{
	  break;
	}    
    }
  solveUpperTri();
  (*x_) = (*x0_) + ( Q_ -> block( 0, 0, n_, k_+1 ) ) * (y_ -> segment(0, k_+1 ));
  finish = clock();
  totalTime_ = double(finish-start)/CLOCKS_PER_SEC;
  totalIters_ = k_+1;
}

template <class precond>
VectorXd* gmres<precond>::retVal()
{
  return x_;
}

template <class precond>
void gmres<precond>::givensQR()
{
  R_ -> col( k_ ) = (*h_);
  p_ -> setZero();
  (*p_)(0) = beta0_;
  
  double f1,f2;
  
  for ( int i = 0; i < k_; i++ )
    {
      f1 = (*R_)(i,k_);
      f2 = (*R_)(i+1,k_);
      (*R_)(i,k_) = (*G_)(i,1) * f1 - (*G_)(i,0) * f2;
      (*R_)(i+1,k_) = (*G_)(i,0) * f1 + (*G_)(i,1) * f2;
      (*p_)(i+1) = (*G_)(i,0) * (*p_)(i);
      (*p_)(i) *= (*G_)(i,1);
    }

  double denom = std::sqrt( (*R_)(k_,k_) * (*R_)(k_,k_) + (*R_)(k_+1,k_) * (*R_)(k_+1,k_) );
  (*G_)(k_,0) = - (*R_)(k_+1,k_) / denom;
  (*G_)(k_,1) = (*R_)(k_,k_) / denom;

  (*R_)(k_,k_) = (*G_)(k_,1) * (*R_)(k_,k_) - (*G_)(k_,0) * (*R_)(k_+1,k_);
  (*R_)(k_+1,k_) = 0;
  
  (*rho_)(k_) = std::abs( (*G_)(k_,0) * (*p_)(k_) ) / beta0_;
  (*p_)(k_) *= (*G_)(k_,1);
  
}

template <class precond>
void gmres<precond>::solveUpperTri()
{
  double backProp = 0;
  for ( int i = k_; i >=0; i-- )
    {
      if ( i < k_ )
	{
	  backProp = ( ( R_ -> row( i ) ).segment( i+1, k_-i ) ).dot( y_ -> segment( i+1, k_-i ) );
	}
      (*y_)(i) = ( (*p_)(i) - backProp ) / (*R_)(i,i);
    }
}

template <class precond>
VectorXd* gmres<precond>::residuals()
{
  return rho_;
}

#endif
