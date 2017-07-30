#ifndef eyePC_h
#define eyePC_h

#include "Eigen/Dense"

/*! A dense vector class used for RHS */
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

/***************************************************************/
/*                     Identity Preconditioner                 */
/***************************************************************/
//! This is the default preconditioner.
/*!
  Essentialy, this precondiotner = no preconditioner!
 */
class eyePC
{
  
 public:
  
  //! Default constructor
  eyePC(){};
  
  //! Destructor
  ~eyePC(){};

  //! Solve function
  VectorXd& solve( VectorXd& b ) { return b; }

};

#endif
