#ifndef params_h
#define params_h
#include <string>
#include <math.h>

//! Parameters class
/*!
  This class loads an input parameter file, and store different parameters.
  All other classes can access to the parameters using this class.
 */

class params
{
  
  //! Path to the input matrix (in matrix market format)
  /* Note that we assume that the input matrix has indexing	\
     Started from 1. */
  std::string Input_Matrix_File_;

  //! Subdividing threshold
  unsigned int treeLevelThreshold_;

  //! Threshold for compression
  double epsilon_;

  //! Method for compression (0:SVD, 1:rSVD)
  int lowRankMethod_;

  //! Method for compression cut-off
  int cutOffMethod_;

  //! A priori rank of compression (used for rSVD or constant rank cutOff method)
  int aPrioriRank_;

  //! Rank cap geometric series factor ( 0 == No cap )
  double rankCapFactor_;

  //! Deploy factor for rSVD (portion of computed singular values to be used)
  double deployFactor_;

  //! GMRES maximum number of iterations
  double gmresMaxIters_;

  //! GMRES residual threshold
  double gmresEpsilon_;

  //! GMRES preconditioner
  int gmresPC_;

  //! GMRES verbose
  bool gmresVerbose_;

  //! ILU drop tol.
  double ILUDropTol_;

  //! ILU Fill
  int ILUFill_;

  //! Divide cols by max entry?
  bool normCols_;
  
 public:
  
  //! Public function to access to input matrix file
  std::string Input_Matrix_File() const{ return Input_Matrix_File_; }
  
  //! Default constructor
  params(){}
  
  //! Constructor
  /*! Should construct a parameter object with the char* of	\
    the path to the input parameter file */
  /*! In the input parameter file, empty line and lines starting	\
    with # will be ignored */
  params(char*);

  //! Constructor
  /*!
    With this constructor we directly provide parameters
  */
  params(char*, int, int, int, double, int, double, double, int, double, int, double, int, bool);

  //! Destructor
  ~params(){}

  //! Returns subdividing_threshold_
  unsigned int treeLevelThreshold() const { return treeLevelThreshold_; }

  //! Returns epsilon_
  double epsilon() const{ return epsilon_; }

  //! Returns lowRankMethod_
  int lowRankMethod() const{ return lowRankMethod_; }

  //! Returns cutOffMethod_
  int cutOffMethod() const{ return cutOffMethod_; }

  //! Returns aPrioryRank_
  double aPrioriRank() const{ return aPrioriRank_; }

  //! Returns rankCapFactor_
  double rankCapFactor() const{ return rankCapFactor_; }

  //! Returns deployFactor_
  double deployFactor() const{ return deployFactor_; }

  //! Returns GMRES maximum number of iterations
  int gmresMaxIters() const{ return gmresMaxIters_; }

  //! Returns GMRES residual threshold
  double gmresEpsilon() const{ return gmresEpsilon_; }

  //! Returns GMRES preconditioner
  int gmresPC() const{ return  gmresPC_; }

  //! Returns GMRES verbose
  bool gmresVerbose() const{ return  gmresVerbose_; }

  //! Returns ILU drop tol.
  double ILUDropTol() const{ return  ILUDropTol_; }

  //! Returns ILU Fill
  int ILUFill() const{ return ILUFill_;}

  //! Returns normCols_
  int normCols() const{ return normCols_;}

};

#endif
