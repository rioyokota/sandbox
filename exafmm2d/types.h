#ifndef types_h
#define types_h
#include <complex>
#include <map>
#include <vector>

// Basic type definitions
typedef float real_t;                                           //!< Floating point type is single precision
typedef std::complex<real_t> complex_t;                         //!< Complex type

// Multipole/local expansion coefficients
const int P = 6;                                                //!< Order of expansions

//! Structure of bodies
struct Body {
  real_t X[2];                                                  //!< Position
  real_t SRC;                                                   //!< Scalar source values
  real_t TRG;                                                   //!< Scalar+vector3 target values
};

//! Structure of cells
struct Cell {
  int NNODE;                                                    //!< Number of child cells
  int NBODY;                                                    //!< Number of descendant bodies
  Cell * CHILD[4];                                              //!< Index of child cells
  Body * BODY;                                                  //!< Iterator of first body
  real_t X[2];                                                  //!< Cell center
  real_t R;                                                     //!< Cell radius
  complex_t M[P];                                               //!< Multipole coefficients
  complex_t L[P];                                               //!< Local coefficients
};

#endif
