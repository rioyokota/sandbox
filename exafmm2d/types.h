#ifndef types_h
#define types_h
#include <complex>
#include <map>
#include <vector>
#include "vec.h"

// Basic type definitions
typedef float real_t;                                           //!< Floating point type is single precision
typedef std::complex<real_t> complex_t;                         //!< Complex type
typedef vec<2,real_t> vec2;                                     //!< Vector of 3 floating point types
typedef std::map<const char*,double> Timer;                     //!< Map of timer event name to timed value

// Multipole/local expansion coefficients
const int P = 6;                                                //!< Order of expansions
typedef vec<P,complex_t> vecP;                                  //!< Multipole/local coefficient type

//! Structure of bodies
struct Body {
  vec2   X;                                                     //!< Position
  real_t SRC;                                                   //!< Scalar source values
  real_t TRG;                                                   //!< Scalar+vector3 target values
};
typedef std::vector<Body> Bodies;                               //!< Vector of bodies
typedef Bodies::iterator B_iter;                                //!< Iterator of body vector

//! Structure of cells
struct Cell {
  int NNODE;                                                    //!< Number of child cells
  int NBODY;                                                    //!< Number of descendant bodies
  Cell * CHILD[4];                                              //!< Index of child cells
  B_iter BODY;                                                  //!< Iterator of first body
  vec2 X;                                                       //!< Cell center
  real_t R;                                                     //!< Cell radius
  vecP M;                                                       //!< Multipole coefficients
  vecP L;                                                       //!< Local coefficients
};

#endif
