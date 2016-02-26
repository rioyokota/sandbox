#ifndef types_h
#define types_h
#include <complex>
#include <map>
#include <vector>
#include "vec.h"

// Basic type definitions
typedef float real_t;                                           //!< Floating point type is single precision
typedef std::complex<real_t> complex_t;                         //!< Complex type
typedef vec<4,int> ivec4;                                       //!< Vector of 4 integer types
typedef vec<2,real_t> vec2;                                     //!< Vector of 3 floating point types
typedef std::map<const char*,double> Timer;                     //!< Map of timer event name to timed value

// Multipole/local expansion coefficients
const int P = 6;                                                //!< Order of expansions
typedef vec<P,complex_t> vecP;                                  //!< Multipole/local coefficient type

//! Structures for defining bounding box
struct Box {
  vec2   X;                                                     //!< Box center
  real_t R;                                                     //!< Box radius
};
struct Bounds {
  vec2 Xmin;                                                    //!< Minimum value of coordinates
  vec2 Xmax;                                                    //!< Maximum value of coordinates
};

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
  int NCHILD;                                                   //!< Number of child cells
  int NBODY;                                                    //!< Number of descendant bodies
  Cell * CHILD;                                                 //!< Index of child cells
  long long ICELL;                                              //!< Cell index
  B_iter BODY;                                                  //!< Iterator of first body
  vec2 X;                                                       //!< Cell center
  real_t R;                                                     //!< Cell radius
  vecP M;                                                       //!< Multipole coefficients
  vecP L;                                                       //!< Local coefficients
};
typedef std::vector<Cell> Cells;                                //!< Vector of cells
typedef Cells::iterator C_iter;                                 //!< Iterator of cell vector

struct Node {
  B_iter BODY;                                                  //!< Iterator for first body in node
  int NBODY;                                                    //!< Number of descendant bodies
  int NNODE;                                                    //!< Number of descendant nodes
  Node * CHILD[4];                                              //!< Pointer to child node
  vec2 X;                                                       //!< Coordinate at center
  real_t R;                                                     //!< Cell radius
  vecP M;                                                       //!< Multipole coefficients
  vecP L;                                                       //!< Local coefficients
};


#endif
