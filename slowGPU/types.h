#ifndef types_h
#define types_h
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
#include "pair.h"
#include "stack.h"
#include "vec.h"

typedef float       real;                                       //!< Real number type on CPU
typedef vec<3,real> vect;                                       //!< 3-D vector type

const int  P     = 3;                                           //!< Order of expansions
const int  NCRIT = 10;                                          //!< Number of bodies per cell
const real EPS2  = 0;                                           //!< Softening parameter (squared)
const real THETA = .5;                                          //!< Multipole acceptance criteria

const int MTERM = P*(P+1)*(P+2)/6;                              //!< Number of Cartesian mutlipole terms
const int LTERM = (P+1)*(P+2)*(P+3)/6;                          //!< Number of Cartesian local terms

struct Cell {
  int  ICELL;                                                   //!< Cell index
  int  NCHILD;                                                  //!< Number of child cells
  int  NCLEAF;                                                  //!< Number of child leafs
  int  NDLEAF;                                                  //!< Number of descendant leafs
  int  PARENT;                                                  //!< Iterator offset of parent cell
  int  CHILD;                                                   //!< Iterator offset of child cells
  int  LEAF;                                                    //!< Iterator of first leaf
  vect X;                                                       //!< Cell center
  real R;                                                       //!< Cell radius
  real RMAX;                                                    //!< Max cell radius
  real RCRIT;                                                   //!< Critical cell radius
};
typedef Pair<Cell*,Cell*> CellPair;                             //!< Pair of interacting cells
typedef Stack<100,CellPair> PairStack;                          //!< Queue of interacting cell pairs

#endif
