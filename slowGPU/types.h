#ifndef types_h
#define types_h
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <queue>
#include <vector>
#include "vec.h"
#include <sys/time.h>

typedef float              real;                                //!< Real number type on CPU
typedef vec<3,real>        vect;                                //!< 3-D vector type


int MPIRANK    = 0;                                             //!< MPI comm rank
int MPISIZE    = 1;                                             //!< MPI comm size
real THETA     = .5;                                            //!< Multipole acceptance criteria

const int  P        = 3;                                        //!< Order of expansions
const int  NCRIT    = 10;                                       //!< Number of bodies per cell
const real EPS2     = 0;                                        //!< Softening parameter (squared)

const int MTERM = P*(P+1)*(P+2)/6;                              //!< Number of Cartesian mutlipole terms
const int LTERM = (P+1)*(P+2)*(P+3)/6;                          //!< Number of Cartesian local terms

struct Body {
  int         ICELL;                                            //!< Cell index
  vect        X;                                                //!< Position
  real        SRC;                                              //!< Scalar source values
  vec<4,real> TRG;                                              //!< Scalar+vector target values
  bool operator<(const Body &rhs) const {                       //!< Overload operator for comparing body index
    return this->ICELL < rhs.ICELL;                             //!< Comparison function for body index
  }
};
typedef std::vector<Body>              Bodies;                  //!< Vector of bodies

struct Cell {
  int      ICELL;                                               //!< Cell index
  int      NCHILD;                                              //!< Number of child cells
  int      NCLEAF;                                              //!< Number of child leafs
  int      NDLEAF;                                              //!< Number of descendant leafs
  int      PARENT;                                              //!< Iterator offset of parent cell
  int      CHILD;                                               //!< Iterator offset of child cells
  int      LEAF;                                                //!< Iterator of first leaf
  vect     X;                                                   //!< Cell center
  real     R;                                                   //!< Cell radius
  real     RMAX;                                                //!< Max cell radius
  real     RCRIT;                                               //!< Critical cell radius
};
typedef std::pair<Cell*,Cell*>         Pair;                    //!< Pair of interacting cells
typedef std::deque<Pair>               PairQueue;               //!< Queue of interacting cell pairs

#endif
