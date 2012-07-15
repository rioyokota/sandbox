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
#include <xmmintrin.h>

typedef unsigned           bigint;                              //!< Big integer type
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

typedef vec<MTERM,real>                        Mset;            //!< Multipole coefficient type for Cartesian
typedef vec<LTERM,real>                        Lset;            //!< Local coefficient type for Cartesian

struct Body {
  int         IBODY;                                            //!< Initial body numbering for sorting back
  int         IPROC;                                            //!< Initial process numbering for partitioning back
  bigint      ICELL;                                            //!< Cell index
  vect        X;                                                //!< Position
  real        SRC;                                              //!< Scalar source values
  vec<4,real> TRG;                                              //!< Scalar+vector target values
  bool operator<(const Body &rhs) const {                       //!< Overload operator for comparing body index
    return this->ICELL < rhs.ICELL;                             //!< Comparison function for body index
  }
};
typedef std::vector<Body>              Bodies;                  //!< Vector of bodies
typedef std::vector<Body>::iterator    B_iter;                  //!< Iterator for body vector

struct Cell {
  bigint   ICELL;                                               //!< Cell index
  int      NCHILD;                                              //!< Number of child cells
  int      NCLEAF;                                              //!< Number of child leafs
  int      NDLEAF;                                              //!< Number of descendant leafs
  int      PARENT;                                              //!< Iterator offset of parent cell
  int      CHILD;                                               //!< Iterator offset of child cells
  B_iter   LEAF;                                                //!< Iterator of first leaf
  vect     X;                                                   //!< Cell center
  real     R;                                                   //!< Cell radius
  real     RMAX;                                                //!< Max cell radius
  real     RCRIT;                                               //!< Critical cell radius
  Mset     M;                                                   //!< Multipole coefficients
  Lset     L;                                                   //!< Local coefficients
};
typedef std::vector<Cell>              Cells;                   //!< Vector of cells
typedef std::vector<Cell>::iterator    C_iter;                  //!< Iterator for cell vector
typedef std::queue<C_iter>             CellQueue;               //!< Queue of cell iterators
typedef std::pair<C_iter,C_iter>       Pair;                    //!< Pair of interacting cells
typedef std::deque<Pair>               PairQueue;               //!< Queue of interacting cell pairs

#endif
