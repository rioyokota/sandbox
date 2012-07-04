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
#include <queue>
#include "cudavec.h"
#include "pair.h"
#include "stack.h"
#include "vec.h"

typedef float real;
typedef vec<3,real> vec3;
typedef vec<4,real> vec4;

const int  P     = 3;
const int  NCRIT = 10;
const real EPS2  = 0;
const real THETA = .6;

const int MTERM = P*(P+1)*(P+2)/6;
const int LTERM = (P+1)*(P+2)*(P+3)/6;
typedef vec<MTERM,real> vecM;
typedef vec<LTERM,real> vecL;

struct Cell {
  int  ICELL;
  int  NCHILD;
  int  NCLEAF;
  int  NDLEAF;
  int  PARENT;
  int  CHILD;
  int  LEAF;
  vec3 X;
  real R;
  real RMAX;
  real RCRIT;
};
typedef Pair<Cell*,Cell*> CellPair;
typedef Stack<100,CellPair> PairStack;

#endif
