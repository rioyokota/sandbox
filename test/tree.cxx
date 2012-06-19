#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <vector>

typedef long bigint;

struct Body {
//  int    IBODY;
//  int    IPROC;
  bigint ICELL;
  float  X[3];
//  float  SRC[1];
//  float  TRG[4];
  bool operator<(const Body &rhs) const {
    return this->ICELL < rhs.ICELL;
  }
};
typedef std::vector<Body> Bodies;
typedef std::vector<Body>::iterator B_iter;

struct Cell {
  unsigned NCHILD;
  unsigned NCLEAF;
  unsigned NDLEAF;
  unsigned PARENT;
  unsigned CHILD;
  B_iter   LEAF;
  float    X[3];
  float    R;
  float    RCRIT;
//  float    M[55];
//  float    L[55];
};
typedef std::vector<Cell> Cells;
typedef std::vector<Cell>::iterator C_iter;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

inline void getIndex(Bodies &bodies, int level) {
  float d = 1.0 / (1 << level);
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    int ix = B->X[0] / d;
    int iy = B->X[1] / d;
    int iz = B->X[2] / d;
    int id = 0;
    for( int l=0; l!=level; ++l ) {
      id += ix % 2 << (3 * l);
      id += iy % 2 << (3 * l + 1);
      id += iz % 2 << (3 * l + 2);
      ix >>= 1;
      iy >>= 1;
      iz >>= 1;
    }
    B->ICELL = id;
  }
}

void bodies2twigs(Bodies &bodies, Cells &cells, int level) {
  int I = -1;
  C_iter C;
  cells.reserve(1 << (3 * level));
  float d = 1.0 / (1 << level);
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    int IC = B->ICELL;
    int ix = B->X[0] / d;
    int iy = B->X[1] / d;
    int iz = B->X[2] / d;
    if( IC != I ) {
      Cell cell;
      cell.NCHILD = 0;
      cell.NCLEAF = 0;
      cell.NDLEAF = 0;
      cell.CHILD  = 0;
      cell.LEAF   = B;
      cell.X[0]   = d * (ix + .5);
      cell.X[1]   = d * (iy + .5);
      cell.X[2]   = d * (iz + .5);
      cell.R      = d * .5;
      cells.push_back(cell);
      C = cells.end()-1;
      I = IC;
    }
    C->NCLEAF++;
    C->NDLEAF++;
  }
}

void twigs2cells(Bodies &bodies, Cells &cells, int level) {
  int begin = 0, end = cells.size();
  float d = 1.0 / (1 << level);
  for( int l=1; l!=level; ++l ) {
    int div = (1 << (3 * l));
    d *= 2;
    int I = -1;
    int p = end - 1;
    for( int c=begin; c!=end; ++c ) {
      B_iter B = cells[c].LEAF;
      int IC = B->ICELL / div;
      int ix = B->X[0] / d;
      int iy = B->X[1] / d;
      int iz = B->X[2] / d;
      if( IC != I ) {
        Cell cell;
        cell.NCHILD = 0;
        cell.NCLEAF = 0;
        cell.NDLEAF = 0;
        cell.CHILD  = c - begin;
        cell.LEAF   = cells[c].LEAF;
        cell.X[0]   = d * (ix + .5);
        cell.X[1]   = d * (iy + .5);
        cell.X[2]   = d * (iz + .5);
        cell.R      = d * .5;
        cells.push_back(cell);
        p++;
        I = IC;
      }
      cells[p].NCHILD++;
      cells[p].NDLEAF += cells[c].NDLEAF;
      cells[c].PARENT = p;
    }
    begin = end;
    end = cells.size();
  }
}

int main() {
  const int numBodies = 10000000;
  const int level = 7;
  double tic, toc;
  tic = get_time();
  Bodies bodies(numBodies);
  Cells cells;
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    for( int d=0; d!=3; ++d ) B->X[d] = drand48();
  }
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  tic = get_time();
  getIndex(bodies,level);
  toc = get_time();
  std::cout << "mort : " << toc-tic << std::endl;

  tic = get_time();
  std::sort(bodies.begin(),bodies.end());
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;

  tic = get_time();
  bodies2twigs(bodies,cells,level);
  toc = get_time();
  std::cout << "twig : " << toc-tic << std::endl;

  tic = get_time();
  twigs2cells(bodies,cells,level);
  toc = get_time();
  std::cout << "cell : " << toc-tic << std::endl;
}
