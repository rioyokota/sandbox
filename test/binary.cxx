#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include "vtk.h"

typedef long bigint;
typedef std::vector<int> Index;
typedef std::vector<int>::iterator I_iter;
struct Position {
  vect X;
};
typedef std::vector<Position> Positions;
typedef std::vector<Position>::iterator P_iter;
bool compareX(Position Pi, Position Pj) { return (Pi.X[0] < Pj.X[0]); }
bool compareY(Position Pi, Position Pj) { return (Pi.X[1] < Pj.X[1]); }
bool compareZ(Position Pi, Position Pj) { return (Pi.X[2] < Pj.X[2]); }

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  const int numBodies = 10000000;
  const int NCRIT = 10;
  double tic, toc;
  tic = get_time();
  Bodies bodies(numBodies);
  Index index(numBodies);
  Positions positions(numBodies);
  Cells cells;
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  srand48(1);
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
    for( int d=0; d!=3; ++d ) B->X[d] = drand48();
  }
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  tic = get_time();
  P_iter P = positions.begin();
  for( B_iter B=bodies.begin(); B!=bodies.end(); ++B, ++P ) {
    P->X[0] = B->X[0];
    P->X[1] = B->X[1];
    P->X[2] = B->X[2];
  }
  toc = get_time();
  std::cout << "copy : " << toc-tic << std::endl;

  tic = get_time();
  int level = numBodies >= NCRIT ? 1 + int(log(numBodies / NCRIT)/M_LN2) : 0;
  index[0] = 0;
  index[1] = positions.size()/2;
  index[2] = positions.size();
  for( int l=0; l!=level; ++l ) {
    std::cout << l << std::endl;
    P_iter P0 = positions.begin();
    if( l % 3 == 0 ) {
      for( int i=0; i!=(1 << l); ++i ) {
        std::nth_element(P0+index[2*i],P0+index[2*i+1],P0+index[2*i+2],compareX);
      }
    } else if( l % 3 == 1 ) {
      for( int i=0; i!=(1 << l); ++i ) {
        std::nth_element(P0+index[2*i],P0+index[2*i+1],P0+index[2*i+2],compareY);
      }
    } else {
      for( int i=0; i!=(1 << l); ++i ) {
        std::nth_element(P0+index[2*i],P0+index[2*i+1],P0+index[2*i+2],compareZ);
      }
    }
    for( int i=(2 << l); i>0; --i ) {
      index[2*i] = index[i];
    }
    for( int i=(2 << l); i>0; --i ) {
      index[2*i-1] = (index[2*i-2] + index[2*i]) / 2;
    }
  }
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;

  tic = get_time();
  for( int i=0; i!=(1 << level); ++i ) {
    for( int b=index[2*i]; b!=index[2*i+2]; ++b ) {
      bodies[b].X[0] = positions[b].X[0];
      bodies[b].X[1] = positions[b].X[1];
      bodies[b].X[2] = positions[b].X[2];
      bodies[b].ICELL = i;
    }
  }
  toc = get_time();
  std::cout << "set B: " << toc-tic << std::endl;

#if 0
  real R0 = 0.5;
  vect X0 = 0.5;
  int Ncell = 0;
  vtkPlot vtk;
  vtk.setDomain(R0,X0);
  vtk.setGroupOfPoints(bodies,Ncell);
  vtk.plot(Ncell);
#endif
}
