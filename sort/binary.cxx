#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include "types.h"

typedef std::vector<vec3> Xvector;
typedef std::vector<vec3>::iterator X_iter;
bool compareX(vec3 Xi, vec3 Xj) { return (Xi[0] < Xj[0]); }
bool compareY(vec3 Xi, vec3 Xj) { return (Xi[1] < Xj[1]); }
bool compareZ(vec3 Xi, vec3 Xj) { return (Xi[2] < Xj[2]); }

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  const int numBodies = 1000000;
  const int mpisize = 1000;
  double tic, toc;
  tic = get_time();
  int * index = new int [numBodies];
  Xvector xvector(numBodies);
  Bodies bodies(numBodies);
  Cells cells;
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  srand48(1);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    for (int d=0; d<3; d++) B->X[d] = drand48();
  }
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  tic = get_time();
  X_iter X = xvector.begin();
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++, X++) {
    X[0] = B->X[0];
    X[1] = B->X[1];
    X[2] = B->X[2];
  }
  toc = get_time();
  std::cout << "copy : " << toc-tic << std::endl;

  tic = get_time();
  int level = 0;
  int size = mpisize - 1;
  while (size > 0) {
    size >>= 1;
    level++;
  }
  index[0] = 0;
  index[1] = xvector.size()/2;
  index[2] = xvector.size();
  for (int l=0; l<level; l++) {
    std::cout << l << std::endl;
    X_iter X0 = xvector.begin();
    if (l % 3 == 0) {
      for (int i=0; i<(1 << l); i++) {
        std::nth_element(X0+index[2*i],X0+index[2*i+1],X0+index[2*i+2],compareX);
      }
    } else if( l % 3 == 1 ) {
      for (int i=0; i<(1 << l); i++) {
        std::nth_element(X0+index[2*i],X0+index[2*i+1],X0+index[2*i+2],compareY);
      }
    } else {
      for (int i=0; i<(1 << l); i++) {
        std::nth_element(X0+index[2*i],X0+index[2*i+1],X0+index[2*i+2],compareZ);
      }
    }
    for (int i=(2 << l); i>0; i--) {
      index[2*i] = index[i];
    }
    for (int i=(2 << l); i>0; i--) {
      index[2*i-1] = (index[2*i-2] + index[2*i]) / 2;
    }
  }
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;

  tic = get_time();
  for (int i=0; i<(1 << level); i++) {
    for (int b=index[2*i]; b<index[2*i+2]; b++) {
      bodies[b].X[0] = xvector[b][0];
      bodies[b].X[1] = xvector[b][1];
      bodies[b].X[2] = xvector[b][2];
      bodies[b].IBODY = i;
    }
  }
  toc = get_time();
  std::cout << "set B: " << toc-tic << std::endl;
  delete[] index;
}
