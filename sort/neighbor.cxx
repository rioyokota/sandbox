#include <cstdlib>
#include <iostream>
#include "neighbor.h"
#include <sys/time.h>

double getTime() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  double tic = getTime();
  const int N = 1 << 25;
  Neighbor NE(N);
  double *X[3];
  for (int d=0; d<3; d++) {
    X[d] = new double [N];
    for (int i=0; i<N; i++) X[d][i] = drand48();
  }
  double Xi[3] = {0.5, 0.5, 0.5}, R = 0.05;
  double toc = getTime();
  std::cout << "Initialize : " << toc-tic << std::endl;

  tic = getTime();
  NE.getBounds(X);
  toc = getTime();
  std::cout << "Get bounds : " << toc-tic << std::endl;

  tic = getTime();
  NE.setKeys(X);
  toc = getTime();
  std::cout << "Set index  : " << toc-tic << std::endl;

  tic = getTime();
  NE.buffer(X);
  toc = getTime(); 
  std::cout << "Buffer     : " << toc-tic << std::endl;

  tic = getTime();
  NE.radixSort();
  toc = getTime(); 
  std::cout << "Radix sort : " << toc-tic << std::endl;

  tic = getTime();
  NE.permute(X);
  toc = getTime();
  std::cout << "Permute    : " << toc-tic << std::endl;

  tic = getTime();
  NE.setRange();
  toc = getTime();
  std::cout << "Set range  : " << toc-tic << std::endl;

  tic = getTime();
  NE.getNeighbor(Xi,X,R);
  toc = getTime();
  std::cout << "Neighbor   : " << toc-tic << std::endl;

  for (int d=0; d<3; d++) delete[] X[d];
}
