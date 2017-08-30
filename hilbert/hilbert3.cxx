#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include <string>
#include "vec.h"
#define HILBERT 1

typedef vec<3,int> ivec3;

uint64_t getKey(ivec3 iX, int levels) {
#if HILBERT
  int M = 1 << (levels - 1);
  for (int Q=M; Q>1; Q>>=1) {
    int P = Q - 1;
    for (int d=0; d<3; d++) {
      if (iX[d] & Q) iX[0] ^= P;
      else {
        int t = (iX[0] ^ iX[d]) & P;
        iX[0] ^= t;
        iX[d] ^= t;
      }
    }
  }
  for (int d=1; d<3; d++) iX[d] ^= iX[d-1];
  int t = 0;
  for (int Q=M; Q>1; Q>>=1)
    if (iX[2] & Q) t ^= Q - 1;
  for (int d=0; d<3; d++) iX[d] ^= t;
#endif
  uint64_t i = 0;
  for (int l=0; l<levels; l++) {
    i |= (iX[2] & (uint64_t)1 << l) << 2*l;
    i |= (iX[1] & (uint64_t)1 << l) << (2*l + 1);
    i |= (iX[0] & (uint64_t)1 << l) << (2*l + 2);
  }
  return i;
}

ivec3 get3DIndex(uint64_t i, int levels) {
  ivec3 iX = 0;
  for (int l=0; l<levels; l++) {
    iX[2] |= (i & (uint64_t)1 << 3*l) >> 2*l;
    iX[1] |= (i & (uint64_t)1 << (3*l + 1)) >> (2*l + 1);
    iX[0] |= (i & (uint64_t)1 << (3*l + 2)) >> (2*l + 2);
  }
#if HILBERT
  int N = 2 << (levels - 1);
  int t = iX[2] >> 1;
  for (int d=2; d>0; d--) iX[d] ^= iX[d-1];
  iX[0] ^= t;
  for (int Q=2; Q!=N; Q<<=1) {
    int P = Q - 1;
    for (int d=2; d>=0; d--) {
      if (iX[d] & Q) iX[0] ^= P;
      else {
        t = (iX[0] ^ iX[d]) & P;
        iX[0] ^= t;
        iX[d] ^= t;
      }
    }
  }
#endif
  return iX;
}

int main(int argc, char ** argv) {
  ivec3 iX;
  int levels = 21;
  iX[0] = atoi(argv[1]);
  iX[1] = atoi(argv[2]);
  iX[2] = atoi(argv[3]);
  uint64_t i = getKey(iX, levels);
  std::cout << i << std::endl;
  iX = get3DIndex(i, levels);
  std::cout << iX << std::endl;
}
