#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include <string>
#include "vec.h"

typedef vec<3,int> ivec3;

uint64_t getKey(ivec3 iX, int levels) {
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
