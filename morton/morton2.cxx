#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include "vec.h"

typedef vec<2,int> ivec2;

uint64_t getKey(ivec2 iX, int levels) {
  uint64_t i = 0;
  for (int l=0; l<levels; l++) {
    i |= (iX[1] & (uint64_t)1 << l) << l;
    i |= (iX[0] & (uint64_t)1 << l) << (l + 1);
  }
  return i;
}

ivec2 get2DIndex(uint64_t i, int levels) {
  ivec2 iX = 0;
  for (int l=0; l<levels; l++) {
    iX[1] |= (i & (uint64_t)1 << 2*l) >> l;
    iX[0] |= (i & (uint64_t)1 << (2*l + 1)) >> (l + 1);
  }
  return iX;
}

int getLevel(uint64_t key) {
  int level = -1;
  uint64_t offset = 0;
  while (key >= offset) {
    level++;
    offset += (uint64_t)1 << 2 * level;
  }
  return level;
}

int main(int argc, char ** argv) {
  ivec2 iX;
  int levels = 31;
  iX[0] = atoi(argv[1]);
  iX[1] = atoi(argv[2]);
  uint64_t i = getKey(iX, levels);
  std::cout << i << std::endl;
  i += (((uint64_t)1 << 2 * levels) - 1) / 3;
  std::cout << getLevel(i) << std::endl;
  i -= (((uint64_t)1 << 2 * levels) - 1) / 3;
  iX = get2DIndex(i, levels);
  std::cout << iX << std::endl;
}
