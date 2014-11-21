#pragma once
#include "types.h"

void swap(int & a, int & b) {
  int c(a); a = b; b = c;
}

uint64_t morton(ivec3 iX, int maxLevel) {
  int mask = 1 << (maxLevel - 1);
  uint64_t key = 0;
  for (int i=0; i<maxLevel; i++) {
    int ix = (iX.x & mask) ? 1 : 0;
    int iy = (iX.y & mask) ? 1 : 0;
    int iz = (iX.z & mask) ? 1 : 0;
    int octant = (ix << 2) + (iy << 1) + iz;
    key = (key << 3) + octant;
    mask >>= 1;
  }
  return key;
}

uint64_t hilbert(ivec3 iX, int maxLevel) {
  const int octantMap[8] = {0, 1, 7, 6, 3, 2, 4, 5};
  int mask = 1 << (maxLevel - 1);
  uint64_t key = 0;
  for (int i=0; i<maxLevel; i++) {
    int ix = (iX.x & mask) ? 1 : 0;
    int iy = (iX.y & mask) ? 1 : 0;
    int iz = (iX.z & mask) ? 1 : 0;
    int octant = (ix << 2) + (iy << 1) + iz;
    if (octant == 1) {
      swap(iX,y, iX.z);
    } else if(octant == 1 || octant == 5) {
      swap(iX.x, iX.y);
    } else if(octant == 4 || octant == 6){
      iX.x = (iX.x) ^ (-1);
      iX.z = (iX.z) ^ (-1);
    } else if(octant == 3 || octant == 7) {
      iX.x = (iX.x) ^ (-1);
      iX.y = (iX.y) ^ (-1);
      swap(iX.x, iX.y);
    } else {
      iX.y = (iX.y) ^ (-1);
      iX.z = (iX.z) ^ (-1);
      swap(iX.y, iX.z);
    }
    key = (key << 3) + octantMap[octant];
    mask >>= 1;
  }
  return key;
}

Uint64s getKeys(Bodies & bodies, Bounds bounds, int maxLevel) {
  Uint64s keys(bodies.size());
  for (int b=0; b<int(bodies.size()); b++) {
    real_t dx = 2 * bounds.R / (1 << maxLevel);
    ivec3 iX = int((bodies[b].X - bounds.Xmin) / dx);
    keys[b] = hilbert(iX, maxLevel);
  }
  return keys;
}
