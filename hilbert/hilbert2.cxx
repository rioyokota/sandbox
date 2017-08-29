#include <iostream>
#include <cstdlib>
#include <stdint.h>

int main(int argc, char ** argv) {
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  int p = 31;
  int M = 1 << (p - 1);
  int Q = M;
  while (Q > 1) {
    int P = Q - 1;
    if (x & Q) {
      x ^= P;
    } else {
      x ^= 0;
    }
    if (y & Q) {
      x ^= P;
    } else {
      int t = (x ^ y) & P;
      x ^= t;
      y ^= t;
    }
    Q >>= 1;
  }
  y ^= x;
  int t = 0;
  Q = M;
  while (Q > 1) {
    if (y & Q) {
      t ^= Q - 1;
    }
    Q >>= 1;
  }
  x ^= t;
  y ^= t;
  uint64_t i = 0;
  for (int b=0; b<31; b++) {
    i |= (y & (uint64_t)1 << b) << b;
    i |= (x & (uint64_t)1 << b) << (b + 1);
  }
  std::cout << i << std::endl;
  /*
  int X = 0, Y = 0;
  for (int b=0; b<31; b++) {
    Y |= (i & (uint64_t)1 << 2*b) >> b;
    X |= (i & (uint64_t)1 << (2*b + 1)) >> (b + 1);
  }
  std::cout << X << " " << Y << std::endl;
  */
}
