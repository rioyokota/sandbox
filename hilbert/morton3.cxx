#include <iostream>
#include <cstdlib>
#include <stdint.h>

int main(int argc, char ** argv) {
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  int z = atoi(argv[3]);
  uint64_t i = 0;
  for (int b=0; b<21; b++) {
    i |= (z & (uint64_t)1 << b) << 2*b;
    i |= (y & (uint64_t)1 << b) << (2*b + 1);
    i |= (x & (uint64_t)1 << b) << (2*b + 2);
  }
  std::cout << i << std::endl;
  int X = 0, Y = 0, Z = 0;
  for (int b=0; b<21; b++) {
    Z |= (i & (uint64_t)1 << 3*b) >> 2*b;
    Y |= (i & (uint64_t)1 << (3*b + 1)) >> (2*b + 1);
    X |= (i & (uint64_t)1 << (3*b + 2)) >> (2*b + 2);
  }
  std::cout << X << " " << Y << " " << Z << std::endl;
}
