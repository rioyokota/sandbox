#include <iostream>
#include <cstdlib>
#include <stdint.h>

int main(int argc, char ** argv) {
  int x = atoi(argv[1]);
  int y = atoi(argv[2]);
  uint64_t i = 0;
  for (int b=0; b<31; b++) {
    i |= (y & (uint64_t)1 << b) << b;
    i |= (x & (uint64_t)1 << b) << (b + 1);
  }
  std::cout << i << std::endl;
  int X = 0, Y = 0;
  for (int b=0; b<31; b++) {
    Y |= (i & (uint64_t)1 << 2*b) >> b;
    X |= (i & (uint64_t)1 << (2*b + 1)) >> (b + 1);
  }
  std::cout << X << " " << Y << std::endl;
}
