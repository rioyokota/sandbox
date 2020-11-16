#include <cstdio>

extern "C" int vecsum_(int *v, int &sum) {
  for(int i=0; i<9; i++)
    sum += v[i];
}
