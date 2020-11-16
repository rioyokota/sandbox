#include <cstdio>

extern "C" int vecsum_(int*);

int main() {
  int v[9] = {1,1,1,1,1,1,1,1,1};
  int sum = vecsum_(v);
  printf("%d\n",sum);
}
