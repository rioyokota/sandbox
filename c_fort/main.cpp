#include <cstdio>

extern "C" int sub_mp_vecsum_(int*);

int main() {
  int v[9] = {1,1,1,1,1,1,1,1,1};
  int sum = sub_mp_vecsum_(v);
  printf("%d\n",sum);
}
