#include "foo.h"

void bar() {
  static int c = 100;
  printf("c: %d\n",c++);
}
