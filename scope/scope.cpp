#include "scope.h"

void bar() {
  static int e = 0;
  printf("%d\n",e++);
}
