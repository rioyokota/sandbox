#include <iostream>

static int a = 1;
extern int b;
int c = 3;

void foo() {
  int d = 0;
  printf("%d\n",++d);
}
