#include <iostream>

static int a = 0;
extern int b;

void foo() {
  printf("a: %d\n",a++);
  printf("b: %d\n",b++);
}
