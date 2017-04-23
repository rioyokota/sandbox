#include <cstdio>
int main() {
  int x = 1;
#pragma omp task shared(x) depend(out: x)
  x = 2;
#pragma omp task shared(x) depend(in: x)
  printf("x + 1 = %d\n", x+1);
#pragma omp task shared(x) depend(in: x)
  printf("x + 2 = %d\n", x+2);
}
