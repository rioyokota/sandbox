#include <cstdio>
int counter = 0;
#pragma omp threadprivate(counter)

int main() {
#pragma omp parallel for
  for (int i=0; i<100; i++) {
    counter++;
  }
  printf("%d\n",counter);
}
