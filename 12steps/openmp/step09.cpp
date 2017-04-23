#include <cstdio>
int main() {
#pragma omp parallel for ordered schedule(dynamic)
  for (int i=0; i<100; i+=5) {
#pragma omp ordered
    printf("%d\n",i);
  }
}
