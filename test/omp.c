#include <stdio.h>
#include <omp.h>

int main() {
#pragma omp parallel
  printf("%d\n",omp_get_num_threads());
}
