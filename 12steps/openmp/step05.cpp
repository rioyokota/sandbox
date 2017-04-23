#include <cstdio>
#include <omp.h>
int main() {
  int section_count = 0;
  omp_set_dynamic(0);
  omp_set_num_threads(2);
#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
    {
      section_count++;
      printf("section_count %d\n",section_count);
    }
#pragma omp section
    {
      section_count++;
      printf("section_count %d\n",section_count);
    }
  }
}
