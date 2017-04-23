#include <cstdio>
int main() {
  float x[10];
  int index[1000];
  for (int i=0; i<1000; i++) {
    index[i] = i % 10;
  }
  for (int i=0; i<10; i++)
    x[i] = 0.0;
#pragma omp parallel for shared(x, index)
  for (int i=0; i<1000; i++) {
#pragma omp atomic update
    x[index[i]]++;
  }
  for (int i=0; i<10; i++)
    printf("%d %f\n",i, x[i]);
}
