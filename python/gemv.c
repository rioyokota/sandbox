#include <stdio.h>

void gemv(double * a, int n) {
  int i;
  for (i=0; i<n; i++)
    printf("%d %lf\n",i,a[i]);
}
