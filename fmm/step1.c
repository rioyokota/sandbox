#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
  int i,j,N=10;
  double xi[N], yi[N], ui[N];
  double xj[N], yj[N], qj[N];
  double r = 2;
  for (i=0; i<N; i++) {
    xi[i] = r+drand48();
    yi[i] = drand48();
    ui[i] = 0;
    xj[i] = drand48();
    yj[i] = drand48();
    qj[i] = 1;
  }
  // P2M
  double M = 0;
  for (i=0; i<N; i++) {
    M += qj[i];
  }
  // M2L
  double L = M / r;
  // L2P
  for (i=0; i<N; i++) {
    ui[i] = L;
  }
  // Check answer
  for (i=0; i<N; i++) {
    double uid = 0;
    for (j=0; j<N; j++) {
      double dx = xi[i] - xj[j];
      double dy = yi[i] - yj[j];
      r = sqrt(dx*dx+dy*dy);
      uid += qj[j] / r;
    }
    printf("%lf %lf\n",ui[i],uid);
  }
}
