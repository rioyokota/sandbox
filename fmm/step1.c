#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
  int i,j,N=10;
  double xi[N], yi[N], ui[N];
  double xj[N], yj[N], qj[N];
  double distance = 2;
  for (i=0; i<N; i++) {
    xi[i] = distance+drand48();
    yi[i] = drand48();
    ui[i] = 0;
    xj[i] = drand48();
    yj[i] = drand48();
    qj[i] = 1;
  }
  double xip = 0.5 + distance;
  double yip = 0.5;
  double xjp = 0.5;
  double yjp = 0.5;
  // P2M
  double qjp = 0;
  for (j=0; j<N; j++) {
    qjp += qj[j];
  }
  // M2L
  double dx = xip - xjp;
  double dy = yip - yjp;
  double r = sqrt(dx*dx+dy*dy);
  double uip = qjp / r;
  // L2P
  for (i=0; i<N; i++) {
    ui[i] = uip;
  }
  // Check answer
  for (i=0; i<N; i++) {
    double uid = 0;
    for (j=0; j<N; j++) {
      dx = xi[i] - xj[j];
      dy = yi[i] - yj[j];
      r = sqrt(dx*dx+dy*dy);
      uid += qj[j] / r;
    }
    printf("%lf %lf\n",ui[i],uid);
  }
}
