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
  double xip,yip,uip;
  xip = 0.5 + distance;
  yip = 0.5;
  double xjp,yjp,qjp;
  xjp = yjp = 0.5;
  // P2M
  qjp = 0;
  for (j=0; j<N; j++) {
    qjp += qj[j];
  }
  // M2L
  double dx, dy, r;
  dx = xip - xjp;
  dy = yip - yjp;
  r = sqrt(dx*dx+dy*dy);
  uip = qjp / r;
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
