#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
  int i,j,N=10;
  double xi[N], yi[N], ui[N*16];
  double xj[N], yj[N], qj[N];
  for (i=0; i<N; i++) {
    xi[i] = drand48();
    yi[i] = drand48();
    xj[i] = drand48();
    yj[i] = drand48();
    qj[i] = 1;
  }
  for (i=0; i<16*N; i++) {
    ui[i] = 0;
  }
  double xip,yip,uip;
  xip = 0.5;
  yip = 0.5;
  double xjp,yjp,qjp;
  xjp = yjp = 0.5;
  // Step 1.
  qjp = 0;
  for (j=0; j<N; j++) {
    qjp += qj[j];
  }
  // Step 2.
  double dx, dy, r;
  int ix, iy, jx, jy;
  for (ix=0; ix<4; ix++) {
    for (iy=0; iy<4; iy++) {
      int ibox = (ix+iy*4)*N;
      uip = 0;
      for (jx=0; jx<4; jx++) {
	for (jy=0; jy<4; jy++) {
	  if (abs(ix-jx)<2&&
	      abs(iy-jy)<2) {
	    for (i=0; i<N; i++) {
	      for (j=0; j<N; j++) {
		dx = xi[i]+ix - xj[j]-jx;
		dy = yi[i]+iy - yj[j]-jy;
		r = sqrt(dx*dx+dy*dy);
		if(r!=0)
		  ui[i+ibox] += qj[j] / r;
	      }
	    }
	  } else {
	    // Step 2.
	    dx = ix - jx;
	    dy = iy - jy;
	    r = sqrt(dx*dx+dy*dy);
	    uip += qjp / r;
	  }	
	}
      }
      // Step 3.
      for (i=0; i<N; i++) {
	ui[i+ibox] += uip;
      }
    }
  }
  // Check answer
  for (ix=0; ix<4; ix++) {
    for (iy=0; iy<4; iy++) {
      int ibox = (ix+iy*4)*N;
      for (i=0; i<N; i++) {
	double uid = 0;
	for (jx=0; jx<4; jx++) {
	  for (jy=0; jy<4; jy++) {
	    for (j=0; j<N; j++) {
	      dx = xi[i]+ix - xj[j]-jx;
	      dy = yi[i]+iy - yj[j]-jy;
	      r = sqrt(dx*dx+dy*dy);
	      if (r != 0)
		uid += qj[j] / r;
	    }
	  }
	}
	printf("%lf %lf\n",ui[i+ibox],uid);
      }
    }
  }
  
}
