#ifndef kernel_h
#define kernel_h
#include <cmath>
#include "types.h"

namespace kernel {
  void P2P(C_iter Ci, C_iter Cj, real_t eps2, vec3 Xperiodic, bool mutual) {
    B_iter Bi = Ci->BODY;
    B_iter Bj = Cj->BODY;
    int ni = Ci->NBODY;
    int nj = Cj->NBODY;
    for (int i=0; i<ni; i++) {
      real_t pot = 0; 
      real_t ax = 0;
      real_t ay = 0;
      real_t az = 0;
      for (int j=0; j<nj; j++) {
	vec3 dX = Bi[i].X - Bj[j].X - Xperiodic;
	real_t R2 = norm(dX) + eps2;
	if (R2 != 0) {
	  real_t invR2 = 1.0 / R2;
	  real_t invR = Bi[i].SRC * Bj[j].SRC * sqrt(invR2);
	  dX *= invR2 * invR;
	  pot += invR;
	  ax += dX[0];
	  ay += dX[1];
	  az += dX[2];
	  if (mutual) {
	    Bj[j].TRG[0] += invR;
	    Bj[j].TRG[1] += dX[0];
	    Bj[j].TRG[2] += dX[1];
	    Bj[j].TRG[3] += dX[2];
	  }
	}
      }
      Bi[i].TRG[0] += pot;
      Bi[i].TRG[1] -= ax;
      Bi[i].TRG[2] -= ay;
      Bi[i].TRG[3] -= az;
    }
  }
};
#endif
