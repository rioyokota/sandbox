#ifndef kernel_h
#define kernel_h
#include <cmath>
#include "types.h"

class Kernel {
 protected:
  vec2 Xperiodic;                                               //!< Coordinate offset for periodic B.C.

 public:
//!< Constructor
  Kernel() : Xperiodic(0) {}

//!< P2P kernel between cells Ci and Cj 
  void P2P(Cell * Ci, Cell * Cj) const {
    B_iter Bi = Ci->BODY;
    B_iter Bj = Cj->BODY;
    for (int i=0; i<Ci->NBODY; i++) {
      real_t pot = 0;
      for (int j=0; j<Cj->NBODY; j++) {
	vec2 dX = Bi[i].X - Bj[j].X - Xperiodic;
	real_t R2 = norm(dX);
	if (R2 != 0) {
	  real_t invR = 1 / sqrt(R2);
	  real_t logR = Bi[i].SRC * Bj[j].SRC * log(invR);
	  pot += logR;
	}
      }
      Bi[i].TRG += pot;
    }
  }

//!< P2M kernel for cell C
  void P2M(Cell * C) const {
    for (B_iter B=C->BODY; B!=C->BODY+C->NBODY; B++) {          // Loop over bodies
      vec2 dX = B->X - C->X;                                    //  Get distance vector
      complex_t Z(dX[0],dX[1]), powZ(1.0, 0.0);                 //  Convert to complex plane
      C->M[0] += B->SRC;                                        //  Add constant term
      for (int n=1; n<P; n++) {                                 //  Loop over coefficients
        powZ *= Z / real_t(n);                                  //   Store z^n / n!
        C->M[n] += powZ * B->SRC;                               //   Add to coefficient
      }                                                         //  End loop
    }                                                           // End loop
  }

//!< M2M kernel for one parent cell Ci
  void M2M(Cell * Ci) const {
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (Ci->CHILD[i]) {                                       //  If child exists
	Cell * Cj = Ci->CHILD[i];                               //   Child cell
	vec2 dX = Cj->X - Ci->X;                                //   Get distance vector
	complex_t Z(dX[0],dX[1]), powZn(1.0, 0.0),
	  powZnk(1.0, 0.0), invZ(powZn/Z);                      //   Convert to complex plane
	for (int k=0; k<P; k++) {                               //   Loop over coefficients
	  complex_t powZ(1.0, 0.0);                             //    z^0 = 1
	  Ci->M[k] += Cj->M[k];                                 //    Add constant term
	  for (int kml=1; kml<=k; kml++) {                      //    Loop over k-l
	    powZ *= Z / real_t(kml);                            //     Store z^(k-l) / (k-l)!
	    Ci->M[k] += Cj->M[k-kml] * powZ;                    //     Add to coefficient
	  }                                                     //    End loop
	}                                                       //   End loop
      }                                                         //  End loop
    }                                                           // End loop
  }

//!< M2L kernel between cells Ci and Cj
  void M2L(Cell * Ci, Cell * Cj) const {
    vec2 dX = Ci->X - Cj->X - Xperiodic;                        // Get distance vector
    complex_t Z(dX[0],dX[1]), powZn(1.0, 0.0),
      powZnk(1.0, 0.0), invZ(powZn/Z);                          // Convert to complex plane
    Ci->L[0] += -Cj->M[0] * log(Z);                             // Log term (for 0th order)
    Ci->L[0] += Cj->M[1] * invZ;                                // Constant term
    powZn = invZ;                                               // 1/z term
    for (int k=2; k<P; k++) {                                   // Loop over coefficients
      powZn *= real_t(k-1)*invZ;                                //  Store (k-1)! / z^k
      Ci->L[0] += Cj->M[k] * powZn;                             //  Add to coefficient
    }                                                           // End loop
    Ci->L[1] += -Cj->M[0] * invZ;                               // Constant term (for 1st order)
    powZn = invZ;                                               // 1/z term
    for (int k=1; k<P; k++) {                                   // Loop over coefficients
      powZn *= real_t(1+k-1)*invZ;                              //  Store (k)! / z^k
      Ci->L[1] += -Cj->M[k] * powZn;                            //  Add to coefficient
    }                                                           // End loop
    real_t Cnk = -1;                                            // Fix sign term
    for (int n=2; n<P; n++) {                                   // Loop over 
      Cnk *= -1;                                                //  Flip sign
      powZnk *= invZ;                                           //  Store 1 / z^n
      powZn = Cnk*powZnk;                                       //  Combine terms
      for (int k=0; k<P; k++) {                                 //  Loop over
	powZn *= real_t(n+k-1)*invZ;                            //   (n+k-1)! / z^k
	Ci->L[n] += Cj->M[k] * powZn;                           //   Add to coefficient
      }                                                         //  End loop
      powZnk *= real_t(n-1);                                    //  Store (n-1)! / z^n
    }                                                           // End loop
  }

//!< L2L kernel for one parent cell Cj
  void L2L(Cell * Cj) const {
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (Cj->CHILD[i]) {                                       //  If child exists
	Cell * Ci = Cj->CHILD[i];                               //   Child cell
	vec2 dX = Ci->X - Cj->X;                                //   Get distance vector
	complex_t Z(dX[0],dX[1]);                               //   Convert to complex plane
	for (int l=0; l<P; l++) {                               //   Loop over coefficients
	  complex_t powZ(1.0, 0.0);                             //    z^0 = 1
	  Ci->L[l] += Cj->L[l];                                 //    Add constant term
	  for (int k=1; k<P-l; k++) {                           //    Loop over coefficients
	    powZ *= Z / real_t(k);                              //     Store z^k / k!
	    Ci->L[l] += Cj->L[l+k] * powZ;                      //     Add to coefficient
	  }                                                     //    End loop
	}                                                       //   End loop
      }                                                         //  End loop
    }                                                           // End loop
  }

//!< L2P kernel for cell Ci
  void L2P(Cell * Ci) const {
    for (B_iter B=Ci->BODY; B!=Ci->BODY+Ci->NBODY; B++) {       // Loop over bodies
      vec2 dX = B->X - Ci->X;                                   //  Get distance vector
      complex_t Z(dX[0],dX[1]), powZ(1.0, 0.0);                 //  Convert to complex plane
      B->TRG /= B->SRC;                                         //  Normalize result
      B->TRG += std::real(Ci->L[0]);                            //  Add constant term
      for (int n=1; n<P; n++) {                                 //  Loop over coefficients
        powZ *= Z / real_t(n);                                  //   Store z^n / n!
        B->TRG += std::real(Ci->L[n] * powZ);                   //   Add real part to solution
      }                                                         //  End loop
    }                                                           // End loop
  }
};
#endif
