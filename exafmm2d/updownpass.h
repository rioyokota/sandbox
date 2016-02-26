#ifndef updownpass_h
#define updownpass_h
#include "kernel.h"
#include "logger.h"

class UpDownPass : public Kernel {
 public:
  real_t theta;                                                 //!< Multipole acceptance criteria

 public:
  UpDownPass(real_t _theta) : Kernel(), theta(_theta) {}

  //! Recursive call for upward pass
  void upwardPass(Cell * C) {
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) upwardPass(C->CHILD[i]);                 //  Recursive call with new task
    }                                                           // End loop over child cells
    C->M = 0;                                                   // Initialize multipole expansion coefficients
    C->L = 0;                                                   // Initialize local expansion coefficients
    if (C->NNODE == 1) P2M(C);                                  // P2M kernel
    M2M(C);                                                     // M2M kernel
  }

//! Recursive call for downward pass
  void downwardPass(Cell * C) const {
    L2L(C);                                                     // L2L kernel
    if (C->NNODE == 1) L2P(C);                                  // L2P kernel
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) downwardPass(C->CHILD[i]);               //  Recursive call with new task
    }                                                           // End loop over chlid cells
  }

};
#endif
