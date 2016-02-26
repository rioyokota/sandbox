#ifndef updownpass_h
#define updownpass_h
#include "kernel.h"
#include "logger.h"

class UpDownPass : public Kernel {
 public:
  real_t theta;                                                 //!< Multipole acceptance criteria

 private:
//! Recursive call for upward pass
  void postOrderTraversal(C_iter C) {
    for (C_iter CC=C->CHILD; CC!=C->CHILD+C->NCHILD; CC++) {    // Loop over child cells
      postOrderTraversal(CC);                                   //  Recursive call with new task
    }                                                           // End loop over child cells
    C->M = 0;                                                   // Initialize multipole expansion coefficients
    C->L = 0;                                                   // Initialize local expansion coefficients
    if (C->NCHILD == 0) P2M(C);                                 // P2M kernel
    M2M(C);                                                     // M2M kernel
  }

//! Recursive call for downward pass
  void preOrderTraversal(C_iter C) const {
    L2L(C);                                                     // L2L kernel
    if (C->NCHILD == 0) L2P(C);                                 // L2P kernel
    for (C_iter CC=C->CHILD; CC!=C->CHILD+C->NCHILD; CC++) {    // Loop over child cells
      preOrderTraversal(CC);                                    //  Recursive call with new task
    }                                                           // End loop over chlid cells
  }

 public:
  UpDownPass(real_t _theta) : Kernel(), theta(_theta) {}

//! Upward pass (P2M, M2M)
  void upwardPass(Cells &cells) {
    startTimer("Upward pass");                                  // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      C_iter C0 = cells.begin();                                //  Set iterator of target root cell
      postOrderTraversal(C0);                                   //  Recursive call for upward pass
    }                                                           // End if for empty cell vector
    stopTimer("Upward pass");                                   // Stop timer
  }

//! Downward pass (L2L, L2P)
  void downwardPass(Cells &cells) {
    startTimer("Downward pass");                                // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      C_iter C0 = cells.begin();                                //  Root cell
      preOrderTraversal(C0);                                    //  Recursive call for downward pass
    }                                                           // End if for empty cell vector
    stopTimer("Downward pass");                                 // Stop timer
  }
};
#endif
