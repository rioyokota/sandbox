#ifndef updownpass_h
#define updownpass_h
#include "kernel.h"
#include "logger.h"

class UpDownPass : public Kernel, public Logger {
 public:
  real_t theta;                                                 //!< Multipole acceptance criteria

 private:
//! Recursive call for upward pass
  void postOrderTraversal(C_iter C, C_iter C0) {
    for (C_iter CC=C0+C->CHILD; CC!=C0+C->CHILD+C->NCHILD; CC++) {// Loop over child cells
      postOrderTraversal(CC, C0);                               //  Recursive call with new task
    }                                                           // End loop over child cells
    C->M = 0;                                                   // Initialize multipole expansion coefficients
    C->L = 0;                                                   // Initialize local expansion coefficients
    if (C->NCHILD == 0) P2M(C);                                 // P2M kernel
    M2M(C,C0);                                                  // M2M kernel
  }

//! Recursive call for downward pass
  void preOrderTraversal(C_iter C, C_iter C0) const {
    L2L(C,C0);                                                  // L2L kernel
    if (C->NCHILD == 0) L2P(C);                                 // L2P kernel
    for (C_iter CC=C0+C->CHILD; CC!=C0+C->CHILD+C->NCHILD; CC++) {// Loop over child cells
      preOrderTraversal(CC, C0);                                //  Recursive call with new task
    }                                                           // End loop over chlid cells
  }

 public:
  UpDownPass(real_t _theta) : Kernel(), theta(_theta) {}

//! Upward pass (P2M, M2M)
  void upwardPass(Cells &cells) {
    startTimer("Upward pass");                                  // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      C_iter C0 = cells.begin();                                //  Set iterator of target root cell
      postOrderTraversal(C0, C0);                               //  Recursive call for upward pass
    }                                                           // End if for empty cell vector
    stopTimer("Upward pass");                                   // Stop timer
  }

//! Downward pass (L2L, L2P)
  void downwardPass(Cells &cells) {
    startTimer("Downward pass");                                // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      C_iter C0 = cells.begin();                                //  Root cell
      if(C0->NCHILD==0) L2P(C0);                                //  If root is the only cell do L2P
      for (C_iter CC=C0+C0->CHILD; CC!=C0+C0->CHILD+C0->NCHILD; CC++) {// Loop over child cells
	preOrderTraversal(CC, C0);                              //    Recursive call for downward pass
      }                                                         //   End loop over child cells
    }                                                           // End if for empty cell vector
    stopTimer("Downward pass");                                 // Stop timer
  }
};
#endif
