#ifndef updownpass_h
#define updownpass_h
#include "kernel.h"
#include "logger.h"

class UpDownPass : public Kernel {
 public:
  real_t theta;                                                 //!< Multipole acceptance criteria

 private:
//! Recursive call for upward pass
  void postOrderTraversal(Cell * C) {
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) postOrderTraversal(C->CHILD[i]);         //  Recursive call with new task
    }                                                           // End loop over child cells
    C->M = 0;                                                   // Initialize multipole expansion coefficients
    C->L = 0;                                                   // Initialize local expansion coefficients
    if (C->NCHILD == 0) P2M(C);                                 // P2M kernel
    M2M(C);                                                     // M2M kernel
  }

//! Recursive call for downward pass
  void preOrderTraversal(Cell * C) const {
    L2L(C);                                                     // L2L kernel
    if (C->NNODE == 1) L2P(C);                                  // L2P kernel
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) preOrderTraversal(C->CHILD[i]);          //  Recursive call with new task
    }                                                           // End loop over chlid cells
  }

 public:
  UpDownPass(real_t _theta) : Kernel(), theta(_theta) {}

//! Upward pass (P2M, M2M)
  void upwardPass(Cells &cells) {
    startTimer("Upward pass");                                  // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      Cell * C0 = &cells[0];                                    //  Set iterator of target root cell
      postOrderTraversal(C0);                                   //  Recursive call for upward pass
    }                                                           // End if for empty cell vector
    stopTimer("Upward pass");                                   // Stop timer
  }

//! Downward pass (L2L, L2P)
  void downwardPass(Cells &cells) {
    startTimer("Downward pass");                                // Start timer
    if (!cells.empty()) {                                       // If cell vector is not empty
      Cell * C0 = &cells[0];                                    //  Root cell
      preOrderTraversal(C0);                                    //  Recursive call for downward pass
    }                                                           // End if for empty cell vector
    stopTimer("Downward pass");                                 // Stop timer
  }
};
#endif
