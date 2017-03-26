#ifndef traversal_h
#define traversal_h
#include "types.h"

namespace exafmm {
  real_t theta;                                                 //!< Multipole acceptance criterion

  //! Recursive call for upward pass
  void upwardPass(Cell * Ci) {
    for (Cell * Cj=Ci->CHILD; Cj!=Ci->CHILD+Ci->NCHILD; Cj++) { // Loop over child cells
      upwardPass(Cj);                                           //  Recursive call
    }                                                           // End loop over child cells
    Ci->M.resize(P, 0);                                         // Allocate and initialize multipole coefs
    Ci->L.resize(P, 0);                                         // Allocate and initialize local coefs
    if (Ci->NCHILD == 0) P2M(Ci);                               // P2M kernel
    M2M(Ci);                                                    // M2M kernel
  }

  //! Dual tree traversal for a single pair of cells
  void traversal(Cell * Ci, Cell * Cj) {
    real_t dX[2];                                               // Distance vector
    for (int d=0; d<2; d++) dX[d] = Ci->X[d] - Cj->X[d];        // Distance vector from source to target
    real_t R2 = (dX[0] * dX[0] + dX[1] * dX[1]) * theta * theta;// Scalar distance squared
    if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {               // If distance is far enough
      M2L(Ci, Cj);                                              //  Use approximate kernels
    } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {            // Else if both cells are leafs
      P2P(Ci, Cj);                                              //   Use exact kernel
    } else if (Cj->NCHILD == 0 || Ci->R >= Cj->R) {             // Else if Cj is leaf or Ci is larger
      for (Cell * Cc=Ci->CHILD; Cc!=Ci->CHILD+Ci->NCHILD; Cc++) {// Loop over Ci's children
        traversal(Cc, Cj);                                      //   Traverse a single pair of cells
      }                                                         //  End loop over Ci's children
    } else {                                                    // Else if Ci is leaf or Cj is larger
      for (Cell * Cc=Cj->CHILD; Cc!=Cj->CHILD+Cj->NCHILD; Cc++) {//  Loop over Cj's children
        traversal(Ci, Cc);                                      //  Traverse a single pair of cells
      }                                                         //  End loop over Cj's children
    }                                                           // End if for leafs and Ci Cj size
  }

  //! Recursive call for downward pass
  void downwardPass(Cell * Cj) {
    L2L(Cj);                                                    // L2L kernel
    if (Cj->NCHILD == 0) L2P(Cj);                               // L2P kernel
    for (Cell * Ci=Cj->CHILD; Ci!=Cj->CHILD+Cj->NCHILD; Ci++) { // Loop over child cells
      downwardPass(Ci);                                         //  Recursive call
    }                                                           // End loop over child cells
  }

  //! Direct summation
  void direct(int ni, Body * Bi, int nj, Body * Bj) {
    Cell * Ci = new Cell();                                     // Allocate single target cell
    Cell * Cj = new Cell();                                     // Allocate single source cell
    Ci->BODY = Bi;                                              // Pointer of first target body
    Ci->NBODY = ni;                                             // Number of target bodies
    Cj->BODY = Bj;                                              // Pointer of first source body
    Cj->NBODY = nj;                                             // Number of source bodies
    P2P(Ci, Cj);                                                // Evaluate P2P kernel
    delete Ci;                                                  // Deallocate target cell
    delete Cj;                                                  // Deallocate source cell
  }
}

#endif
