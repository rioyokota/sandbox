#ifndef traversal_h
#define traversal_h
#include "types.h"

namespace exafmm {
  real_t theta;                                                 //!< Multipole acceptance criterion

  //! Recursive call for upward pass
  void upwardPass(Cell * C) {
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) upwardPass(C->CHILD[i]);                 //  Recursive call with new task
    }                                                           // End loop over child cells
    C->M.resize(P, 0);                                          // Allocate and initialize multipole coefs
    C->L.resize(P, 0);                                          // Allocate and initialize local coefs
    if (C->NNODE == 1) P2M(C);                                  // P2M kernel
    M2M(C);                                                     // M2M kernel
  }

  //! Dual tree traversal for a single pair of cells
  void traversal(Cell * Ci, Cell * Cj) {
    real_t dX[2];                                               // Distance vector
    for (int d=0; d<2; d++) dX[d] = Ci->X[d] - Cj->X[d];        // Distance vector from source to target
    real_t R2 = (dX[0] * dX[0] + dX[1] * dX[1]) * theta * theta;// Scalar distance squared
    if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {               // If distance is far enough
      M2L(Ci, Cj);                                              //  Use approximate kernels
    } else if (Ci->NNODE == 1 && Cj->NNODE == 1) {              // Else if both cells are bodies
      P2P(Ci, Cj);                                              //   Use exact kernel
    } else if (Cj->NNODE == 1 || Ci->R >= Cj->R) {              // Else if Cj is leaf or Ci is larger
      for (int i=0; i<4; i++) {                                 //  Loop over Ci's children
        if (Ci->CHILD[i]) traversal(Ci->CHILD[i], Cj);          //   Traverse a single pair of cells
      }                                                         //  End loop over Ci's children
    } else {                                                    // Else if Ci is leaf or Cj is larger
      for (int i=0; i<4; i++) {                                 //  Loop over Cj's children
        if (Cj->CHILD[i]) traversal(Ci, Cj->CHILD[i]);          //   Traverse a single pair of cells
      }                                                         //  End loop over Cj's children
    }                                                           // End if for leafs and Ci Cj size
  }

  //! Recursive call for downward pass
  void downwardPass(Cell * C) {
    L2L(C);                                                     // L2L kernel
    if (C->NNODE == 1) L2P(C);                                  // L2P kernel
    for (int i=0; i<4; i++) {                                   // Loop over child cells
      if (C->CHILD[i]) downwardPass(C->CHILD[i]);               //  Recursive call with new task
    }                                                           // End loop over child cells
  }

  //! Direct summation
  void direct(int ni, Body * ibodies, int nj, Body * jbodies) {
    Cell * Ci = new Cell();                                     // Allocate single target cell
    Cell * Cj = new Cell();                                     // Allocate single source cell
    Ci->BODY = ibodies;                                         // Pointer of first target body
    Ci->NBODY = ni;                                             // Number of target bodies
    Cj->BODY = jbodies;                                         // Pointer of first source body
    Cj->NBODY = nj;                                             // Number of source bodies
    P2P(Ci, Cj);                                                // Evaluate P2P kernel
  }
}

#endif
