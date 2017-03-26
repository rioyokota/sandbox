#ifndef traversal_h
#define traversal_h
#include "types.h"

namespace exafmm {
  int images;                                                   //!< Number of periodic image sublevels
  real_t theta;                                                 //!< Multipole acceptance criterion
  real_t Xperiodic[2];                                          //!< Periodic coordinate offset

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
    for (int d=0; d<2; d++) dX[d] = Ci->X[d] - Cj->X[d] - Xperiodic[d];// Distance vector from source to target
    real_t R2 = (dX[0] * dX[0] + dX[1] * dX[1]) * theta * theta;// Scalar distance squared
    if (R2 > (Ci->R + Cj->R) * (Ci->R + Cj->R)) {               // If distance is far enough
      M2L(Ci, Cj, Xperiodic);                                   //  Use approximate kernels
    } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {            // Else if both cells are leafs
      P2P(Ci, Cj, Xperiodic);                                   //   Use exact kernel
    } else if (Cj->NCHILD == 0 || Ci->R >= Cj->R) {             // Else if Cj is leaf or Ci is larger
      for (Cell * ci=Ci->CHILD; ci!=Ci->CHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
        traversal(ci, Cj);                                      //   Traverse a single pair of cells
      }                                                         //  End loop over Ci's children
    } else {                                                    // Else if Ci is leaf or Cj is larger
      for (Cell * cj=Cj->CHILD; cj!=Cj->CHILD+Cj->NCHILD; cj++) {//  Loop over Cj's children
        traversal(Ci, cj);                                      //  Traverse a single pair of cells
      }                                                         //  End loop over Cj's children
    }                                                           // End if for leafs and Ci Cj size
  }

  //! Tree traversal of periodic cells
  void traversePeriodic(Cell * Ci0, Cell * Cj0, real_t cycle) {
    for (int d=0; d<2; d++) Xperiodic[d] = 0;                   // Periodic coordinate offset
    Cell * Cp = new Cell();                                     // Last cell is periodic parent cell
    Cell * Cj = new Cell();                                     // Last cell is periodic parent cell
    Cp->M.resize(P, 0);                                         // Allocate and initialize multipole coefs
    Cj->M.resize(P, 0);                                         // Allocate and initialize multipole coefs
    *Cp = *Cj = *Cj0;                                           // Copy values from source root
    Cp->CHILD = Cj;                                             // Child cells for periodic center cell
    Cp->NCHILD = 1;                                             // Define only one child
    for (int level=0; level<images-1; level++) {                // Loop over sublevels of tree
      for (int ix=-1; ix<=1; ix++) {                            //  Loop over x periodic direction
        for (int iy=-1; iy<=1; iy++) {                          //   Loop over y periodic direction
          if (ix != 0 || iy != 0) {                             //    If periodic cell is not at center
            for (int cx=-1; cx<=1; cx++) {                      //     Loop over x periodic direction (child)
              for (int cy=-1; cy<=1; cy++) {                    //      Loop over y periodic direction (child)
                Xperiodic[0] = (ix * 3 + cx) * cycle;           //       Coordinate offset for x periodic direction
                Xperiodic[1] = (iy * 3 + cy) * cycle;           //       Coordinate offset for y periodic direction
                M2L(Ci0, Cp, Xperiodic);                        //       Perform M2L kernel
              }                                                 //      End loop over y periodic direction (child)
            }                                                   //     End loop over x periodic direction (child)
          }                                                     //    Endif for periodic center cell
        }                                                       //   End loop over y periodic direction
      }                                                         //  End loop over x periodic direction
      std::vector<complex_t> M(P);                              //  Multipole expansions
      for (int n=0; n<P; n++) {                                 //  Loop over order of expansions
        M[n] = Cp->M[n];                                        //   Save multipoles of periodic parent
        Cp->M[n] = 0;                                           //   Reset multipoles of periodic parent
      }                                                         //  End loop over order of expansions
      for (int ix=-1; ix<=1; ix++) {                            //  Loop over x periodic direction
        for (int iy=-1; iy<=1; iy++) {                          //   Loop over y periodic direction
          if( ix != 0 || iy != 0) {                             //    If periodic cell is not at center
            Cj->X[0] = Cp->X[0] + ix * cycle;                   //     Set new x coordinate for periodic image
            Cj->X[1] = Cp->X[1] + iy * cycle;                   //     Set new y cooridnate for periodic image
            for (int n=0; n<P; n++) Cj->M[n] = M[n];            //     Copy multipoles to new periodic image
            M2M(Cp);                                            //     Evaluate periodic M2M kernels for this sublevel
          }                                                     //    Endif for periodic center cell
        }                                                       //   End loop over y periodic direction
      }                                                         //  End loop over x periodic direction
      cycle *= 3;                                               //  Increase center cell size three times
    }                                                           // End loop over sublevels of tree
  }

  //! Evaluate P2P and M2L using dual tree traversal
  void traversal(Cell * Ci0, Cell * Cj0, real_t cycle) {
    if (images == 0) {                                          // If non-periodic boundary condition
      for (int d=0; d<2; d++) Xperiodic[d] = 0;                 //  No periodic shift
      traversal(Ci0, Cj0);                                      //  Traverse the tree
    } else {                                                    // If periodic boundary condition
      for (int ix=-1; ix<=1; ix++) {                            //  Loop over x periodic direction
        for (int iy=-1; iy<=1; iy++) {                          //   Loop over y periodic direction
          Xperiodic[0] = ix * cycle;                            //    Coordinate shift for x periodic direction
          Xperiodic[1] = iy * cycle;                            //    Coordinate shift for y periodic direction
          traversal(Ci0, Cj0);                                  //    Traverse the tree for this periodic image
        }                                                       //   End loop over y periodic direction
      }                                                         //  End loop over x periodic direction
      traversePeriodic(Ci0, Cj0, cycle);                        //  Traverse tree for periodic images
    }                                                           // End if for periodic boundary condition
  }                                                             // End if for empty cell vectors

  //! Recursive call for downward pass
  void downwardPass(Cell * Cj) {
    L2L(Cj);                                                    // L2L kernel
    if (Cj->NCHILD == 0) L2P(Cj);                               // L2P kernel
    for (Cell * Ci=Cj->CHILD; Ci!=Cj->CHILD+Cj->NCHILD; Ci++) { // Loop over child cells
      downwardPass(Ci);                                         //  Recursive call
    }                                                           // End loop over child cells
  }

  //! Direct summation
  void direct(int ni, Body * Bi, int nj, Body * Bj, real_t cycle) {
    Cell * Ci = new Cell();                                     // Allocate single target cell
    Cell * Cj = new Cell();                                     // Allocate single source cell
    Ci->BODY = Bi;                                              // Pointer of first target body
    Ci->NBODY = ni;                                             // Number of target bodies
    Cj->BODY = Bj;                                              // Pointer of first source body
    Cj->NBODY = nj;                                             // Number of source bodies
    int prange = 0;                                             // Range of periodic images
    for (int i=0; i<images; i++) {                              // Loop over periodic image sublevels
      prange += int(powf(3.,i));                                //  Accumulate range of periodic images
    }                                                           // End loop over perioidc image sublevels
    for (int ix=-prange; ix<=prange; ix++) {                    // Loop over x periodic direction
      for (int iy=-prange; iy<=prange; iy++) {                  //  Loop over y periodic direction
        Xperiodic[0] = ix * cycle;                              //   Coordinate shift for x periodic direction
        Xperiodic[1] = iy * cycle;                              //   Coordinate shift for y periodic direction
        P2P(Ci, Cj, Xperiodic);                                 //   Evaluate P2P kernel
      }                                                         //  End loop over y periodic direction
    }                                                           // End loop over x periodic direction
    delete Ci;                                                  // Deallocate target cell
    delete Cj;                                                  // Deallocate source cell
  }
}

#endif
