#ifndef traversal_h
#define traversal_h
#include "kernel.h"
#include "logger.h"
#include "thread.h"

class Traversal : public Kernel, public Logger {
 private:
  int nspawn;                                                   //!< Threshold of NBODY for spawning new threads
  int images;                                                   //!< Number of periodic image sublevels
  real_t theta;                                                 //!< Multipole acceptance criterion
  C_iter Ci0;                                                   //!< Begin iterator for target cells
  C_iter Cj0;                                                   //!< Begin iterator for source cells

//! Dual tree traversal for a range of Ci and Cj
  void traverse(C_iter CiBegin, C_iter CiEnd, C_iter CjBegin, C_iter CjEnd) {
    if (CiEnd - CiBegin == 1 || CjEnd - CjBegin == 1) {         // If only one cell in range
      if (CiBegin == CjBegin) {                                 //  If Ci == Cj
        assert(CiEnd == CjEnd);
        traverse(CiBegin, CjBegin);                             //   Call traverse for single pair
      } else {                                                  //  If Ci != Cj
        for (C_iter Ci=CiBegin; Ci!=CiEnd; Ci++) {              //   Loop over all Ci cells
          for (C_iter Cj=CjBegin; Cj!=CjEnd; Cj++) {            //    Loop over all Cj cells
            traverse(Ci, Cj);                                   //     Call traverse for single pair
          }                                                     //    End loop over all Cj cells
        }                                                       //   End loop over all Ci cells
      }                                                         //  End if for Ci == Cj
    } else {                                                    // If many cells are in the range
      C_iter CiMid = CiBegin + (CiEnd - CiBegin) / 2;           //  Split range of Ci cells in half
      C_iter CjMid = CjBegin + (CjEnd - CjBegin) / 2;           //  Split range of Cj cells in half
      spawn_tasks {                                             //  Initialize task group
	spawn_task0(traverse(CiBegin, CiMid, CjBegin, CjMid));  //  Spawn Ci:former Cj:former
	traverse(CiMid, CiEnd, CjMid, CjEnd);                   //  No spawn Ci:latter Cj:latter
	sync_tasks;                                             //  Synchronize task group
	spawn_task0(traverse(CiBegin, CiMid, CjMid, CjEnd));    //  Spawn Ci:former Cj:latter
	traverse(CiMid, CiEnd, CjBegin, CjMid);                 //  No spawn Ci:latter Cj:former
	sync_tasks;                                             //  Synchronize task group
      }
    }                                                           // End if for many cells in range
  }

//! Split cell and call traverse() recursively for child
  void splitCell(C_iter Ci, C_iter Cj) {
    if (Cj->NCHILD == 0) {                                      // If Cj is leaf
      assert(Ci->NCHILD > 0);                                   //  Make sure Ci is not leaf
      for (C_iter ci=Ci0+Ci->CHILD; ci!=Ci0+Ci->CHILD+Ci->NCHILD; ci++ ) {// Loop over Ci's children
        traverse(ci, Cj);                                       //   Traverse a single pair of cells
      }                                                         //  End loop over Ci's children
    } else if (Ci->NCHILD == 0) {                               // Else if Ci is leaf
      assert(Cj->NCHILD > 0);                                   //  Make sure Cj is not leaf
      for (C_iter cj=Cj0+Cj->CHILD; cj!=Cj0+Cj->CHILD+Cj->NCHILD; cj++ ) {// Loop over Cj's children
        traverse(Ci, cj);                                       //   Traverse a single pair of cells
      }                                                         //  End loop over Cj's children
    } else if (Ci->NBODY + Cj->NBODY >= nspawn) {               // Else if cells are still large
      traverse(Ci0+Ci->CHILD, Ci0+Ci->CHILD+Ci->NCHILD,         //  Traverse for range of cell pairs
               Cj0+Cj->CHILD, Cj0+Cj->CHILD+Cj->NCHILD);
    } else if (Ci->R >= Cj->R) {                                // Else if Ci is larger than Cj
      for (C_iter ci=Ci0+Ci->CHILD; ci!=Ci0+Ci->CHILD+Ci->NCHILD; ci++ ) {// Loop over Ci's children
        traverse(ci, Cj);                                       //   Traverse a single pair of cells
      }                                                         //  End loop over Ci's children
    } else {                                                    // Else if Cj is larger than Ci
      for (C_iter cj=Cj0+Cj->CHILD; cj!=Cj0+Cj->CHILD+Cj->NCHILD; cj++ ) {// Loop over Cj's children
        traverse(Ci, cj);                                       //   Traverse a single pair of cells
      }                                                         //  End loop over Cj's children
    }                                                           // End if for leafs and Ci Cj size
  }

//! Dual tree traversal for a single pair of cells
  void traverse(C_iter Ci, C_iter Cj) {
    vec2 dX = Ci->X - Cj->X - Xperiodic;                        // Distance vector from source to target
    real_t R2 = norm(dX) * theta * theta;                       // Scalar distance squared
    if (R2 > (Ci->R+Cj->R)*(Ci->R+Cj->R)) {                     //  If distance is far enough
      M2L(Ci, Cj);                                              //   Use approximate kernels
    } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {            //  Else if both cells are bodies
      P2P(Ci, Cj);                                              //    Use exact kernel
    } else {                                                    //  Else if cells are close but not bodies
      splitCell(Ci, Cj);                                        //   Split cell and call function recursively for child
    }                                                           //  End if for multipole acceptance
  }


//! Tree traversal of periodic cells
  void traversePeriodic(real_t cycle) {
    startTimer("Traverse periodic");                            // Start timer
    Xperiodic = 0;                                              // Periodic coordinate offset
    Cells pcells(6);                                            // Create cells
    C_iter Ci = pcells.end()-1;                                 // Last cell is periodic parent cell
    *Ci = *Cj0;                                                 // Copy values from source root
    Ci->CHILD = 0;                                              // Child cells for periodic center cell
    Ci->NCHILD = 8;                                             // Number of child cells for periodic center cell
    C_iter C0 = Cj0;                                            // Placeholder for Cj0
    for (int level=0; level<images-1; level++) {                // Loop over sublevels of tree
      for (int ix=-1; ix<=1; ix++) {                            //  Loop over x periodic direction
        for (int iy=-1; iy<=1; iy++) {                          //   Loop over y periodic direction
          if (ix != 0 || iy != 0) {                             //    If periodic cell is not at center
            for (int cx=-1; cx<=1; cx++) {                      //     Loop over x periodic direction (child)
              for (int cy=-1; cy<=1; cy++) {                    //      Loop over y periodic direction (child)
                Xperiodic[0] = (ix * 3 + cx) * cycle;           //       Coordinate offset for x periodic direction
                Xperiodic[1] = (iy * 3 + cy) * cycle;           //       Coordinate offset for y periodic direction
                M2L(Ci0, Ci);                                   //       Perform M2L kernel
              }                                                 //      End loop over y periodic direction (child)
            }                                                   //     End loop over x periodic direction (child)
          }                                                     //    Endif for periodic center cell
        }                                                       //   End loop over y periodic direction
      }                                                         //  End loop over x periodic direction
      Cj0 = pcells.begin();                                     //  Redefine Cj0 for M2M
      C_iter Cj = Cj0;                                          //  Iterator of periodic neighbor cells
      for (int ix=-1; ix<=1; ix++) {                            //  Loop over x periodic direction
        for (int iy=-1; iy<=1; iy++) {                          //   Loop over y periodic direction
          if( ix != 0 || iy != 0) {                             //    If periodic cell is not at center
            Cj->X[0] = Ci->X[0] + ix * cycle;                   //     Set new x coordinate for periodic image
            Cj->X[1] = Ci->X[1] + iy * cycle;                   //     Set new y cooridnate for periodic image
            Cj->M    = Ci->M;                                   //     Copy multipoles to new periodic image
            Cj++;                                               //     Increment periodic cell iterator
          }                                                     //    Endif for periodic center cell
        }                                                       //   End loop over y periodic direction
      }                                                         //  End loop over x periodic direction
      Ci->M = 0;                                                //  Reset multipoles of periodic parent
      M2M(Ci,Cj0);                                              //  Evaluate periodic M2M kernels for this sublevel
      cycle *= 3;                                               //  Increase center cell size three times
      Cj0 = C0;                                                 //  Reset Cj0 back
    }                                                           // End loop over sublevels of tree
    stopTimer("Traverse periodic");                             // Stop timer
  }

 public:
  Traversal(int nspawn, int images, real_t theta) : Kernel(0.0), nspawn(nspawn), images(images), theta(theta) {}

//! Evaluate P2P and M2L using dual tree traversal
  void dualTreeTraversal(Cells &icells, Cells &jcells, real_t cycle) {
    startTimer("Traverse");                                     // Start timer
    if (!icells.empty() && !jcells.empty()) {                   // If neither of the cell vectors are empty
      Ci0 = icells.begin();                                     //  Set iterator of target root cell
      Cj0 = jcells.begin();                                     //  Set iterator of source root cell
      if (images == 0) {                                        //  If non-periodic boundary condition
        Xperiodic = 0;                                          //   No periodic shift
        traverse(Ci0, Cj0);                                     //   Traverse the tree
      } else {                                                  //  If periodic boundary condition
        for (int ix=-1; ix<=1; ix++) {                          //   Loop over x periodic direction
          for (int iy=-1; iy<=1; iy++) {                        //    Loop over y periodic direction
            Xperiodic[0] = ix * cycle;                          //     Coordinate shift for x periodic direction
            Xperiodic[1] = iy * cycle;                          //     Coordinate shift for y periodic direction
            traverse(Ci0, Cj0);                                 //     Traverse the tree for this periodic image
          }                                                     //    End loop over y periodic direction
        }                                                       //   End loop over x periodic direction
        traversePeriodic(cycle);                                //   Traverse tree for periodic images
      }                                                         //  End if for periodic boundary condition
    }                                                           // End if for empty cell vectors
    stopTimer("Traverse");                                      // Stop timer
  }

  //! Direct summation
  void direct(Bodies &ibodies, Bodies &jbodies, real_t cycle) {
    Cells cells(2);                                             // Define a pair of cells to pass to P2P kernel
    C_iter Ci = cells.begin(), Cj = cells.begin()+1;            // First cell is target, second cell is source
    Ci->BODY = ibodies.begin();                                 // Iterator of first target body
    Ci->NBODY = ibodies.size();                                 // Number of target bodies
    Cj->BODY = jbodies.begin();                                 // Iterator of first source body
    Cj->NBODY = jbodies.size();                                 // Number of source bodies
    int prange = 0;                                             // Range of periodic images
    for (int i=0; i<images; i++) {                              // Loop over periodic image sublevels
      prange += int(std::pow(3.,i));                            //  Accumulate range of periodic images
    }                                                           // End loop over perioidc image sublevels
    for (int ix=-prange; ix<=prange; ix++) {                    // Loop over x periodic direction
      for (int iy=-prange; iy<=prange; iy++) {                  //  Loop over y periodic direction
        Xperiodic[0] = ix * cycle;                              //   Coordinate shift for x periodic direction
        Xperiodic[1] = iy * cycle;                              //   Coordinate shift for y periodic direction
        P2P(Ci, Cj);                                            //   Evaluate P2P kernel
      }                                                         //  End loop over y periodic direction
    }                                                           // End loop over x periodic direction
  }

//! Normalize bodies after direct summation
  void normalize(Bodies &bodies) {
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {       // Loop over bodies
      B->TRG /= B->SRC;                                         //  Normalize by target charge
    }                                                           // End loop over bodies
  }

};
#endif
