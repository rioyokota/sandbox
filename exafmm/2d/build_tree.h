#ifndef buildtree_h
#define buildtree_h
#include "types.h"

namespace exafmm {
  int ncrit;                                                    //!< Number of bodies per leaf cell

  //! Build cells of tree adaptively using a top-down approach based on recursion
  void buildTree(Body * bodies, Body * buffer, int begin, int end, Cell * cell,
                 real_t X[2], real_t R, int level=0, bool direction=false) {
    //! Create a tree cell
    cell->BODY = bodies + begin;                                // Pointer of first body in cell
    if(direction) cell->BODY = buffer + begin;                  // Pointer of first body in cell
    cell->NBODY = end - begin;                                  // Number of bodies in cell
    cell->NCHILD = 0;                                           // Initialize counter for child cells
    for (int d=0; d<2; d++) cell->X[d] = X[d];                  // Center position of cell
    cell->R = R / (1 << level);                                 // Cell radius
    //! If cell is a leaf
    if (end - begin <= ncrit) {                                 // If number of bodies is less than threshold
      if (direction) {                                          //  If direction of data is from bodies to buffer
        for (int i=begin; i<end; i++) {                         //   Loop over bodies in cell
          for (int d=0; d<2; d++) buffer[i].X[d] = bodies[i].X[d];//  Copy bodies coordinates to buffer
          buffer[i].q = bodies[i].q;                            //    Copy bodies source to buffer
        }                                                       //   End loop over bodies in cell
      }                                                         //  End if for direction of data
      return;                                                   //  Return without recursion
    }                                                           // End if for number of bodies
    //! Count number of bodies in each quadrant
    int size[4] = {0,0,0,0};
    real_t x[2];                                                // Coordinates of bodies
    for (int i=begin; i<end; i++) {                             // Loop over bodies in cell
      for (int d=0; d<2; d++) x[d] = bodies[i].X[d];            //  Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //  Which quadrant body belongs to
      size[quadrant]++;                                         //  Increment body count in quadrant
    }                                                           // End loop over bodies in cell
    //! Exclusive scan to get offsets
    int offset = begin;                                         // Offset of first quadrant
    int offsets[4], counter[4];                                 // Offsets and counter for each quadrant
    for (int i=0; i<4; i++) {                                   // Loop over elements
      offsets[i] = offset;                                      //  Set value
      offset += size[i];                                        //  Increment offset
      if (size[i]) cell->NCHILD++;                              //  Increment child cell counter
    }                                                           // End loop over elements
    //! Sort bodies by quadrant
    for (int i=0; i<4; i++) counter[i] = offsets[i];            // Copy offsets to counter
    for (int i=begin; i<end; i++) {                             // Loop over bodies
      for (int d=0; d<2; d++) x[d] = bodies[i].X[d];            //  Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //  Which quadrant body belongs to`
      for (int d=0; d<2; d++) buffer[counter[quadrant]].X[d] = bodies[i].X[d];// Permute bodies coordinates out-of-place according to quadrant
      buffer[counter[quadrant]].q = bodies[i].q;                //  Permute bodies sources out-of-place according to quadrant
      counter[quadrant]++;                                      //  Increment body count in quadrant
    }                                                           // End loop over bodies
    //! Loop over children and recurse
    real_t Xchild[2];                                           // Coordinates of children
    Cell * child = new Cell[cell->NCHILD];                      // Allocate memory for child cells
    cell->CHILD = child;                                        // Pointer for first child cell
    int c = 0;                                                  // Counter for child cells
    for (int i=0; i<4; i++) {                                   // Loop over children
      for (int d=0; d<2; d++) Xchild[d] = X[d];                 //  Initialize center position of child cell
      real_t r = R / (1 << (level + 1));                        //  Radius of cells for child's level
      for (int d=0; d<2; d++) {                                 //  Loop over dimensions
        Xchild[d] += r * (((i & 1 << d) >> d) * 2 - 1);         //   Shift center position to that of child cell
      }                                                         //  End loop over dimensions
      if (size[i]) {                                            //  If child exists
        buildTree(buffer, bodies, offsets[i], offsets[i] + size[i],// Recursive call for each child
                  &child[c], Xchild, R, level+1, !direction);
        c++;                                                    //   Increment child cell counter
      }                                                         //  End if for child
    }                                                           // End loop over children
  }
}

#endif
