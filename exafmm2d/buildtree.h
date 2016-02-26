#ifndef buildtree_h
#define buildtree_h
#include "logger.h"
#include "types.h"

class BuildTree {
 private:
  int ncrit;                                                    //!< Number of bodies per leaf cell

 private:
//! Build nodes of tree adaptively using a top-down approach based on recursion (uses task based thread parallelism)
  Node * buildNodes(Bodies& bodies, Bodies& buffer, B_iter B0, int begin, int end,
		    vec2 X, real_t R0, int level=0, bool direction=false) {
    if (begin == end) return NULL;                              // If no bodies left, return null pointer
    //! Create a tree node
    Node * node = new Node();                                   // Allocate memory for single node
    node->BODY = B0 + begin;                                    // Iterator of first body in node
    node->NBODY = end - begin;                                  // Number of bodies in node
    node->NNODE = 1;                                            // Initialize counter for decendant nodes
    node->X = X;                                                // Center position of node
    node->R = R0 / (1 << level);                                // Cell radius
    for (int i=0; i<4; i++) node->CHILD[i] = NULL;              //  Initialize pointers to children
    //! If node is a leaf
    if (end - begin <= ncrit) {                                 // If number of bodies is less than threshold
      if (direction)                                            //  If direction of data is from bodies to buffer
        for (int i=begin; i<end; i++) buffer[i] = bodies[i];    //   Copy bodies to buffer
      return node;                                              //  Return node pointer
    }                                                           // End if for number of bodies
    //! Count number of bodies in each quadrant 
    ivec4 size = 0;
    for (int i=begin; i<end; i++) {                             //  Loop over bodies in node
      vec2 x = bodies[i].X;                                     //   Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //   Which quadrant body belongs to
      size[quadrant]++;                                         //   Increment body count in quadrant
    }                                                           //  End loop over bodies in node
    //! Exclusive scan to get offsets
    int offset = begin;                                         // Offset of first quadrant
    ivec4 offsets;                                              // Output vector
    for (int i=0; i<4; i++) {                                   // Loop over elements
      offsets[i] = offset;                                      //  Set value
      offset += size[i];                                        //  Increment offset
    }                                                           // End loop over elements
    //! Sort bodies by quadrant
    ivec4 counter = offsets;
    for (int i=begin; i<end; i++) {                             //  Loop over bodies
      vec2 x = bodies[i].X;                                     //   Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //   Which quadrant body belongs to`
      buffer[counter[quadrant]] = bodies[i];                    //   Permute bodies out-of-place according to quadrant
      counter[quadrant]++;                                      //   Increment body count in quadrant
    }                                                           //  End loop over bodies
    //! Loop over children and recurse
    for (int i=0; i<4; i++) {                                   // Loop over children
      vec2 Xchild = X;                                          //   Initialize center position of child node
      real_t r = R0 / (1 << (level + 1));                       //   Radius of cells for child's level
      for (int d=0; d<2; d++) {                                 //   Loop over dimensions
	Xchild[d] += r * (((i & 1 << d) >> d) * 2 - 1);         //    Shift center position to that of child node
      }                                                         //   End loop over dimensions
      node->CHILD[i] = buildNodes(buffer, bodies, B0,           //   Recursive call for each child
				  offsets[i], offsets[i] + size[i],//   Range of bodies is calcuated from quadrant offset
				  Xchild, R0, level+1, !direction);//   Alternate copy direction bodies <-> buffer
    }                                                           // End loop over children
    //! Accumulate number of decendant nodes
    for (int i=0; i<4; i++) {                                   // Loop over children
      if (node->CHILD[i]) {                                     //  If child exists
	node->NNODE += node->CHILD[i]->NNODE;                   //   Increment child node counter
      }                                                         //  End if for child
    }                                                           // End loop over chlidren
    return node;                                                // Return quadtree node
  }

//! Create cell data structure from nodes
  void nodes2cells(Node * node, C_iter C, C_iter C0, C_iter CN, int level=0) {
    C->R = node->R;                                             // Cell radius
    C->X = node->X;                                             // Cell center
    C->BODY = node->BODY;                                       // Iterator of first body in cell
    C->NBODY = node->NBODY;                                     // Number of decendant bodies
    if (node->NNODE == 1) {                                     // If node has no children
      C->CHILD = &*C0;                                          //  Set index of first child cell to zero
      C->NCHILD = 0;                                            //  Number of child cells
      C->NNODE = node->NNODE;                                   //  Number of child cells
      C->NBODY = node->NBODY;                                   //  Number of bodies in cell
    } else {                                                    // Else if node has children
      int nchild = 0;                                           //  Initialize number of child cells
      int quadrants[4];                                         //  Map of child index to quadrants (for when nchild < 4)
      for (int i=0; i<4; i++) {                                 //  Loop over quadrants
        if (node->CHILD[i]) {                                   //   If child exists for that quadrant
          quadrants[nchild] = i;                                //    Map quadrant to child index
          nchild++;                                             //    Increment child cell counter
        }                                                       //   End if for child existance
      }                                                         //  End loop over quadrants
      C_iter Ci = CN;                                           //  CN points to the next free memory address
      C->CHILD = &*Ci;                                          //  Set Index of first child cell
      C->NCHILD = nchild;                                       //  Number of child cells
      C->NNODE = node->NNODE;                                   //  Number of child cells
      CN += nchild;                                             //  Increment next free memory address
      for (int i=0; i<nchild; i++) {                            //  Loop over children
	int quadrant = quadrants[i];                            //   Get quadrant from child index
	nodes2cells(node->CHILD[quadrant], Ci, C0, CN, level+1);// Recursive call for each child
	Ci++;                                                   //   Increment cell iterator
	CN += node->CHILD[quadrant]->NNODE - 1;                 //   Increment next free memory address
      }                                                         //  End loop over children
      for (int i=0; i<nchild; i++) {                            //  Loop over children
        int quadrant = quadrants[i];                            //   Get quadrant from child index
        delete node->CHILD[quadrant];                           //   Free child pointer to avoid memory leak
      }                                                         //  End loop over children
    }                                                           // End if for child existance
  }

  //! Transform Xmin & Xmax to X (center) & R (radius)
  Box bounds2box(Bounds bounds) {
    vec2 Xmin = bounds.Xmin;                                    // Set local Xmin
    vec2 Xmax = bounds.Xmax;                                    // Set local Xmax
    Box box;                                                    // Bounding box
    for (int d=0; d<2; d++) box.X[d] = (Xmax[d] + Xmin[d]) / 2; // Calculate center of domain
    box.R = 0;                                                  // Initialize localRadius
    for (int d=0; d<2; d++) {                                   // Loop over dimensions
      box.R = std::max(box.X[d] - Xmin[d], box.R);              //  Calculate min distance from center
      box.R = std::max(Xmax[d] - box.X[d], box.R);              //  Calculate max distance from center
    }                                                           // End loop over dimensions
    box.R *= 1.00001;                                           // Add some leeway to radius
    return box;                                                 // Return box.X and box.R
  }

 public:
  BuildTree(int _ncrit) : ncrit(_ncrit) {}

//! Build tree structure top down
  Cells buildTree(Bodies &bodies, Bounds bounds) {
    Box box = bounds2box(bounds);                               // Get box from bounds
    Cells cells;                                                // Initialize cell array
    Bodies buffer = bodies;                                     // Copy bodies to buffer
    startTimer("Grow tree");                                    // Start timer
    B_iter B0 = bodies.begin();                                 // Iterator of first body
    Node * N0 = buildNodes(bodies, buffer, B0, 0, bodies.size(), box.X, box.R);// Build tree recursively
    stopTimer("Grow tree");                                     // Stop timer
    startTimer("Link tree");                                    // Start timer
    cells.resize(N0->NNODE);                                    //  Allocate cells array
    C_iter C0 = cells.begin();                                  //  Cell begin iterator
    nodes2cells(N0, C0, C0, C0+1);                              //  Convert nodes to cells recursively
    delete N0;                                                  //  Deallocate nodes
    stopTimer("Link tree");                                     // Stop timer
    return cells;                                               // Return cells array
  }

};
#endif
