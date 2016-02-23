#ifndef buildtree_h
#define buildtree_h
#include "logger.h"
#include "types.h"

class BuildTree : public Logger {
 private:
  typedef vec<4,int> ivec4;                                     //!< Vector of 4 integer types

  struct Node {
    int NBODY;                                                  //!< Number of descendant bodies
    int NNODE;                                                  //!< Number of descendant nodes
    Node * CHILD[4];                                            //!< Pointer to child node
    B_iter BODY;                                                //!< Iterator for first body in node
    vec2 X;                                                     //!< Coordinate at center
  };

  int ncrit;                                                    //!< Number of bodies per leaf cell
  B_iter B0;                                                    //!< Iterator of first body

 private:
//! Exclusive scan with offset
  inline ivec4 exclusiveScan(ivec4 input, int offset) const {
    ivec4 output;                                               // Output vector
    for (int i=0; i<4; i++) {                                   // Loop over elements
      output[i] = offset;                                       //  Set value
      offset += input[i];                                        //  Increment offset
    }                                                           // End loop over elements
    return output;                                              // Return output vector
  }

//! Create an tree node
  Node * makeNode(int begin, int end, vec2 X) const {
    Node * node = new Node();                                   // Allocate memory for single node
    node->BODY = B0 + begin;                                    // Index of first body in node
    node->NBODY = end - begin;                                  // Number of bodies in node
    node->NNODE = 1;                                            // Initialize counter for decendant nodes
    node->X = X;                                                // Center position of node
    for (int i=0; i<4; i++) node->CHILD[i] = NULL;              //  Initialize pointers to children
    return node;                                                // Return node
  }

//! Build nodes of tree adaptively using a top-down approach based on recursion (uses task based thread parallelism)
  Node * buildNodes(Bodies& bodies, Bodies& buffer, int begin, int end,
		    vec2 X, real_t R0, int level=0, bool direction=false) {
    if (begin == end) return NULL;                              // If no bodies left, return null pointer
    if (end - begin <= ncrit) {                                 // If number of bodies is less than threshold
      if (direction)                                            //  If direction of data is from bodies to buffer
        for (int i=begin; i<end; i++) buffer[i] = bodies[i];    //   Copy bodies to buffer
      return makeNode(begin, end, X);                           //  Create an tree node and return it's pointer
    }                                                           // End if for number of bodies
    Node * node = makeNode(begin, end, X);                      // Create an tree node with child nodes
    ivec4 size = 0;
    for (int i=begin; i<end; i++) {                             //  Loop over bodies in node
      vec2 x = bodies[i].X;                                     //   Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //   Which quadrant body belongs to
      size[quadrant]++;                                         //   Increment body count in quadrant
    }                                                           //  End loop over bodies in node
    ivec4 offset = exclusiveScan(size, begin);                  // Exclusive scan to obtain offset from quadrant count
    ivec4 offset2 = offset;
    for (int i=begin; i<end; i++) {                             //  Loop over bodies
      vec2 x = bodies[i].X;                                     //   Position of body
      int quadrant = (x[0] > X[0]) + ((x[1] > X[1]) << 1);      //   Which quadrant body belongs to`
      buffer[offset2[quadrant]] = bodies[i];                    //   Permute bodies out-of-place according to quadrant
      offset2[quadrant]++;                                      //   Increment body count in quadrant
    }                                                           //  End loop over bodies
    for (int i=0; i<4; i++) {                                   // Loop over children
      vec2 Xchild = X;                                          //   Initialize center position of child node
      real_t r = R0 / (1 << (level + 1));                       //   Radius of cells for child's level
      for (int d=0; d<2; d++) {                                 //   Loop over dimensions
	Xchild[d] += r * (((i & 1 << d) >> d) * 2 - 1);         //    Shift center position to that of child node
      }                                                         //   End loop over dimensions
      node->CHILD[i] = buildNodes(buffer, bodies,               //   Recursive call for each child
				  offset[i], offset[i] + size[i],//   Range of bodies is calcuated from quadrant offset
				  Xchild, R0, level+1, !direction);//   Alternate copy direction bodies <-> buffer
    }                                                           // End loop over children
    for (int i=0; i<4; i++) {                                   // Loop over children
      if (node->CHILD[i]) {                                     //  If child exists
	node->NNODE += node->CHILD[i]->NNODE;                   //   Increment child node counter
      }                                                         //  End if for child
    }                                                           // End loop over chlidren
    return node;                                                // Return quadtree node
  }

//! Create cell data structure from nodes
  void nodes2cells(Node * node, C_iter C, C_iter C0, C_iter CN, real_t R0, int level=0, int iparent=0) {
    C->PARENT = iparent;                                        // Index of parent cell
    C->R = R0 / (1 << level);                                   // Cell radius
    C->X = node->X;                                             // Cell center
    C->BODY = node->BODY;                                       // Iterator of first body in cell
    C->NBODY = node->NBODY;                                     // Number of decendant bodies
    if (node->NNODE == 1) {                                     // If node has no children
      C->CHILD  = 0;                                            //  Set index of first child cell to zero
      C->NCHILD = 0;                                            //  Number of child cells
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
      C->CHILD = Ci - C0;                                       //  Set Index of first child cell
      C->NCHILD = nchild;                                       //  Number of child cells
      CN += nchild;                                             //  Increment next free memory address
      for (int i=0; i<nchild; i++) {                            //  Loop over children
	int quadrant = quadrants[i];                            //   Get quadrant from child index
	nodes2cells(node->CHILD[quadrant], Ci, C0, CN, R0, level+1, C-C0);// Recursive call for each child
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
    if (!bodies.empty()) {                                      // If bodies vector is empty
      Bodies buffer = bodies;                                   // Copy bodies to buffer
      startTimer("Grow tree");                                  // Start timer
      B0 = bodies.begin();                                      // Bodies iterator
      Node * N0 = buildNodes(bodies, buffer, 0, bodies.size(), box.X, box.R);// Build tree recursively
      stopTimer("Grow tree");                                   // Stop timer
      startTimer("Link tree");                                  // Start timer
      cells.resize(N0->NNODE);                                  //  Allocate cells array
      C_iter C0 = cells.begin();                                //  Cell begin iterator
      nodes2cells(N0, C0, C0, C0+1, box.R);                     //  Convert nodes to cells recursively
      delete N0;                                                //  Deallocate nodes
      stopTimer("Link tree");                                   // Stop timer
    }                                                           // End if for empty root
    return cells;                                               // Return cells array
  }

};
#endif
