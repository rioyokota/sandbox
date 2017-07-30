#include "edge.h"
#include "redNode.h"

// Function to determine if the edge is between two well-separated nodes
bool edge::isWellSeparated()
{

  // we use the function redParent here
  // A and B are well separated if their redParents are not adjacent!
  int   levelo = 0;
  return !( source_->redParent( levelo )->AdjList()->count( destination_->redParent( levelo ) ) ); // This is the main criterion.
 


}

// Constructor
edge::edge( node* s, node* d)
{
  
  // load source/des. pointers
  source_ = s;
  destination_ = d;

  // Initially the edge is not compressed
  compressed_ = false;  

}


// Destructor
edge::~edge()
{
  delete matrix;
}

// Returns [true] if any of the source or dest. nodes are eliminated
bool edge::isEliminated()
{
  if ( ( source_ -> isEliminated() ) || ( destination_ -> isEliminated() ) )
    {
      return true;
    }
  else
    {
      return false;
    }
}

// Compress the edge. i.e.:
// Set the compressed flag to [true]
// Deallocate the matrix memory
void edge::compress()
{
  
  // Set the flag to true
  compressed_ = true;

  // Free the matrix memory
  delete matrix;

}

