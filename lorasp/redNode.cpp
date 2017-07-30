#include "redNode.h"
#include "blackNode.h"
#include "tree.h"
#include "edge.h"

// constructor for redNode
redNode::redNode( blackNode* P, tree* T, bool W, int first, int last): node( (node*) P, T, first, last )
{
  
  // Set its parent
  parent_ = P;

  // Set if it is left or right child
  which_ = W;

  // Set its child to Null (later, an actual blacknode may be assigned)
  child_ = NULL;

  // Set its level
  if ( P == NULL ) 
    {
      level_ = 0;
    }
  else
    {
      level_ = P->parent()->level()+1;
    }

  // When a redNode born, it should be added to the redNode list of the tree
  T->addRedNode(level_,this);

  // We set m, and n based on number of columns/rows corresponding to this redNode.
  // However, for non-leaf redNodes, m and n will be changed after low-rank approximation
  m( last - first + 1 );
  n( last - first + 1 );

  // Initially not eliminated
  eliminated_ = false;
  
}

// Destructor for redNode: delete its blackChild (if any)
redNode::~redNode()
{
  if ( child_ != NULL )
    {
      delete child_;
    }
  
  // Delete interaction edges
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      delete outEdges[i];
    }
}

// Continue creating the tree (Red -> Black)
void redNode::createBlackNode()
{
  child_ = new blackNode( this, Tree(), IndexFirst(), IndexLast() );
}

redNode* redNode::redParent( int l )
{
  // recursive implementation :D

  if ( l == 0 ) //base
    {
      return this;
    }
  else
    {
      if ( level_ == 0 )
	{
	  return this;
	}
      else
	{
	  return parent_->parent()->redParent( l-1 );
	}
    }
}
