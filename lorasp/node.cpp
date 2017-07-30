#include "node.h"
#include "edge.h"
#include "tree.h"
#include <iostream>

// constructor for node
node::node( node* P, tree* T, int first, int last)
{
  index_first_ = first;
  index_last_ = last;
  tree_ptr_ = T;
  parent_ = P;
  eliminated_ = false;
  VAR_ = NULL;
  RHS_ = NULL;
}

node::~node ()
{
  if ( VAR_ != NULL )
    {
      delete VAR_;
    }

  if ( RHS_ != NULL )
    {
      delete RHS_;
    }
}

// Erase compressed edges from the list of incoming/outgoing edges
void node::eraseCompressedEdges()
{

  std::vector<edge*>::iterator it;

  // Erase compressed edges from incoming edges list
  for ( it = inEdges.begin(); it != inEdges.end(); )
    {
      if ( (*it) -> isCompressed() )
	{
	  it = inEdges.erase( it );
	}
      else
	{
	  ++it;
	}
    }

  // Erase compressed edges from outgoing edges list
  for ( it = outEdges.begin(); it != outEdges.end(); )
    {
      if ( (*it) -> isCompressed() )
	{
	  it = outEdges.erase( it );
	}
      else
	{
	  ++it;
	}
    }
  /* 
  // Erase compressed edges from incoming edges list
  for ( int i = inEdges.size()-1; i >= 0; i-- )
    {
      if ( inEdges[i] -> isCompressed() )
	{
	  inEdges.erase( inEdges.begin() + i );
	}
    }

  // Erase compressed edges from outgoing edges list
  for ( int i = outEdges.size()-1; i >= 0; i-- )
    {
      if ( outEdges[i] -> isCompressed() )
	{
	  outEdges.erase( outEdges.begin() + i );
	}
    }
  */
}


// Set the elimination flag to [true]
void node::eliminate()
{
  eliminated_ = true;
  order = tree_ptr_ -> count;
  tree_ptr_ -> count ++;
}


// Set the elimination flag to [false]
void node::deEliminate()
{
  eliminated_ = false;
}


// Solve U VAR = RHS
void node::solveU()
{
  // Check if this node has any unknowns
  if ( n() == 0 )
    {
      return;
    }
  
  // A vecotr to store the effect of all other incoming edges
  VectorXd potential = VectorXd::Zero( n() );
  
  // Go through all incoming edges
  for ( unsigned int i = 0; i < inEdges.size(); i++ )
    {
      // Check if source of the edge is already solved
      if ( inEdges[i]->source()->order > order )
	{
	  potential += (*(inEdges[i]->matrix)) * (*(inEdges[i]->source()->VAR()));
	}
    }
  
  // solve unknowns
  *VAR() = ( *invPivot ) * ( *RHS()  - potential );
}

// Solve L * z = b (the result is stored in RHS)
void node::solveL()
{
  // Check if this node has any unknowns
  if ( n() == 0 )
    {
      return;
    }
  // For RHS update: invPivot * RHS
  VectorXd factor_RHS = (*invPivot) * (*( RHS() ) );
  edge* Y;
  // Loop over all outgoing edges ( exclude the selfEdge ):Y
  for ( unsigned int j = 0; j < outEdges.size(); j++ )
    {
      
      Y = outEdges[j];
      if ( Y->destination()->order > order )
	{
	  // Update RHS_Y_Dest.
	  *( Y -> destination() -> RHS() ) -= (*(Y->matrix)) * factor_RHS;
	}
    }  
}

