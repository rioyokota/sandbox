#include "blackNode.h"
#include "redNode.h"
#include "superNode.h"
#include "edge.h"
#include "scotch.h"
#include "Eigen/Sparse"
#include <iostream>
#include "tree.h"
#include "time.h"


// constructor for blackNode
blackNode::blackNode( redNode* P, tree* T, int first, int last): node( (node*) P, T , first, last )
{
  // Set its parent
  parent_ = P;

  // Not elminated initially
  eliminated_ = false;

  // We set m, and n based on number of columns/rows corresponding to this redNode.
  // However, for non-leaf redNodes, m and n will be changed after low-rank approximation
  m( last - first + 1 );
  n( last - first + 1 );

  // Block width
  int BlockWidth = last - first + 1;

  // If this is an epmty node, we just create two empty children, and a super child
  if ( isEmpty() )
    {
      leftChild_ = new redNode( this, Tree(), 0, first, last );
      rightChild_ = new redNode( this, Tree(), 1, first, last );
      superChild_ = new superNode( this, Tree(), first, last );
      return;
    }

  // Get the Inner/Outer list for the corresponding block in symbolic matrix
  // Note that the symbolic matrix has zero diagonal (i.e., no self edge!)
  spMatBool SubMatrix = T->SymbMatrix()->block( first , first , BlockWidth, BlockWidth );
  

  /**************************************************************************/
  /*                            SCOTCH PARTITIONING                         */
  /**************************************************************************/

  /********** Initializing SCOTCH VARIABLES **********/
  
  // The first index in arrays is 0
  SCOTCH_Num baseval = 0;

  // Number of vertices = number of rows
  SCOTCH_Num vertnbr = SubMatrix.rows();

  // Note that non-self edges count twice
  SCOTCH_Num edgenbr = SubMatrix.nonZeros();

  // No 
  bool memAlloc = false;

  //! Array of start indices in edgetab   
  SCOTCH_Num*  verttab;
  
  //! Adjacency array of every vertex
  SCOTCH_Num*  edgetab;
  
  // See SCOTCH user_guid to read more about SCOTCH_Num
  if ( sizeof(SCOTCH_Num) == sizeof(int) )
    {
      verttab = SubMatrix.outerIndexPtr();
      edgetab = SubMatrix.innerIndexPtr();
    }
  else
    {
      std::cout<<"This happened!"<<std::endl;
      verttab = new SCOTCH_Num [vertnbr + 1];
      edgetab = new SCOTCH_Num [edgenbr];
      for ( int i = 0; i <= vertnbr; i++ )
	{
	  verttab[i] = SubMatrix.outerIndexPtr()[i];
	}
      for ( int i = 0; i < edgenbr; i++ )
	{
	  edgetab[i] = SubMatrix.innerIndexPtr()[i];
	}
      memAlloc = true;
    }
  
  /********** Initializing SCOTCH PARTITIONER **********/
  
  // Pointer to SCOTCH GRAPJ
  SCOTCH_Graph* graphPtr;

  // Define SCOTCH graph with error handling
  graphPtr = SCOTCH_graphAlloc();
  if ( graphPtr == NULL )
    {
      std::cout<<"Error! Could not allocate graph."<<std::endl;
      exit(EXIT_FAILURE);
    }

  // Initiate SCOTCH graph with error handling
  if ( SCOTCH_graphInit(graphPtr) != 0 )
    {
      std::cout<<"Error! Could not initialize graph."<<std::endl;
      exit(EXIT_FAILURE);
    }
  
  // Define SCOTCH graph with error handling
  if ( SCOTCH_graphBuild( graphPtr, baseval, vertnbr, verttab,		\
			  verttab + 1, NULL,NULL, edgenbr, edgetab, NULL) !=0 )
    {
      std::cout<<"Error! Failed to build graph."<<std::endl;
      exit(EXIT_FAILURE);
    }
  
  if (SCOTCH_graphCheck(graphPtr) !=0)
    {
      std::cout<<"Error! Graph inconsistent."<<std::endl;
      exit(EXIT_FAILURE);
    }

  // Initialize partitioning strategy
  SCOTCH_Strat* partStratPtr = SCOTCH_stratAlloc() ;
  if(SCOTCH_stratInit(partStratPtr) != 0)
    {
      std::cout<<"Error! Could not initialize partitioning strategy."<<std::endl;
      exit(EXIT_FAILURE);
  }
  
  // Partition graph
  SCOTCH_Num* parttab = new SCOTCH_Num[BlockWidth];
  if(SCOTCH_graphPart(graphPtr, 2, partStratPtr, parttab) !=0 )
    {
      std::cout<<"Error! Partitioning Failed."<<std::endl;
      exit(EXIT_FAILURE);
    }

  /*********** Modify the permutation vector of this level ************/
  // Find size of the first partion
  int size1 = 0;
  for ( int i = 0; i < BlockWidth; i++ )
    {
      if ( parttab[i] == 0 ) size1++;
    }

  // now, modify the global permutation vecotr
  int i1 = 0; // counter over the first partion
  int i2 = 0; // counter over the second partion 
  for ( int i = 0; i < BlockWidth; i++)
    {
      if ( parttab[i] == 0 )
	{
	  (*(T->permutationVector))(first+i) = first + i1;
	  i1++;
	}
      else
	{
	  (*(T->permutationVector))(first+i) = first + size1 + i2;
	  i2++;
	}

    }

  // printing the partitioning on the screen
  /*
  for ( int i = 0; i < BlockWidth; i++)
    std::cout<<i<<"->"<<parttab[i]<<std::endl;
  */
  
  /******** Now creating redNode children **********/
  // Note that at this point the global matrix has not permuted yet
  // However, the range of cols/row of children is known :)
  leftChild_ = new redNode( this, Tree(), 0, first, first+size1-1 );
  rightChild_ = new redNode( this, Tree(), 1, first+size1, last );

  // Here we create the super child (i.e., the combination of the left and right red children)
  // Later, we will actually create the corresponding edges of the super child.
  superChild_ = new superNode( this, Tree(), first, last );
  
  // If used any extra memoty here we clean it heare
  SCOTCH_graphExit(graphPtr);
  delete[] parttab;
  if ( memAlloc )
    {
      delete verttab;
      delete edgetab;
    }

}

// Destructor for blackNode: delete its Children
blackNode::~blackNode()
{
  delete leftChild_;
  delete rightChild_;
  delete superChild_;

  // Delete interaction edges
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      delete outEdges[i];
    }
}

redNode* blackNode::redParent( int l )
{
  // cross-recursive
  
  if ( l == 0 ) 
    {
      return parent_;
    }
  else
    {
      return parent_->redParent(l);
    }
}

// Merge left and right redNodes to a superNode
// Note that here we assume all redChildren have edges only within the same level
double blackNode::mergeChildren()
{
  clock_t start, finish;
  start = clock();
  // Set the number of variables/equations for the superChild
  superChild_ -> m( leftChild_->m() + rightChild_->m() );
  superChild_ -> n( leftChild_->n() + rightChild_->n() );
   // Check if any of the children has any variable
  if ( superChild_ -> m() == 0 )
    {
      finish = clock();
      return double(finish-start)/CLOCKS_PER_SEC;
    }

  // Define a Hash table:
  // key: parent(blackNode) value: pointer to the interaction matrices (LL,RL,LR,RR)
  std::map <blackNode*, densMatStr4> interMat;

  // Go over outgoing edges of left and right children, and add their parents to the interactSet
  
  // Pointer to a redNode that has interaction with one of the children
  redNode* interactant;
  
  // Parent of a redNode that has interaction with one of the children
  blackNode* blackParent;

  // A temporary variable to create the 2x2 block of matrices(pointers)
  densMatStr4 blocks;
  
  // Left child
  // First remove previosuly compressed edges from the list
  leftChild_ -> eraseCompressedEdges();

  for ( unsigned int i = 0; i < leftChild_->outEdges.size(); i++ ) 
    {
      // Check if the edge is not eliminated
      if ( leftChild_ -> outEdges[i] -> isEliminated() == false )
	{
	  // The redNode child I am interacting with
	  interactant = (redNode*) leftChild_->outEdges[i]->destination();
	  
	  // Parent of the interactant
	  blackParent = interactant->parent();
	  
	  if ( interMat.count(blackParent) == 0 ) // i.e., has not been observed so far
	    {
	      // Initialyy all blocks are NULL
	      blocks.fill(NULL);
	    }
	  
	  else // i.e., if we have already observed an interaction with a child of this blackParent
	    {
	      // get the previous block
	      blocks = interMat[ blackParent ];
	    }
	  
	  // Depending on which(destination) fill either second or fourth block
	  blocks[ 2 * interactant->which() ] = leftChild_->outEdges[i]->matrix;
	  
	  // Update/Add the block in the Hash table
	  interMat[ blackParent ] = blocks;
	}
    }

  // Right child
  // First remove previosuly compressed edges from the list
  rightChild_ -> eraseCompressedEdges();

  for ( unsigned int i = 0; i < rightChild_->outEdges.size(); i++ ) 
    {
      // Check if the edge is not eliminated
      if ( rightChild_ -> outEdges[i] -> isEliminated() == false )
	{
	  // The redNode child I am interacting with
	  interactant = (redNode*) rightChild_->outEdges[i]->destination();
	  
	  // Parent of the interactant
	  blackParent = interactant->parent();
	  
	  if ( interMat.count(blackParent) == 0 ) // i.e., has not been observed so far
	    {
	      // Initialyy all blocks are NULL
	      blocks.fill(NULL);
	    }
	  
	  else // i.e., if we have already observed an interaction with a child of this blackParent
	    {
	      // get the previous block
	      blocks = interMat[ blackParent ];
	    }
	  
	  // Depending on which(destination) fill either first or third block      
	  blocks[ 2 * interactant->which() + 1 ] = rightChild_->outEdges[i]->matrix;
      
	  // Update/Add the block in the Hash table
	  interMat[ blackParent ] = blocks;
	}
    }

  
  // Now the Hash table is complete, we need to create combined matrices
  // Go through all interactant, and create the corresponding edge
  
  // Iterator to loop over interMat
  std::map <blackNode*, densMatStr4>::iterator it;
  
  for ( it = interMat.begin(); it != interMat.end(); ++it )
    {
      // Creating the edge and add it to in/out edges list of two superChildren
      superChild_ -> outEdges.push_back( new edge(superChild_, it->first->superChild()) );
      it -> first -> superChild() -> inEdges.push_back( superChild_->outEdges.back() );
      
      
      // Define the block matrices:
      densMat topLeft( it -> first -> leftChild() -> m() , leftChild_ -> n() );
      densMat topRight( it -> first -> leftChild() -> m() , rightChild_ -> n() );
      densMat botLeft( it -> first -> rightChild() -> m() , leftChild_ -> n() );
      densMat botRight( it -> first -> rightChild() -> m() , rightChild_ -> n() );
			

      // Use the hash table and fill the block (zeros if no interaction)
      topLeft = (it->second)[0] != NULL ? *((it->second)[0]) : densMat::Zero(topLeft.rows(), topLeft.cols());
      topRight = (it->second)[1] != NULL ? *((it->second)[1]) : densMat::Zero(topRight.rows(), topRight.cols());
      botLeft = (it->second)[2] != NULL ? *((it->second)[2]) : densMat::Zero(botLeft.rows(), botLeft.cols());
      botRight = (it->second)[3] != NULL ? *((it->second)[3]) : densMat::Zero(botRight.rows(), botRight.cols());

      
      // Number of equations in destination
      int mDestination = it -> first -> leftChild() -> m() + it -> first -> rightChild() -> m();
      
      // Creating the combined interaction matrix
      superChild_ -> outEdges.back() -> matrix = new densMat( mDestination, superChild_->n() );
      
      // Fill the combined matrix with blocks using Eigen comma initializer      
      // *(superChild_ -> outEdges.back() -> matrix) << topLeft, topRight, botLeft, botRight;
      if ( topLeft.rows() * topLeft.cols() > 0 ) //i.e., if this is actually a finite block
	{
	  superChild_ -> outEdges.back() -> matrix -> block( 0, 0, topLeft.rows(), topLeft.cols() ) << topLeft;
	}
      if ( topRight.rows() * topRight.cols() > 0 ) //i.e., if this is actually a finite block
	{	 
	  superChild_ -> outEdges.back() -> matrix -> block( 0, topLeft.cols(), topRight.rows(), topRight.cols() ) << topRight;
	}
      if ( botLeft.rows() * botLeft.cols() > 0 ) //i.e., if this is actually a finite block
	{
	  superChild_ -> outEdges.back() -> matrix -> block( topLeft.rows(), 0, botLeft.rows(), botLeft.cols() ) << botLeft;
	}
      if ( botRight.rows() * botRight.cols() > 0 ) //i.e., if this is actually a finite block
	{
	  superChild_ -> outEdges.back() -> matrix -> block( topLeft.rows(), topLeft.cols(), botRight.rows(), botRight.cols() ) << botRight;
	}

    }

  // Delete the matrix from non-eliminated outgoing edges from red-children:

  // Left child:
  for ( unsigned int i = 0; i < leftChild_->outEdges.size(); i++ ) 
    {
      // Check if the edge is not eliminated
      if ( leftChild_ -> outEdges[i] -> isEliminated() == false )
	{
	  delete leftChild_ -> outEdges[i] -> matrix;
	}
    }

  // Rightt child:
  for ( unsigned int i = 0; i < rightChild_->outEdges.size(); i++ ) 
    {
      // Check if the edge is not eliminated
      if ( rightChild_ -> outEdges[i] -> isEliminated() == false )
	{
	  delete rightChild_ -> outEdges[i] -> matrix;
	}
    }

  // Allocate memory for RHS and VAR of the superChild
  superChild_ -> RHS( new VectorXd( superChild_ -> m() ) );
  superChild_ -> VAR( new VectorXd( superChild_ -> n() ) );

  mergeRHS();
  
  finish = clock();
  return double(finish-start)/CLOCKS_PER_SEC;
}

void blackNode::mergeRHS()
{
  // Set the RHS
  if ( leftChild() -> m() > 0 ) // i.e., if the left child corresponds to any equation
    {
      superChild_ -> RHS() -> segment( 0, leftChild() -> m() ) = *( leftChild() -> RHS() );
    }
  if ( rightChild() -> m() > 0 ) // i.e., if the right child corresponds to any equation
    {
      superChild_ -> RHS() -> segment( leftChild() -> m(), rightChild() -> m() ) = *( rightChild() -> RHS() );
    }
  //*( superChild_ -> RHS() ) << *( leftChild() -> RHS() ) , *( rightChild() -> RHS() );
}


// Apply the schur-complement
timeTuple2 blackNode::schurComp()
{
  timeTuple2 times;
  times[0] = 0;
  times[1] = 0;
  clock_t start, finish;
  start = clock();
  
  // std::cout<<" I am a black node and have "<<inEdges.size()<<" inEdges and "<<outEdges.size()<<" outEdges, and n = "<<n()<<std::endl;

  // Step 0: Check if this node has any variable
  if ( n() == 0 )
    {
      eliminate();
      finish = clock();
      times[1] += double(finish-start)/CLOCKS_PER_SEC;
      return times;
    }
    
  // Create the edges to the redParent
  // Note that based on the algorithm the associated matrices are just -Identity
  // Note that m = n = m_parent = n_parent (all equals to k, the low rank apprx.)
  outEdges.push_back( new edge( this, parent() ) );
  parent() -> inEdges.push_back( outEdges.back() );
  outEdges.back() -> matrix = new densMat( m(), n() );
  *( outEdges.back() -> matrix ) = - ( densMat::Identity( m(), n() ) );
  
  inEdges.push_back( new edge( parent(), this ) );
  parent() -> outEdges.push_back( inEdges.back() );
  inEdges.back() -> matrix = new densMat( m(), n() );
  *( inEdges.back() -> matrix ) = - ( densMat::Identity( m(), n() ) );

  // Now, we just repeat the elimination algorithm, same as the one for superNode

  // First we detect the selfEdge (i.e., the edge from supernode to itself)
  edge* selfEdge = NULL;
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      if ( outEdges[i] -> destination() == this )
	{
	  selfEdge = outEdges[i];
	  break;
	}
    }

  
  // Now compute the inverse of the selfEdge matrix (i.e., inverse of the pivot)
  invPivot = new densMat( m(), n() );
  
  finish = clock();
  times[1] += double(finish-start)/CLOCKS_PER_SEC;
  start = clock();
  
  *invPivot = selfEdge -> matrix -> inverse();

  finish = clock();
  times[0] += double(finish-start)/CLOCKS_PER_SEC;
  start = clock();

  if ( invPivot->norm() > 1e9 )
    {
      // std::cout<<"(blackNode) normInvPivot = "<< invPivot->norm()<<" \n";
      Tree()->largePivot = true;
    }

  if ( selfEdge->matrix->norm() > 1e9 )
    {
      // std::cout<<"(blackNode) normPivot = "<< selfEdge->matrix->norm() <<" \n";
    }

  edge* X;
  edge* Y;
  // Loop over all incoming edges ( exclude the selfEdge ): X
  for ( unsigned int i = 0; i < inEdges.size(); i++ )
    {

      X = inEdges[i];
      if ( ( X -> source() != this ) && !( X -> isEliminated() ) )
	{

	  // At this point we can compute invPivot * X
	  densMat factor = (*invPivot) * (*(X->matrix));
	  
	    
	  // Loop over all outgoing edges ( exclude the selfEdge ):Y
	  for ( unsigned int j = 0; j < outEdges.size(); j++ )
	    {
	      
	      Y = outEdges[j];
	      if ( ( Y -> destination() != this ) && !( Y -> isEliminated() ) )
		{
		  
		  // Here we have ei -> selfEdge -> ej
		  // After elimination we have: i->j  with matrix = -X * invPivot * Y
		  // We first check if the edge from i to j already exist or not.
		  // if exist we just add the fill-in, otherwise create a new edge
		  edge* fillinEdge = NULL;
		  for ( unsigned int k = 0; k < X->source()->outEdges.size(); k++ )
		    {
		      if ( X->source()->outEdges[k]->destination() == Y->destination() )
			{
			  fillinEdge = X->source()->outEdges[k];
			  break;
			}
		    }
		  if ( fillinEdge == NULL )
		    {
		      X->source()->outEdges.push_back( new edge( X->source(), Y->destination() ) );
		      Y->destination()->inEdges.push_back( X->source()->outEdges.back() );
		      X->source()->outEdges.back()->matrix = new densMat( Y->matrix->rows(), X->matrix->cols() );
		      *( X->source()->outEdges.back()->matrix ) = -(*(Y->matrix)) * factor;			
		    }
		  else
		    {
		      *(fillinEdge -> matrix) -= (*(Y->matrix)) * factor;
		    }
		}
	    }
	}
    }
  
  // Now, mark the node as eliminated
  eliminate();
  
  finish = clock();
  times[1] += double(finish-start)/CLOCKS_PER_SEC;
  return times;
}
